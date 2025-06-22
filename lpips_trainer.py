import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from accelerate import Accelerator

from modules import LPIPS, DiffToLogits
import lpips

class BAPPSDataset(Dataset):

    """
    Perceptual Dataset Loader

    Args:
        - path_to_root: Path to BAPPS Dataset Root
        - train: Training Split vs Validation Splits
        - dirs: Which directories do you want to load images?
            train: ["cnn", "mix", "traditional"]
            val: ["cnn", "color", "deblur", "frameinterp", "suprres", "traditional"]
        -img_size: What image size do you want to train on?

    We will be training on 64x64 images, as that is what the LPIPS paper does here
    https://github.com/richzhang/PerceptualSimilarity/blob/master/data/dataset/twoafc_dataset.py

    As far as I can tell, we can inference on any resolution we want later (its a convolution after all)
    and the model seems to be robust to resolution differences. So lets go with this for now!
    
    """

    def __init__(self,
                 path_to_root,
                 train=True,
                 dirs=None,
                 img_size=64):
        
        if train:
            split = "train"
            if dirs is None:
                dirs = ["cnn", "mix", "traditional"]

        else:          
            split = "val"
            if dirs is None:
                dirs = ["cnn", "color", "deblur", 
                        "frameinterp", "superres", "traditional"]
                
        if isinstance(dirs, str):
            dirs = [dirs]
 
        path_to_dirs = [os.path.join(path_to_root, split, dir) for dir in dirs]

        self.samples = self._generate_dataset(path_to_dirs)

        self.transforms = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)), 
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # scales [0,1] -> [-1,1]
            ]
        )

    def _generate_dataset(self, path_to_dirs):

        samples = []
        for dir in path_to_dirs:
            
            path_to_p0 = os.path.join(dir, "p0")
            path_to_p1 = os.path.join(dir, "p1")
            path_to_ref = os.path.join(dir, "ref")
            path_to_target = os.path.join(dir, "judge")

            file_idxs = [file.split(".")[0] for file in os.listdir(path_to_p0)]
            
            for idx in file_idxs:

                p0 = os.path.join(path_to_p0, f"{idx}.png")
                p1 = os.path.join(path_to_p1, f"{idx}.png")
                ref = os.path.join(path_to_ref, f"{idx}.png")
                target = os.path.join(path_to_target, f"{idx}.npy")

                samples.append((p0,p1,ref,target))

        return samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        img1, img2, ref, target = self.samples[idx]

        ### Load Perturbed Images and Original Reference ###
        img1 = self.transforms(Image.open(img1).convert('RGB'))
        img2 = self.transforms(Image.open(img2).convert('RGB'))
        ref = self.transforms(Image.open(ref).convert('RGB'))

        ### Load Labels ###
        target = np.load(target)[0]

        return img1, img2, ref, target


class LPIPSForTraining(nn.Module):

    """
    LPIPS + DiffToLogits Prediction Head

    Training LPIPS w/ VGG16 Backbone on Ranking Task on the BAPPS Dataset

    Args:
        - pretrained_backbone: Do you want to start with a Pretrained VGG16
        - train_backbone: Do you want gradient updated on the VGG16?
        - use_dropout: Do you want dropout layers before channel projections
        - img_range: Are your images [-1 to 1] or [0 to 1]
        - middle_channels: How many hidden channels in prediction head
    """

    def __init__(self,
                 pretrained_backbone=True,
                 train_backbone=False, 
                 use_dropout=True,
                 img_range="negative_one_to_one",
                 middle_channels=32):
        
        super(LPIPSForTraining, self).__init__()

        self.lpips = LPIPS(pretrained_backbone=pretrained_backbone, 
                           train_backbone=train_backbone, 
                           use_dropout=use_dropout, 
                           img_range=img_range)
        
        self.head = DiffToLogits(middle_channels=middle_channels)

    def bce_rank_loss(self, diff1, diff2, target):

        ### Compute Logits comparing the differences ###
        ### Outputs are in shape (B,1,1,1) ###
        outputs = self.head(diff1, diff2)

        ### Reshape Outputs to match targets for Loss ###
        outputs = outputs.reshape(*target.shape)

        ### Compute BCE Loss (labels between 0 and 1) ###
        loss = F.binary_cross_entropy_with_logits(input=outputs, target=target)

        return loss

    def clamp_weights(self):

        ### By enforcing non-negative weights on the linear layer w, LPIPS ensures that:
        ###     - Larger differences in feature activations consistently contribute to 
        ###       a larger distance
        ### If we have negative weights, then we could have a negative score
        ### Therefore we ensure that the contribution of each pixel along the channel dimension
        ### is either positive or zero (so we are positively accumulating differences)

        ### This can be found at the end of page 13 in the paper: https://arxiv.org/pdf/1801.03924

        for module in self.lpips.modules():
            if (hasattr(module, "weight") and module.kernel_size==(1,1)):
                module.weight.data = torch.clamp(module.weight.data, min=0)

    def checkpoint_model(self, path_to_checkpoint, checkpoint_name):
        
        path_to_lpips = os.path.join(path_to_checkpoint, checkpoint_name)

        ### We really only care about the lpips so just save that! ###
        print(f"Saving Checkpoint to {path_to_lpips}")
        torch.save(self.lpips.state_dict(), path_to_lpips)

    def forward(self, img1, img2, ref, target):

        ### Compute LPIPS Difference Between Each Perturbed Image and Ref ###
        diff1 = self.lpips(img1, ref)
        diff2 = self.lpips(img2, ref)

        ### Compute BCE Ranking Loss ###
        loss = self.bce_rank_loss(diff1, diff2, target)

        return loss, diff1, diff2

class LRScheduler:
    def __init__(self, optimizer, initial_lr, total_iterations, decay_iterations, min_lr=None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.total_iterations = total_iterations
        self.decay_iterations = decay_iterations
        self.constant_iterations = total_iterations - decay_iterations
        self.min_lr = min_lr if min_lr is not None else 0
        self.current_step = 0

    def step(self):

        if self.current_step < self.constant_iterations:
            lr = self.initial_lr
        else:
            decay_ratio = (self.current_step - self.constant_iterations) / self.decay_iterations
            lr = max(self.min_lr, self.initial_lr * (1 - decay_ratio))
        
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.current_step += 1
        
def compute_accuracy(diff1, diff2, target):
    
    """
    target = 0 -> diff1 (from image1) was tagged as perceptually closer to the refernce
    target = 1 -> diff2 (from image2) was tagged as perceptually closer to the reference
    target = 0.5 -> diff1 & diff2 are identically close to the reference

    Because we have the extra 0.5, we cannot calculate accuracy the normal way, but we can 
    use the workaround like the following. Lets say we have:

    diff2 < diff1: [True, True, False, True] -> A boolean vector checking, for every sample
                                                if diff2 is smaller than diff1 (and therefore)
                                                perceptually closer to the reference image

    We can first convert this to integer, and it acts as our predictions:

    preds: [1, 1, 0, 1]

    Lets also pretend our labels are the following -> targets: [1, 0, 0.5, 1]
    This means, in sample 1 and 4, image2 was tagged as perceptually closer, 
    in sample 2 image1 was tagged as perceptually closer, and lastly in sample 3, 
    both images are equal

    If we just did preds == targets, we will always count the 0.5 samples as wrong
    which isn't technically true, they are half rigth and half wrong. So the technique
    that the LPIPS implementation uses is just to say, things labled as 0.5, lets give 
    the model half-credit

    accuracy = preds * targets + (1 - preds) * (1 - targets)

    preds * targets -> [1, 1, 0, 1] * [1, 0, 0.5, 1] -> [1, 0, 0, 1]
    (1 - preds) * (1 - targets) -> [0, 0, 1, 0] * [0, 1, 0.5, 0] -> [0, 0, 0.5, 0]

    preds * targets + (1 - preds) * (1 - targets) -> [1, 0, 0, 1] + [0, 0, 0.5, 0] -> [1, 0, 0.5, 1]

    final_output: [1, 0, 0.5, 1]

    Notice the output is 1 (for correct) when the pred matches the target, 0 (for incorrect) when the 
    pred doesnt match the target, and finally 0.5 (half credit) for things that are labeled 0.5

    Lastly, we can average the tensor to get the percent accuracy:

    mean([1, 0, 0.5, 1]) -> 2.5/4 = 0.625

    """

    preds = (diff2 < diff1).flatten().int()
    target = target.flatten()
    accuracy = torch.mean(preds * target + (1 - preds) * (1 - target))

    return accuracy

def set_precision(args):

    dtype_dict = {"float32": torch.float32,
                  "float16": torch.float16, 
                  "bfloat16": torch.bfloat16}

    dtype = dtype_dict["float32"]

    if args.mixed_precision:
        device_properties = torch.cuda.get_device_properties(0).major
        
        if device_properties >= 8:
            print("Training with BFLOAT16")
            dtype = dtype_dict["bfloat16"]
        else:
            print("Training With FLOAT16")
            dtype = dtype_dict["float16"]

    return dtype

def trainer(args):

    ### Check if Working Directory Exists ###
    if not os.path.isdir(args.work_dir):
        os.makedirs(args.work_dir, exist_ok=True)

    ### Prepare DataLoaders ###
    train_set = BAPPSDataset(path_to_root=args.path_to_root, train=True, img_size=args.img_size)
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    ### Define Model ###
    model = LPIPSForTraining(pretrained_backbone=args.pretrained_backbone, 
                             train_backbone=args.train_backbone, 
                             use_dropout=args.use_dropout, 
                             img_range=args.img_range,
                             middle_channels=args.middle_channels)
    
    ### Define Optimizer ###
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

    ### Define LR Scheduler Decay ###
    scheduler = LRScheduler(optimizer=optimizer, 
                            initial_lr=args.initial_lr,
                            total_iterations=len(trainloader) * args.num_epochs,
                            decay_iterations=len(trainloader) * args.decay_epochs)

    ### Prepare Everything ###
    model, optimizer, trainloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, scheduler
    )
    
    total_training_iterations = len(trainloader) * args.num_epochs
    accelerator.print("TRAINING FOR {} ITERATIONS".format(total_training_iterations))

    ### Start Training ###
    iterations = 0

    while iterations < total_training_iterations:

        for batch in trainloader:
            
            ### Grab Image Options 1/2, the Reference Image, and the Target ###
            img1, img2, ref, target = (t.to(accelerator.device) for t in batch)

            ### Compute Loss and Store our Diffs from LPIPS ###]
            loss, diff1, diff2 = model(img1, img2, ref, target)

            accelerator.backward(loss)

            ### Clip Gradients ###
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            ### Update Model ###
            optimizer.step()
            optimizer.zero_grad()

            ### Clamp the Weights to be positive ###
            accelerator.unwrap_model(model).clamp_weights()

            ### Update Scheduler ###
            scheduler.step()

            ### Count Iterations ###
            iterations += 1

            if iterations % args.logging_steps == 0:

                accuracy = compute_accuracy(diff1, diff2, target)
                
                accuracy = torch.mean(accelerator.gather_for_metrics(accuracy)).item()
                loss = torch.mean(accelerator.gather_for_metrics(loss)).item()

                log = {"iteration": iterations, 
                       "loss": round(loss, 4), 
                       "accuracy": round(accuracy * 100, 2),
                       "lr": optimizer.param_groups[0]["lr"]}
                
                accelerator.print(log)
        
    ### Checkpoint Model ###
    accelerator.unwrap_model(model).checkpoint_model(path_to_checkpoint=args.work_dir, 
                                                     checkpoint_name=args.checkpoint_name)
    
def eval(args):

    accelerator.print("EVALUATING ON VALIDATION")

    ### Store Models to Evaluate ##
    models_to_eval = [] 
    
    ### Load Model ###
    my_lpips_model = LPIPS(pretrained_weights=os.path.join(args.work_dir, args.checkpoint_name)).eval()
    models_to_eval.append(("LPIPS Reproduction", my_lpips_model))

    ### Load LPIPS Package ###
    if args.eval_lpips_pkg:
        pkg_lpips_model = lpips.LPIPS(pretrained=True, net="vgg", verbose=False).eval()
        models_to_eval.append(("Original LPIPS", pkg_lpips_model))

    ### Loop Over Splits ###
    val_dataset_splits = ["cnn", "traditional", "color", "deblur", "frameinterp", "superres"]

    ### Loop Over Models to Evaluate ###
    for (name, model) in models_to_eval:
        
        accelerator.print("-----------")
        accelerator.print("Evaluating:", name)
        accelerator.print("-----------")

        model = model.to(accelerator.device)
        for split in val_dataset_splits:
            
            ### Load Dataset ###
            dataset = BAPPSDataset(path_to_root=args.path_to_root, 
                                img_size=args.img_size,
                                train=False, 
                                dirs=split)

            loader = DataLoader(dataset, batch_size=args.eval_batch_size, 
                                shuffle=False, num_workers=args.num_workers)

            accs = []
            for batch in loader:

                ### Grab Batch ###
                img1, img2, ref, target = (t.to(accelerator.device) for t in batch)

                ### Compute Diffs between Images and Refs ###
                with torch.no_grad():
                    diff1 = model(img1, ref)
                    diff2 = model(img2, ref)

                accs.append(compute_accuracy(diff1, diff2, target).item())

            accs = np.mean(accs)

            print(f"Dataset: {split.upper()} -> Accuracy: {round(accs, 3)}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="LPIPS Training Arguments")

    parser.add_argument("--path_to_root", 
                        help="Path to BAPPS Dataset Root",
                        required=True, 
                        type=str)
    
    parser.add_argument("--work_dir", 
                        help="Path to where you want to save checkpoints",
                        required=True, 
                        type=str)
    
    parser.add_argument("--checkpoint_name", 
                        help="Name for the final checkpoint",
                        default="lpips_vgg.pt",
                        required=False, 
                        type=str)
    
    parser.add_argument("--batch_size", 
                        help="Batch size to train with (will get multipled by n_gpus)",
                        default=64, 
                        required=False, 
                        type=int)
    
    parser.add_argument("--eval_batch_size", 
                        help="Batch size to train with",
                        default=256, 
                        required=False, 
                        type=int)
    
    parser.add_argument("--img_size", 
                        help="What image size do you want to use?",
                        default=64, 
                        required=False, 
                        type=int)

    parser.add_argument("--num_workers",
                        help="DataLoader workers", 
                        default=8, 
                        required=False,
                        type=int)
    
    parser.add_argument("--num_epochs",
                        help="How many epochs do you want to train for?", 
                        default=10, 
                        required=False,
                        type=int)
    
    parser.add_argument("--decay_epochs",
                        help="How many epochs do you want linearly decay LR?", 
                        default=5, 
                        required=False,
                        type=int)
    
    parser.add_argument("--initial_lr",
                        help="What learning rate do you want to use?", 
                        default=1e-4, 
                        required=False,
                        type=float)
    
    parser.add_argument("--logging_steps", 
                        help="After how many iterations do you want to print logs",
                        default=1000, 
                        required=False, 
                        type=int)
    
    parser.add_argument("--pretrained_backbone", 
                        help="Use a pretrained backbone", 
                        action='store_true')

    parser.add_argument("--train_backbone", 
                        help="Allow training of the backbone", 
                        action='store_true')

    parser.add_argument("--use_dropout", 
                        help="Enable dropout layers", 
                        action='store_true')

    parser.add_argument("--img_range", 
                        help="Image range options: 'minus_one_to_one' or 'zero_to_one' (default: 'minus_one_to_one')", 
                        default="minus_one_to_one", 
                        required=False,
                        type=str)

    parser.add_argument("--middle_channels", 
                        help="Number of middle channels in the model (default: 32)", 
                        default=32, 
                        required=False,
                        type=int)
    
    parser.add_argument("--evaluation_only", 
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        type=bool)
    
    parser.add_argument("--eval_lpips_pkg",
                        action=argparse.BooleanOptionalAction, 
                        default=False, 
                        type=bool)
    
    parser.add_argument("--mixed_precision",
                        action=argparse.BooleanOptionalAction, 
                        default=False, 
                        type=bool)
    
    args = parser.parse_args()

    ### Define Accelerator ###
    accelerator = Accelerator()

    if not args.evaluation_only:
        trainer(args)

    ### Evaluate on one GPU only ###
    if accelerator.is_main_process:
        eval(args)

    accelerator.wait_for_everyone()
    accelerator.end_training()