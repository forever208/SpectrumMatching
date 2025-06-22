import os
os.environ["TORCH_DISTRIBUTED_DEBUG"]="INFO"
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import matplotlib.pyplot as plt
from diffusers.optimization import get_scheduler
from safetensors.torch import load_file

from modules import LDM, LDMConfig
from dataset import get_dataset
from utils import save_generated_images, load_testing_text_encodings, \
    load_testing_imagenet_encodings

### Load Arguments ###
def experiment_config_parser():

    parser = argparse.ArgumentParser(description="Experiment Configuration")

    parser.add_argument("--experiment_name", 
                        help="Name of Experiment being Launched", 
                        required=True, 
                        type=str,
                        metavar="experiment_name")
    
    parser.add_argument("--wandb_run_name",
                        required=True, 
                        type=str,
                        metavar="wandb_run_name")
    
    parser.add_argument("--working_directory", 
                        help="Working Directory where checkpoints and logs are stored, inside a \
                        folder labeled by the experiment name", 
                        required=True, 
                        type=str,
                        metavar="working_directory")

    parser.add_argument("--log_wandb",
                        action=argparse.BooleanOptionalAction, 
                        help="Do you want to log to WandB?")
    
    parser.add_argument("--resume_from_checkpoint", 
                        help="Pass name of checkpoint folder to resume training from",
                        default=None,
                        type=str, 
                        metavar="resume_from_checkpoint")
    
    parser.add_argument("--training_config",
                        help="Path to config file for all training information",
                        required=True,
                        type=str, 
                        metavar="training_config")
    
    parser.add_argument("--model_config",
                        help="Path to config file for all model information",
                        required=True, 
                        type=str, 
                        metavar="model_config")
    
    parser.add_argument("--path_to_vae_backbone",
                        help="To train this model, you need a pretrained VAE first!",
                        required=True, 
                        type=str, 
                        metavar="path_to_vae_backbone")
    
    parser.add_argument("--dataset",
                        help="What dataset do you want to train on?",
                        choices=("conceptual_captions", "imagenet", "coco", "celeba", "celebahq", "birds", "ffhd"),
                        required=True,
                        type=str)

    parser.add_argument("--path_to_dataset",
                        help="Root directory of dataset",
                        required=True,
                        type=str)
    
    parser.add_argument("--path_to_save_gens",
                        help="Folder you want to store the testing generations througout training",
                        required=True, 
                        type=str)
    

    args = parser.parse_args()

    return args

args = experiment_config_parser()

### Load Training Config ###
with open(args.training_config, "r") as f:
    training_config = yaml.safe_load(f)["training_args"]

with open(args.model_config, "r") as f:
    ldm_config = yaml.safe_load(f)

### Load Accelerator ###
path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment, 
                          gradient_accumulation_steps=training_config["gradient_accumulations_steps"], 
                          log_with="wandb")

### Make sure Directory to Save Generations Exists ###
if not os.path.isdir(args.path_to_save_gens):
    accelerator.print(f"Creating Directory {args.path_to_save_gens}")
    os.mkdir(args.path_to_save_gens)

### Load Config ###
config = LDMConfig(**ldm_config["vae"], **ldm_config["unet"])
scaling_constants = ldm_config["scaling_constants"]

### Set the VAE Unit Variance Scaling ###
if args.dataset in scaling_constants.keys():
    vae_scale_factor = scaling_constants[args.dataset]
    accelerator.print(f"Using Scaling Constant of {vae_scale_factor}")
else:
    accelerator.print("Using Scaling Constant of 1. Compute with compute_vae_scaling.py and set in ldm.config")
    vae_scale_factor = 1
config.vae_scale_factor = vae_scale_factor

### Check Conditioning Based On Dataset ###
config.class_conditioning = False
config.text_conditioning = False
sample_text_embeddings = None
sample_class_labels = None

if args.dataset == "imagenet":
    config.class_conditioning = True

    ### Load Which Labels To Test Generation ###
    sample_class_labels = load_testing_imagenet_encodings(
        path_to_imagenet_labels="inputs/imagenet_class_prompt.txt"
    )

elif args.dataset == "conceptual_captions":
    config.text_conditioning = True
    config.pre_encoded_text = training_config["pre_encoded_text"]

    ### Load Sample Text Prompt and their Embeddings ###
    sample_text_embeddings = load_testing_text_encodings(
        path_to_text="inputs/sample_text_cond_prompts.txt",
        model=config.text_conditioning_hf_model
    )

### Set Loss Function in Config ###
config.diffusion_loss_fn = training_config["loss_fn"]

### Load Model ###
model = LDM(config)

### Load VAE Backbone ###
state_dict = load_file(args.path_to_vae_backbone)
model._load_vae_state_dict(state_dict)
model = model.to(accelerator.device)

### Check Model Parameters ###
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print("Number of Parameters:", params)

### Prep Dataset ###
mini_batchsize = training_config["per_gpu_batch_size"] // training_config["gradient_accumulations_steps"]
dataset, collate_fn = get_dataset(dataset=args.dataset,
                      path_to_data=args.path_to_dataset,
                      num_channels=config.in_channels,
                      img_size=config.img_size,
                      random_resize=training_config["random_resize"],
                      interpolation=training_config["interpolation"],
                      return_caption=config.text_conditioning,
                      return_classes=config.class_conditioning)    

accelerator.print("Number of Training Samples:", len(dataset))

dataloader = DataLoader(dataset, 
                        batch_size=mini_batchsize,
                        pin_memory=training_config["pin_memory"],
                        num_workers=training_config["num_workers"],
                        shuffle=True,
                        collate_fn=collate_fn)

effective_epochs = (training_config["per_gpu_batch_size"] * \
                        accelerator.num_processes * \
                            training_config["total_training_iterations"]) / len(dataset)

accelerator.print("Effective Epochs:", round(effective_epochs, 2))

### Load Optimizer ###
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=training_config["learning_rate"],
                              betas=(training_config["optimizer_beta1"], training_config["optimizer_beta2"]),
                              weight_decay=training_config["optimizer_weight_decay"])

### Get Learning Rate Scheduler ###
lr_scheduler = get_scheduler(
        training_config["lr_scheduler"],
        optimizer=optimizer,
        num_training_steps=training_config["total_training_iterations"],
        num_warmup_steps=training_config["lr_warmup_steps"]
    )

model, optimizer, lr_scheduler, dataloader = accelerator.prepare(
    model, optimizer, lr_scheduler, dataloader
)

### Load From Checkpoint ###
if args.resume_from_checkpoint is not None:
    path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
    accelerator.load_state(path_to_checkpoint)
    completed_steps = int(args.resume_from_checkpoint.split("_")[-1])
    accelerator.print(f"Resuming from Iteration: {completed_steps}")
else:
    completed_steps = 0

### Latent Space Dimensions ###
compressed_dim = config.img_size//(2**(len(config.vae_channels_per_block) - 1))
latent_space_dim = (config.latent_channels, compressed_dim, compressed_dim)

### Start Training ###
progress_bar = tqdm(range(completed_steps, training_config["total_training_iterations"]), disable=not accelerator.is_main_process)
accumulated_loss = 0
train = True

while train:

    for batch in dataloader:

        prepped_batch = {}
        prepped_batch["images"] = batch["images"].to(accelerator.device)
        prepped_batch["text_conditioning"] = batch["text_conditioning"] \
            if "text_conditioning" in batch.keys() else None
        prepped_batch["text_attention_mask"] = batch["text_attention_mask"] \
            if "text_attention_mask" in batch.keys() else None        
        prepped_batch["class_conditioning"] = batch["class_conditioning"] \
            if "class_conditioning" in batch.keys() else None 
        prepped_batch["cfg_dropout_prob"] = config.cfg_dropout_prob
        
        with accelerator.accumulate():

            ### Compute Loss ###
            loss = model(**prepped_batch)

            ### Train Model ###
            accumulated_loss += loss / training_config["gradient_accumulations_steps"]

            ## Compute Gradients ###
            accelerator.backward(loss)

            ### Clip Gradients ###
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:

            loss_gathered = accelerator.gather_for_metrics(accumulated_loss)
            mean_loss_gathered = torch.mean(loss_gathered).item()

            log = {"loss": mean_loss_gathered,
                   "learning_rate": lr_scheduler.get_last_lr()[0],
                   "iteration": completed_steps}
            
            accelerator.print(log)
            
            accelerator.log(log,
                            step=completed_steps)

            ### Reset and Iterate ###
            accumulated_loss = 0
            completed_steps += 1
            progress_bar.update(1)

        if completed_steps % training_config["val_generation_freq"] == 0:
            
            if accelerator.is_main_process:

                model.eval()

                unwrapped_model = accelerator.unwrap_model(model)

                ### Handle Context ###
                text_conditioning = text_attention_mask = class_conditioning = None

                if sample_text_embeddings is not None:

                    ### Grab the already prepped text conditioning and its attention mask ###
                    text_conditioning = sample_text_embeddings["text_conditioning"].to(accelerator.device)
                    text_attention_mask = sample_text_embeddings["text_attention_mask"].to(accelerator.device)

                    ### Start with Some Noise at Latent Space Dimensions ###
                    latent = torch.randn((len(text_conditioning), *latent_space_dim))
                
                elif sample_class_labels is not None:
                    
                    ### Grab class indexes from GenericImageLoader.classes
                    class_conditioning = torch.tensor([dataset.classes[i] for i in sample_class_labels], device=accelerator.device)

                    ### Get Embeddings from Class Encoder ###
                    class_conditioning = unwrapped_model.class_encoder(batch_size=len(class_conditioning), 
                                                                       class_conditioning=class_conditioning,
                                                                       cfg_dropout_prob=0)            

                    ### Start with Some Noise at Latent Space Dimensions ###
                    latent = torch.randn((len(class_conditioning), *latent_space_dim))

                else:
                    text_conditioning, text_attention_mask = None, None

                    ### Start with Some Noise at Latent Space Dimensions ###
                    latent = torch.randn((training_config["num_val_random_samples"], *latent_space_dim))

                
                with torch.no_grad():

                    ### Iteratively Pass Through UNet and use Sampler to remove noise ###
                    for t in tqdm(np.arange(config.num_diffusion_timesteps)[::-1], disable=not accelerator.is_main_process):

                        ### Generate Timesteps and Get Embeddings ###
                        ts = torch.full((training_config["num_val_random_samples"], ), t)
                        timestep_embeddings = unwrapped_model.sinusoidal_time_embeddings(ts.to(accelerator.device))

                        noise_pred = unwrapped_model.unet(latent.to(accelerator.device), 
                                                          timestep_embeddings,
                                                          text_conditioning=text_conditioning, 
                                                          text_attention_mask=text_attention_mask,
                                                          class_conditioning=class_conditioning)
                        
                        latent = unwrapped_model.ddpm_sampler.remove_noise(latent, ts, noise_pred.detach().cpu())

                    ### Decode Latent Back to Image Space ###
                    images = unwrapped_model._vae_decode_images(latent.to(accelerator.device))

                    save_generated_images(images,
                                          path_to_save_folder=args.path_to_save_gens,
                                          step=completed_steps)
                    
                model.train()

            accelerator.wait_for_everyone()
        
        if (completed_steps % training_config["checkpoint_iterations"] == 0) or (completed_steps == training_config["total_training_iterations"]-1):
            path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{completed_steps}")
            accelerator.save_state(output_dir=path_to_checkpoint)


