import yaml
import argparse
import torch 
from tqdm import tqdm
from torch.utils.data import DataLoader 
from safetensors.torch import load_file

from modules import LDMConfig, VAE
from dataset import get_dataset

def main(args):

    print("Computing Scaling Constant for:", args.dataset)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Load VAE Config ###
    with open("configs/ldm.yaml", "r") as f:
        vae_config = yaml.safe_load(f)
        config = LDMConfig(**vae_config["vae"])

    ### Load Model ###
    model = VAE(config)

    ### Load PreTrained Weights ###
    state_dict = load_file(args.path_to_pretrained_weights)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)

    ### Load Dataset ###
    dataset = get_dataset(dataset=args.dataset, 
                          path_to_data=args.path_to_dataset,
                          return_caption=False,
                          train=False)
    
    if args.num_batches is None:
        samples = len(dataset)
    else:
        samples = args.batch_size * args.num_batches

    print(f"Using {samples} Samples to Compute Statistics")

    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        shuffle=True)    

    pixel_sum = 0
    pixel_sum_squared = 0
    num_pixels = 0
    step_counter = 0

    pbar = tqdm(range(args.num_batches if args.num_batches is not None else len(loader)))

    for images in loader:

        with torch.no_grad():
            latents = model.encode(images["images"].to(device))["posterior"]

        pixel_sum += latents.sum()
        pixel_sum_squared += (latents**2).sum()
        num_pixels += latents.numel()

        step_counter += 1
        pbar.update(1)

        if args.num_batches is not None and step_counter >= args.num_batches:
            break
        
    mean = pixel_sum / num_pixels
    mean_squared = pixel_sum_squared / num_pixels
    variance = mean_squared - mean**2
    std = torch.sqrt(variance).item()

    # N(A,B) / sqrt(B) -> N(A, 1)
    normalization_constant = 1 / std

    print(normalization_constant)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute Scaling Factor for Latent Space")

    parser.add_argument("--path_to_pretrained_weights",
                        help="Path to pretrained VAE Model",
                        required=True)
    
    parser.add_argument("--batch_size", 
                        help="What batch size for inference?",
                        type=int,
                        default=128)
    
    parser.add_argument("--num_batches",
                        default=None,
                        type=int,
                        help="How many batches do you want to use to estimate scaling? None will use the entire dataset")
    
    parser.add_argument("--num_workers",
                        help="How many workers for dataloader?",
                        type=int,
                        default=8)
    
    parser.add_argument("--dataset",
                        help="What dataset do you want to train on?",
                        choices=("conceptual_captions", "imagenet", "coco", "celeba", "celebahq", "birds", "ffhd"),
                        required=True,
                        type=str)

    parser.add_argument("--path_to_dataset",
                        help="Root directory of dataset",
                        required=True,
                        type=str)
    
    args = parser.parse_args()

    main(args)



