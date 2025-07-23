import yaml
import argparse
import torch 
from tqdm import tqdm
from torch.utils.data import DataLoader 
from safetensors.torch import load_file

from modules import LDMConfig, VAE
from dataset import get_dataset
import numpy as np


def save_latents(args):
    print("Computing Scaling Constant for:", args.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Load VAE Config ###
    with open("configs/ldm_f8d4.yaml", "r") as f:
        vae_config = yaml.safe_load(f)
        config = LDMConfig(**vae_config["vae"])

    ### Load Model and weights ###
    model = VAE(config)
    state_dict = load_file(args.path_to_pretrained_weights)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)

    ### Load Dataset ###
    dataset, _ = get_dataset(dataset=args.dataset, path_to_data=args.path_to_dataset, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    if args.num_batches is None:
        samples = len(dataset)
    else:
        samples = args.batch_size * args.num_batches
    print(f"Using {samples} Samples to Compute Statistics")


    step_counter = 0
    all_latents = []
    pbar = tqdm(range(args.num_batches if args.num_batches is not None else len(loader)))
    for images in loader:
        with torch.no_grad():
            latents = model.encode(images["images"].to(device))["posterior"]  # (batch, 4, 32, 32)

        all_latents.append(latents.cpu())
        step_counter += 1
        pbar.update(1)

        if args.num_batches is not None and step_counter >= args.num_batches:
            break

    # Concatenate and save
    all_latents = torch.cat(all_latents, dim=0)  # (total_samples, 4, 32, 32)
    np.save("latents.npy", all_latents.numpy())



def get_scaling_bound(latent_path=None, pct=95.0):
    # Load latents
    all_latents = np.load(latent_path)  # shape: (N, C, H, W)
    print(f"Loaded latents with shape {all_latents.shape}")

    # Flatten all values
    flat_latents = all_latents.flatten()

    # Compute global statistics
    mean_value = np.mean(flat_latents)
    upper_bound = np.percentile(flat_latents, pct)
    lower_bound = np.percentile(flat_latents, 100-pct)
    bound = max(abs(upper_bound), abs(lower_bound))

    print(f"Mean: {mean_value:.6f}")
    print(f"{pct}th Percentile: {upper_bound:.6f}")
    print(f"{100-pct}th Percentile: {lower_bound:.6f}")
    print(f"scaling facor is {1 / bound}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Compute Scaling Factor for Latent Space")
    # parser.add_argument("--path_to_pretrained_weights", help="pretrained VAE Model", required=True)
    # parser.add_argument("--batch_size", help="batch size for inference?", type=int, default=128)
    # parser.add_argument("--num_batches", default=None, type=int, help="None uses the entire dataset")
    # parser.add_argument("--num_workers", help="for dataloader", type=int, default=8)
    # parser.add_argument("--dataset", help="name of dataset",
    #                     choices=("conceptual_captions", "imagenet", "coco", "celeba", "celeba256", "birds", "ffhd"),
    #                     required=True, type=str)
    # parser.add_argument("--path_to_dataset", help="Root directory of dataset", required=True, type=str)
    # args = parser.parse_args()
    # save_latents(args)

    get_scaling_bound(latent_path='latents.npy', pct=99.99)



