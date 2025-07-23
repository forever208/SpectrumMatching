import os

import yaml
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import numpy as np

from modules import LDMConfig, VAE
from dataset import get_dataset


def main(args):
    print("extract image latents for:", args.dataset)
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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=8, pin_memory=False, persistent_workers=True)
    samples = len(dataset)
    print(f"found {samples} samples in {args.dataset}")

    idx = 0
    os.makedirs(args.path_to_latents, exist_ok=True)
    for batch in tqdm(loader):
        with torch.no_grad():
            img = batch["images"].to(device)
            moments = model.forward_enc(img)  # mean and logvar, (batch, 8, 32, 32)
            moments = moments.detach().cpu().numpy()

            for moment in moments:
                np.save(f'{args.path_to_latents}/{idx}.npy', moment)
                idx += 1

    print(f'saved {idx} latent files')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract image latents for later LDM training")
    parser.add_argument("--path_to_pretrained_weights", help="pretrained VAE Model", required=True)
    parser.add_argument("--batch_size", help="batch size for inference?", type=int, default=128)
    parser.add_argument("--dataset", help="name of dataset",
                        choices=("conceptual_captions", "imagenet", "coco", "celeba", "celeba256", "birds", "ffhd"),
                        required=True, type=str)
    parser.add_argument("--path_to_dataset", help="Root directory of dataset", required=True, type=str)
    parser.add_argument("--path_to_latents", required=True, type=str)
    args = parser.parse_args()

    main(args)



