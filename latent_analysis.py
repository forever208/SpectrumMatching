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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from DCT_utils import Batch_DCT_to_RGB, split_into_blocks, idct_transform, combine_blocks


def visualize_latent(path_to_pretrained_weights=None, config_file=None,
                     dataset=None, img_sz=None, path_to_dataset=None,):

    print("visualize image latents for:", dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Load VAE Config ###
    with open(config_file, "r") as f:
        vae_config = yaml.safe_load(f)
        config = LDMConfig(**vae_config["vae"])

    ### Load Model and weights ###
    model = VAE(config)
    state_dict = load_file(path_to_pretrained_weights)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)

    ### Load Dataset ###
    dataset, _ = get_dataset(dataset=dataset, path_to_data=path_to_dataset, num_channels=3, img_size=img_sz,
                             random_resize=False, random_flip_p=0.0, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                        num_workers=8, pin_memory=False, persistent_workers=True)
    samples = len(dataset)
    print(f"found {samples} samples in {dataset}")

    for batch in tqdm(loader):
        with torch.no_grad():
            img = batch["images"].to(device)  # (1, 3, img_h, img_w)
            latent = model.encode(img, scale_factor=1.0)  # mean and logvar, (batch, 8, 32, 32)

            latent = latent["posterior"].squeeze(0)  # (C, H, W)
            C, H, W = latent.shape
            latent = latent.cpu().numpy()  # (C, H, W)

            # Apply PCA to reduce C → 3 for RGB visualization
            latent_flat = np.transpose(latent, (1, 2, 0)).reshape(-1, C)  # (C, H, W) --> (H*W, C)
            pca = PCA(n_components=3, random_state=42)
            latent_pca = pca.fit_transform(latent_flat)  # (H*W, C) --> (H*W, 3)

            # Reshape back to image grid
            latent_img = latent_pca.reshape(H, W, 3)

            # Normalize to [0,1] for visualization
            latent_img = (latent_img - latent_img.min()) / (latent_img.max() - latent_img.min() + 1e-8)

            # Prepare the original image for display (1, 3, h, w) -> (h, w, 3)
            img_disp = img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (h, w, 3)
            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)

            # Side-by-side plot
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img_disp)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(latent_img)
            plt.title("Latent PCA")
            plt.axis("off")

            plt.tight_layout()
            plt.show()
            plt.close()



def visualize_recon(path_to_pretrained_weights=None, config_file=None,
                    dataset=None, img_sz=None, path_to_dataset=None,):

    print("visualize image latents for:", dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Load VAE Config ###
    with open(config_file, "r") as f:
        vae_config = yaml.safe_load(f)
        config = LDMConfig(**vae_config["vae"])

    ### Load Model and weights ###
    model = VAE(config)
    state_dict = load_file(path_to_pretrained_weights)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)

    ### Load Dataset ###
    dataset, _ = get_dataset(dataset=dataset, path_to_data=path_to_dataset, num_channels=3, img_size=img_sz,
                             random_resize=False, random_flip_p=0.0, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                        num_workers=8, pin_memory=False, persistent_workers=True)
    samples = len(dataset)
    print(f"found {samples} samples in {dataset}")

    for batch in tqdm(loader):
        with torch.no_grad():
            img = batch["images"].to(device)  # (1, 3, img_h, img_w)

            model_outputs = model(img)
            reconstructions = model_outputs["reconstruction"]  # (batch, 3, H, W)

            # Normalize to [0,1] for visualization
            recon_disp = reconstructions.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (h, w, 3)
            recon_disp = (recon_disp - recon_disp.min()) / (recon_disp.max() - recon_disp.min() + 1e-8)

            # Prepare the original image for display (1, 3, h, w) -> (h, w, 3)
            img_disp = img.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (h, w, 3)
            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)

            # Side-by-side plot
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img_disp)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(recon_disp)
            plt.title("Recon Image")
            plt.axis("off")

            plt.tight_layout()
            plt.show()
            plt.close()


if __name__ == "__main__":
    # visualize_latent(
    #     path_to_pretrained_weights='celeba256_SDVAE_bf16_b48_f8d4_flip/SDVAE/checkpoint_930000/model.safetensors',
    #     config_file='configs/ldm_f8d4.yaml', dataset='celeba256', img_sz=256,
    #     path_to_dataset='/home/mang/Downloads/celeba256/celeba256_visual',
    # )

    visualize_recon(
        path_to_pretrained_weights='celeba256_SDVAE_bf16_b48_f8d4_flip/SDVAE/checkpoint_930000/model.safetensors',
        config_file='configs/ldm_f8d4.yaml', dataset='celeba256', img_sz=256,
        path_to_dataset='/home/mang/Downloads/celeba256/celeba256_visual',
    )