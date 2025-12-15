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
from utils_DCT import latent_spectral_reg_dct


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


def spectrum_difference(path_to_pretrained_weights=None, config_file=None, dataset=None,
                        img_sz=None, path_to_dataset=None, bs=1, max_samples=1000, n_bins=16):

    print("evaluating specturm dfference for:", dataset)
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
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False,
                        num_workers=8, pin_memory=True, persistent_workers=True)
    total_in_dataset = len(dataset)
    print(f"found {total_in_dataset} samples in {dataset}")

    target_N = min(max_samples, total_in_dataset)
    sx_chunks = []
    sz_chunks = []
    loss_sum = 0.0
    n_collected = 0

    for batch in tqdm(loader):
        if n_collected >= target_N:
            break

        with torch.no_grad():
            img = batch["images"].to(device)  # (batch, 3, img_h, img_w)
            latent = model.encode(img, scale_factor=1.0)  # mean and logvar, (batch, 8, 32, 32)
            latent = latent["posterior"]  # (batch, C, H, W)

            sx, sz, kl_loss = latent_spectral_reg_dct(
                img, latent,
                blur_ks=7, blur_sigma=1.2, n_bins=n_bins,
                loss_type="kl", center="none", remove_dc=False, return_dist=True
            )

            sx = sx.detach().cpu()  # sx, sz expected (B,n_bins)
            sz = sz.detach().cpu()

            B = sx.shape[0]
            remaining = target_N - n_collected
            take = min(B, remaining)
            sx_chunks.append(sx[:take])
            sz_chunks.append(sz[:take])

            # kl_loss could be scalar (already mean) or (B,) per-sample
            if torch.is_tensor(kl_loss):
                kl_loss_t = kl_loss.detach().cpu()
                if kl_loss_t.ndim == 0:
                    loss_sum += float(kl_loss_t.item()) * take
                else:
                    loss_sum += float(kl_loss_t[:take].sum().item())
            else:
                loss_sum += float(kl_loss) * take  # python float

            n_collected += take

    sx_all = torch.cat(sx_chunks, dim=0).numpy()  # (N,n_bins)
    sz_all = torch.cat(sz_chunks, dim=0).numpy()  # (N,n_bins)
    mean_kl = loss_sum / n_collected

    print(f"Collected N={n_collected} samples")
    print(f"Mean KL loss over N samples: {mean_kl:.6f}")

    # ============================================================
    # Plot 1: mean ± std curves across bins
    # ============================================================
    bins = np.arange(n_bins)
    sx_mean, sx_std = sx_all.mean(axis=0), sx_all.std(axis=0)
    sz_mean, sz_std = sz_all.mean(axis=0), sz_all.std(axis=0)

    plt.figure(figsize=(10, 4))
    plt.plot(bins, sx_mean, marker="o", label="x mean")
    plt.fill_between(bins, sx_mean - sx_std, sx_mean + sx_std, alpha=0.2)
    plt.plot(bins, sz_mean, marker="o", label="z mean")
    plt.fill_between(bins, sz_mean - sz_std, sz_mean + sz_std, alpha=0.2)
    plt.xlabel("Radial frequency bin (low → high)")
    plt.ylabel("Normalized band energy" if np.allclose(sx_all.sum(1), 1, atol=1e-3) else "Band energy")
    plt.title(f"P(x) vs P(z) over {n_collected} samples (mean ± std)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # # ============================================================
    # # Plot 2: per-bin distribution (boxplots)
    # # ============================================================
    # # side-by-side boxplots for each bin: sx then sz
    # data = []
    # positions = []
    # labels = []
    # for k in range(n_bins):
    #     data.append(sx_all[:, k])
    #     positions.append(2 * k + 1)
    #     labels.append(f"{k}\n(sx)")
    #     data.append(sz_all[:, k])
    #     positions.append(2 * k + 2)
    #     labels.append(f"{k}\n(sz)")
    #
    # plt.figure(figsize=(max(12, n_bins * 0.8), 5))
    # plt.boxplot(data, positions=positions, showfliers=False)
    # plt.xticks(positions, labels, rotation=0)
    # plt.xlabel("Bin index (each bin has sx and sz)")
    # plt.ylabel("Band energy value")
    # plt.title(f"Per-bin distribution of sx and sz over {n_collected} samples")
    # plt.grid(True, axis="y", alpha=0.3)
    # plt.tight_layout()
    # plt.show()



if __name__ == "__main__":
    # visualize_latent(
    #     path_to_pretrained_weights='/home/mang/Downloads/celeba256_SDVAE_bf16_b48_f16d16_flip_400k/SDVAE/checkpoint_330000/model.safetensors',
    #     config_file='configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #     path_to_dataset='/home/mang/Downloads/celeba256/celeba256_visual',
    # )

    # visualize_recon(
    #     path_to_pretrained_weights='celeba256_SDVAE_bf16_b48_f8d4_flip/SDVAE/checkpoint_930000/model.safetensors',
    #     config_file='configs/ldm_f8d4.yaml', dataset='celeba256', img_sz=256,
    #     path_to_dataset='/home/mang/Downloads/celeba256/celeba256_visual',
    # )

    spectrum_difference(
        path_to_pretrained_weights='/home/mang/Downloads/celeba256_SDVAE_bf16_b48_f16d16_flip_400k/SDVAE/checkpoint_330000/model.safetensors',
        config_file='configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
        path_to_dataset='/home/mang/Downloads/celeba256/celeba256',
        bs=8, max_samples=1000, n_bins=16,
    )