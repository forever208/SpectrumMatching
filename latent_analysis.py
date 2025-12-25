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
import torch.nn.functional as F


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



def _to_img01(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,3,H,W) in [-1,1] or [0,1]
    returns: (H,W,3) in [0,1] CPU for matplotlib (uses first sample)
    """
    x = x.detach().float().cpu()
    if x.min() < -0.1:
        x = (x + 1.0) * 0.5
    x = x.clamp(0, 1)
    return x[0].permute(1, 2, 0)


def downsample_recon(path_to_pretrained_weights=None, config_file=None,
                     dataset=None, img_sz=None, path_to_dataset=None,
                     down_factor: int = 2,              # 2 or 4
                     batch_size: int = 8,
                     max_vis: int = 10,                 # only used if batch_size==1
                     num_workers: int = 8):
    """
    Batch-capable version:
      - encode img -> z
      - downsample img and z by down_factor
      - decode down_z -> recon_down_img
      - compute per-batch pixel-wise L1/MSE (and dataset means)
      - visualize only when batch_size == 1 (first max_vis samples)

    Returns mean L1/MSE over all samples.
    """
    assert down_factor in (2, 4), "down_factor must be 2 or 4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"downsample_recon | dataset={dataset} | down_factor={down_factor} | batch_size={batch_size}")

    # ---------- Load VAE Config ----------
    with open(config_file, "r") as f:
        vae_config = yaml.safe_load(f)
        config = LDMConfig(**vae_config["vae"])

    # ---------- Load Model ----------
    model = VAE(config)
    state_dict = load_file(path_to_pretrained_weights)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    # ---------- Load Dataset ----------
    dataset_obj, _ = get_dataset(
        dataset=dataset,
        path_to_data=path_to_dataset,
        num_channels=3,
        img_size=img_sz,
        random_resize=False,
        random_flip_p=0.0,
        train=False
    )
    loader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    total_samples = len(dataset_obj)
    print(f"found {total_samples} samples in dataset")

    # We'll accumulate sums weighted by batch size to get true dataset mean.
    l1_sum = 0.0
    mse_sum = 0.0
    n_seen = 0

    shown = 0
    for batch in tqdm(loader):
        with torch.no_grad():
            img = batch["images"].to(device, non_blocking=True)  # (B,3,H,W)
            B, _, H, W = img.shape

            # ---- Encode ----
            outputs = model.encode(img)
            z = outputs["posterior"]  # (B,C,zh,zw) tensor expected

            # ---- Downsample img ----
            down_H, down_W = H // down_factor, W // down_factor
            down_img = F.interpolate(img, size=(down_H, down_W), mode="bicubic", align_corners=False)  # (B, C, down_H, down_W)

            # ---- Downsample z ----
            zh, zw = z.shape[-2:]
            down_zh, down_zw = max(1, zh // down_factor), max(1, zw // down_factor)
            down_z = F.interpolate(z, size=(down_zh, down_zw), mode="bicubic", align_corners=False)

            # ---- Decode ----
            recon_down_img = model.decode(down_z)  # (B, C, down_H, down_W)

            # enforce size match
            assert recon_down_img.shape == down_img.shape

            # ---- Pixel-wise errors (mean over batch+pixels+channels) ----
            l1_batch = (recon_down_img - down_img).abs().mean()          # scalar tensor
            mse_batch = ((recon_down_img - down_img) ** 2).mean()        # scalar tensor

            # accumulate weighted by batch size
            l1_sum += float(l1_batch.item()) * B
            mse_sum += float(mse_batch.item()) * B
            n_seen += B

            # ---- Visualization only if batch_size == 1 ----
            if batch_size == 1 and shown < max_vis:
                fig = plt.figure(figsize=(10, 4))
                ax1 = fig.add_subplot(1, 2, 1)
                ax1.imshow(_to_img01(down_img))
                ax1.set_title(f"down_img ({down_H}x{down_W})")
                ax1.axis("off")

                ax2 = fig.add_subplot(1, 2, 2)
                ax2.imshow(_to_img01(recon_down_img))
                ax2.set_title(f"recon_down_img | L1={l1_batch.item():.4f} MSE={mse_batch.item():.4f}")
                ax2.axis("off")

                plt.tight_layout()
                plt.show()
                shown += 1

    mean_l1 = l1_sum / max(n_seen, 1)
    mean_mse = mse_sum / max(n_seen, 1)

    print(f"Mean L1 over {n_seen} samples:  {mean_l1:.6f}")
    # print(f"Mean MSE over {n_seen} samples: {mean_mse:.6f}")

    return {
        "mean_l1": mean_l1,
        "mean_mse": mean_mse,
        "n_seen": n_seen
    }


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
                loss_type="kl", center="mean", remove_dc=True, return_dist=True
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
    #     path_to_pretrained_weights='/home/mang/Downloads/celeba256_SDVAE_bf16_b48_f16_flip_400k/SDVAE/checkpoint_300000/model.safetensors',
    #     config_file='configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #     path_to_dataset='/home/mang/Downloads/celeba256/celeba256_visual',
    # )

    spectrum_difference(
        path_to_pretrained_weights='/home/mang/Downloads/celeba256_SDVAE_bf16_b48_f16_flip_400k/SDVAE/checkpoint_300000/model.safetensors',
        config_file='configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
        path_to_dataset='/home/mang/Downloads/celeba256/celeba256',
        bs=8, max_samples=1000, n_bins=16,
    )

    # downsample_recon(
    #     path_to_pretrained_weights='/home/mang/Downloads/celeba256_SM_b48_f16d16_64bins_nolog_KLx1/SDVAE/checkpoint_130000/model.safetensors',
    #     config_file='configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #     path_to_dataset='/home/mang/Downloads/celeba256/celeba256_visual',
    #     down_factor=2, batch_size=1,
    #
    # )