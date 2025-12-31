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
from utils_DCT import latent_spectral_reg_dct, split_into_blocks_torch, combine_blocks_torch, dct_2d_torch_unified, idct_2d_torch_unified, gaussian_blur, downsample_to, rmsc
import torch.nn.functional as F
from utils import convert_to_PIL_imgs
from eval_utils.fid_score import calculate_fid_given_paths
import shutil


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
    print(f"loading ckpt from {path_to_pretrained_weights}")
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

    # print(f"diff of low: {np.abs((sx_all[:, 0] - sz_all[:, 0])).mean()}")
    # print(f"diff of high: {np.abs((sx_all[:, -1] - sz_all[:, -1])).mean()}")

    print(f'px: {list(np.round(sx_mean, 4))}')
    print(f'px: {list(np.round(sz_mean, 4))}')

    # plt.figure(figsize=(10, 4))
    # plt.plot(bins, sx_mean, marker="o", label="x mean")
    # plt.fill_between(bins, sx_mean - sx_std, sx_mean + sx_std, alpha=0.2)
    # plt.plot(bins, sz_mean, marker="o", label="z mean")
    # plt.fill_between(bins, sz_mean - sz_std, sz_mean + sz_std, alpha=0.2)
    # plt.xlabel("Radial frequency bin (low → high)")
    # plt.ylabel("Normalized band energy" if np.allclose(sx_all.sum(1), 1, atol=1e-3) else "Band energy")
    # plt.title(f"P(x) vs P(z) over {n_collected} samples (mean ± std)")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


def visualize_PSD(px, pz, n_bins=16):
    px = [0.2404, 0.3026, 0.1509, 0.1053, 0.0536, 0.04, 0.0246, 0.018, 0.0137, 0.0117, 0.0096, 0.0077, 0.0068, 0.0056, 0.0051, 0.0045]

    # SDVAE
    pz_200k = [0.1162, 0.1412, 0.0890, 0.0758, 0.0629, 0.0575, 0.0550, 0.0507, 0.0488, 0.0483, 0.0463, 0.0443, 0.0425, 0.0412, 0.0404, 0.0400]
    pz_250k = [0.1166, 0.1410, 0.0892, 0.0758, 0.0628, 0.0571, 0.0548, 0.0504, 0.0485, 0.0481, 0.0461, 0.0442, 0.0427, 0.0416, 0.0408, 0.0402]
    pz_300k = [0.1156, 0.1394, 0.0883, 0.0754, 0.0623, 0.0571, 0.0545, 0.0505, 0.0488, 0.0483, 0.0465, 0.0445, 0.0433, 0.0422, 0.0417, 0.0413]
    pz_330k = [0.1146, 0.1386, 0.0881, 0.0754, 0.0623, 0.0572, 0.0546, 0.0506, 0.0490, 0.0484, 0.0467, 0.0447, 0.0436, 0.0425, 0.0420, 0.0416]
    pz_380k = [0.1115, 0.1336, 0.0861, 0.0744, 0.0623, 0.0575, 0.0554, 0.0516, 0.0500, 0.0495, 0.0478, 0.0458, 0.0448, 0.0438, 0.0432, 0.0427]

    # dowensam
    pz_250k = [0.1411, 0.1471, 0.1202, 0.1006, 0.0826, 0.0715, 0.0644, 0.0506, 0.0442, 0.0392, 0.0278, 0.0474, 0.0167, 0.0155, 0.0150, 0.0162]
    pz_300k = [0.1390, 0.1448, 0.1184, 0.0994, 0.0820, 0.0711, 0.0644, 0.0510, 0.0451, 0.0402, 0.0288, 0.0489, 0.0176, 0.0163, 0.0159, 0.0170]
    pz_350k = [0.1375, 0.1430, 0.1169, 0.0986, 0.0818, 0.0713, 0.0650, 0.0518, 0.0458, 0.0409, 0.0294, 0.0495, 0.0180, 0.0168, 0.0163, 0.0174]
    pz_380k = [0.1378, 0.1433, 0.1173, 0.0988, 0.0816, 0.0708, 0.0642, 0.0510, 0.0453, 0.0407, 0.0293, 0.0511, 0.0181, 0.0168, 0.0164, 0.0176]
    pz_450k = [0.1349, 0.1404, 0.1153, 0.0977, 0.0814, 0.0711, 0.0648, 0.0520, 0.0463, 0.0418, 0.0304, 0.0521, 0.0189, 0.0176, 0.0171, 0.0184]

    bins = np.arange(n_bins)

    plt.figure(figsize=(10, 4))
    plt.plot(bins, px, marker="o", label="x mean")
    plt.plot(bins, pz, marker="o", label="z mean")
    plt.xlabel("Radial frequency bin (low → high)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def spectrum_loss(path_to_pretrained_weights=None, config_file=None, dataset=None,
                        img_sz=None, path_to_dataset=None, bs=1, max_samples=5000):

    print("evaluating specturm loss for ckpt:", path_to_pretrained_weights)
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
    num_iterations = target_N // bs
    SpecLoss = []

    eval_iter = iter(loader)
    for i in tqdm(range(num_iterations), desc='spectrum loss'):
        batch = next(eval_iter)

        with torch.no_grad():
            img = batch["images"].to(device)  # (batch, 3, img_h, img_w)
            latent = model.encode(img, scale_factor=1.0)  # mean and logvar, (batch, 8, 32, 32)
            latent = latent["posterior"]  # (batch, C, H, W)

            kl_loss = latent_spectral_reg_dct(
            img, latent,
            blur_ks=7, blur_sigma=1.2, n_bins=16,
            loss_type="kl", log_power=True, center="mean", remove_dc=True,
            )

            SpecLoss.append(kl_loss)

    SpecLoss = torch.tensor(SpecLoss)
    SpecLoss = SpecLoss.mean().item()

    print(f"Mean Spec loss over {num_iterations * bs} samples: {SpecLoss:.6f}")


def downsample_recon_L1_loss(path_to_pretrained_weights=None, config_file=None,
                             dataset=None, img_sz=None, path_to_dataset=None,
                             down_factor: int = 2,  # 2 or 4
                             batch_size: int = 8,
                             max_vis: int = 10,  # only used if batch_size==1
                             num_workers: int = 8,
                             max_samples = 5000):
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
    print(f"loading ckpt from {path_to_pretrained_weights}")

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
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    total_samples = len(dataset_obj)
    print(f"found {total_samples} samples in dataset")
    target_N = min(max_samples, total_samples)
    num_iterations = target_N // batch_size

    # We'll accumulate sums weighted by batch size to get true dataset mean.
    l1_sum = 0.0
    mse_sum = 0.0
    n_seen = 0

    shown = 0
    eval_iter = iter(loader)
    for i in tqdm(range(num_iterations)):
        batch = next(eval_iter)

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


def downsample_rFID(path_to_pretrained_weights=None, config_file=None,
                    dataset=None, img_sz=None, path_to_dataset=None, path_to_save_imgs=None,
                    down_factor: int = 2,  # 2 or 4
                    batch_size: int = 8,
                    num_workers: int = 8,
                    max_samples = 5000):

    assert down_factor in (2, 4), "down_factor must be 2 or 4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"downsample_recon | dataset={dataset} | down_factor={down_factor} | batch_size={batch_size}")
    print(f"loading ckpt from {path_to_pretrained_weights}")

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
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    total_samples = len(dataset_obj)
    print(f"found {total_samples} samples in dataset")
    target_N = min(max_samples, total_samples)
    num_iterations = target_N // batch_size

    # We'll accumulate sums weighted by batch size to get true dataset mean.
    eval_org_imgs_path = os.path.join(path_to_save_imgs, "eval_down_org_imgs")
    eval_recon_imgs_path = os.path.join(path_to_save_imgs, "eval_down_recon_imgs")
    os.makedirs(eval_org_imgs_path, exist_ok=True)
    os.makedirs(eval_recon_imgs_path, exist_ok=True)

    eval_iter = iter(loader)
    for n in tqdm(range(num_iterations)):
        batch = next(eval_iter)

        with torch.no_grad():
            img = batch["images"].to(device, non_blocking=True)  # (B,3,H,W)
            _, _, H, W = img.shape

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

            down_img = convert_to_PIL_imgs(down_img)  # a list PIL images
            recon_down_img = convert_to_PIL_imgs(recon_down_img)  # a list PIL images

            for b_id in range(batch_size):  # distributed image save
                img_id = batch_size * n + b_id

                if img_id >= target_N:
                    break

                down_img[b_id].save(os.path.join(eval_org_imgs_path, f"{img_id}.jpg"))
                recon_down_img[b_id].save(os.path.join(eval_recon_imgs_path, f"{img_id}.jpg"))

    print(f'{len(os.listdir(eval_org_imgs_path))} images in {eval_org_imgs_path}')
    print(f'{len(os.listdir(eval_recon_imgs_path))} images in {eval_recon_imgs_path}')

    fid = calculate_fid_given_paths([eval_org_imgs_path, eval_recon_imgs_path], device=device)
    print(f"downsample rFID is {fid}")
    shutil.rmtree(eval_org_imgs_path)  # remove the image folder
    shutil.rmtree(eval_recon_imgs_path)  # remove the image folder


def lowpass_rFID(path_to_pretrained_weights=None, config_file=None,
                 dataset=None, img_sz=None, path_to_dataset=None, path_to_save_imgs=None,
                 batch_size: int = 8,
                 num_workers: int = 8,
                 max_samples = 5000,
                 k = 4,
                 blk_sz = 8):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"lowpass_recon | dataset={dataset} | remove_high_freq_corner={k} | batch_size={batch_size}")
    print(f"loading ckpt from {path_to_pretrained_weights}")

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
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    total_samples = len(dataset_obj)
    print(f"found {total_samples} samples in dataset")
    target_N = min(max_samples, total_samples)
    num_iterations = target_N // batch_size

    # We'll accumulate sums weighted by batch size to get true dataset mean.
    eval_org_imgs_path = os.path.join(path_to_save_imgs, "eval_lowpass_org_imgs")
    eval_recon_imgs_path = os.path.join(path_to_save_imgs, "eval_lowpass_recon_imgs")
    os.makedirs(eval_org_imgs_path, exist_ok=True)
    os.makedirs(eval_recon_imgs_path, exist_ok=True)

    eval_iter = iter(loader)
    for n in tqdm(range(num_iterations)):
        batch = next(eval_iter)

        with torch.no_grad():
            img = batch["images"].to(device, non_blocking=True)  # (B,3,H,W)
            _, _, H, W = img.shape

            # ---- Encode ----
            outputs = model.encode(img)
            z = outputs["posterior"]  # (B,C,h,w) tensor expected
            _, _, h, w = z.shape

            # ---- low_pass z and img ----
            z = split_into_blocks_torch(z, blk_sz)  # (B, C, num_blocks, b, b)
            img = split_into_blocks_torch(img, blk_sz)  # (B, C, NUM_blocks, b, b)

            z = dct_2d_torch_unified(z, center="none")  # (B, C, num_blocks, b, b)
            img = dct_2d_torch_unified(img, center="none")  # (B, C, NUM_blocks, b, b)

            max_sum = 2 * (blk_sz - 1)  # 14 for 8x8
            thresh = max_sum - (k - 1)  # 15 - k for 8x8

            u = torch.arange(blk_sz, device=z.device).view(blk_sz, 1)
            v = torch.arange(blk_sz, device=z.device).view(1, blk_sz)
            hf_mask = (u + v) >= thresh  # (8,8) True => to be zeroed

            z[..., hf_mask] = 0  # low-pass filter
            img[..., hf_mask] = 0

            z = idct_2d_torch_unified(z, center="none")  # (B, C, num_blocks, b, b)
            img = idct_2d_torch_unified(img, center="none") # (B, C, NUM_blocks, b, b)

            z = combine_blocks_torch(z, h, w, blk_sz)  # (B, C, h, w)
            img = combine_blocks_torch(img, H, W, blk_sz)  # (B, C, H, W)

            # ---- Decode ----
            recon_lowpass_img = model.decode(z)  # (B, C, H, W)

            lowpass_img = convert_to_PIL_imgs(img)  # a list PIL images
            recon_lowpass_img = convert_to_PIL_imgs(recon_lowpass_img)  # a list PIL images

            for b_id in range(batch_size):  # distributed image save
                img_id = batch_size * n + b_id

                if img_id >= target_N:
                    break

                lowpass_img[b_id].save(os.path.join(eval_org_imgs_path, f"{img_id}.jpg"))
                recon_lowpass_img[b_id].save(os.path.join(eval_recon_imgs_path, f"{img_id}.jpg"))

    print(f'{len(os.listdir(eval_org_imgs_path))} images in {eval_org_imgs_path}')
    print(f'{len(os.listdir(eval_recon_imgs_path))} images in {eval_recon_imgs_path}')

    fid = calculate_fid_given_paths([eval_org_imgs_path, eval_recon_imgs_path], device=device)
    print(f"downsample rFID is {fid}")
    shutil.rmtree(eval_org_imgs_path)  # remove the image folder
    shutil.rmtree(eval_recon_imgs_path)  # remove the image folder


def rmsc_loss(path_to_pretrained_weights=None, config_file=None, dataset=None,
                        img_sz=None, path_to_dataset=None, bs=1, max_samples=5000):

    print("evaluating RMSC loss for ckpt:", path_to_pretrained_weights)
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
    num_iterations = target_N // bs
    x_RMSC = []
    z_RMSC = []
    eval_iter = iter(loader)
    for i in tqdm(range(num_iterations), desc='RMSC loss'):
        batch = next(eval_iter)

        with torch.no_grad():
            img = batch["images"].to(device)  # (batch, 3, img_h, img_w)
            latent = model.encode(img, scale_factor=1.0)  # mean and logvar, (batch, 8, 32, 32)
            latent = latent["posterior"]  # (batch, C, H, W)
            _, _, hz, wz = latent.shape


            img = gaussian_blur(img, kernel_size=7, sigma=1.2)
            img = downsample_to(img, (hz, wz))
            img_rmsc = rmsc(img, patch_sz=1)  # (batch, )
            latent_rmsc = rmsc(latent, patch_sz=1)  # (batch, )
            x_RMSC.append(img_rmsc)
            z_RMSC.append(latent_rmsc)

    x_RMSC = torch.cat(x_RMSC).mean().item()
    z_RMSC = torch.cat(z_RMSC).mean().item()

    print(f"Mean RMSC of x over {num_iterations * bs} samples: {x_RMSC:.6f}")
    print(f"Mean RMSC of z over {num_iterations * bs} samples: {z_RMSC:.6f}")
    print(f"RMSC loss is {z_RMSC - x_RMSC}")


if __name__ == "__main__":
    # visualize_latent(
    #     path_to_pretrained_weights='/home/mang/Downloads/celeba256_SDVAE_bf16_b48_f16d16_flip_400k/SDVAE/checkpoint_330000/model.safetensors',
    #     config_file='configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #     path_to_dataset='/home/mang/Downloads/celeba256/celeba256_visual',
    # )


    # spectrum_difference(
    #     path_to_pretrained_weights='/leonardo_work/EUHPC_B29_014/LDM_exps/celeba256_SDVAE_bf16_b48_f16_flip_400k/SDVAE/checkpoint_300000/model.safetensors',
    #     config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #     path_to_dataset='/leonardo_work/EUHPC_B29_014/datasets/celeba256/celeba256',
    #     bs=8, max_samples=30000, n_bins=16,
    # )

    # spectrum_loss(
    #     path_to_pretrained_weights='/leonardo_work/EUHPC_B29_014/LDM_exps/celeba256_SDVAE_bf16_b48_f16_flip_400k/SDVAE/checkpoint_250000/model.safetensors',
    #     config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #     path_to_dataset='/leonardo_work/EUHPC_B29_014/datasets/celeba256/celeba256',
    #     bs=100, max_samples=5000
    # )

    # downsample_recon_L1_loss(
    #     path_to_pretrained_weights='/leonardo_work/EUHPC_B29_014/LDM_exps/celeba256_SDVAE_b48_f16d16_downsam/SDVAE/checkpoint_50000/model.safetensors',
    #     config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #     path_to_dataset='/leonardo_work/EUHPC_B29_014/datasets/celeba256/celeba256',
    #     down_factor=4, batch_size=100, max_samples=5000,
    # )

    # downsample_rFID(
    #     path_to_pretrained_weights='/leonardo_work/EUHPC_B29_014/LDM_exps/celeba256_SDVAE_b48_f16d16_downsam/SDVAE/checkpoint_440000/model.safetensors',
    #     config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #     path_to_dataset='/leonardo_work/EUHPC_B29_014/datasets/celeba256/celeba256',
    #     path_to_save_imgs='/leonardo_work/EUHPC_B29_014',
    #     down_factor=4, batch_size=100, max_samples=5000,
    # )

    # lowpass_rFID(
    #     path_to_pretrained_weights='/leonardo_work/EUHPC_B29_014/LDM_exps/celeba256_SM_b48_f16_16bins_log000_decodeSM_d16/SDVAE/checkpoint_200000/model.safetensors',
    #     config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #     path_to_dataset='/leonardo_work/EUHPC_B29_014/datasets/celeba256/celeba256',
    #     path_to_save_imgs='/leonardo_work/EUHPC_B29_014',
    #     batch_size=100, max_samples=10000, blk_sz=4, k=4
    # )

    rmsc_loss(
        path_to_pretrained_weights='/leonardo_work/EUHPC_B29_014/LDM_exps/celeba256_SDVAE_b48_f16_downsam/SDVAE/checkpoint_380000/model.safetensors',
        config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
        path_to_dataset='/leonardo_work/EUHPC_B29_014/datasets/celeba256/celeba256',
        bs=100, max_samples=30000
    )