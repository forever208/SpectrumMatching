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


def visualize_latent_pca_paperstyle(
    path_to_pretrained_weights_list, config_file,
    dataset, img_sz, path_to_dataset,
    titles=("SD-VAE","+Ours","SDXL-VAE","+Ours"),
    n_fit_imgs=64, pixel_subsample=4096, n_show=3, seed=42,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(config_file, "r") as f:
        config = LDMConfig(**yaml.safe_load(f)["vae"])

    # load dataset (UNCHANGED from your code)
    ds, _ = get_dataset(dataset=dataset, path_to_data=path_to_dataset, num_channels=3, img_size=img_sz,
                        random_resize=False, random_flip_p=0.0, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False,
                        num_workers=8, pin_memory=False, persistent_workers=True)

    # load 4 models (same arch, different weights)
    models = []
    for w in path_to_pretrained_weights_list:
        m = VAE(config); m.load_state_dict(load_file(w), strict=True)
        models.append(m.to(device).eval())

    rng = np.random.default_rng(seed)
    it_fit = iter(loader)

    # fit PCA SEPARATELY for each model on many latents (paper-style)
    pcas = []
    for mi, m in enumerate(models):
        Xs = []
        it_fit = iter(loader)
        for _ in tqdm(range(n_fit_imgs), desc=f"Fit PCA: {titles[mi]}"):
            try: batch = next(it_fit)
            except StopIteration: break
            img = batch["images"].to(device)
            with torch.no_grad():
                lat = m.encode(img, scale_factor=1.0)["posterior"].squeeze(0)   # (C,H,W)
                x = lat.permute(1,2,0).reshape(-1, lat.shape[0]).detach().cpu().numpy()  # (HW,C)
            if pixel_subsample and x.shape[0] > pixel_subsample:
                x = x[rng.choice(x.shape[0], pixel_subsample, replace=False)]
            Xs.append(x)
        X = np.concatenate(Xs, 0)
        pca = PCA(n_components=3, random_state=seed).fit(X)
        pcas.append(pca)

    # visualize: use each model's fixed PCA basis for all images
    it = iter(loader)
    for _ in range(n_show):
        try: batch = next(it)
        except StopIteration: break
        img = batch["images"].to(device)
        img_disp = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)

        fig, axes = plt.subplots(1, 1 + len(models), figsize=(3.2*(1+len(models)), 3.2))
        axes[0].imshow(img_disp); axes[0].set_title("Image"); axes[0].axis("off")

        for mi, m in enumerate(models):
            with torch.no_grad():
                lat = m.encode(img, scale_factor=1.0)["posterior"].squeeze(0)   # (C,H,W)
                C,H,W = lat.shape
                x = lat.permute(1,2,0).reshape(-1, C).detach().cpu().numpy()
            y = pcas[mi].transform(x).reshape(H, W, 3)
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)  # per-image minmax (matches many papers)
            axes[mi+1].imshow(y); axes[mi+1].set_title(titles[mi]); axes[mi+1].axis("off")

        plt.subplots_adjust(left=0, right=1, top=0.90, bottom=0, wspace=0.02)
        plt.show()
        plt.close(fig)


def visualize_latent_PCA(path_to_pretrained_weights=None, config_file=None,
                         dataset=None, img_sz=None, path_to_dataset=None, ):

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
            img = batch["images"].to(device)  # (1,3,H,W)
            latent = model.encode(img, scale_factor=1.0)
            latent = latent["posterior"].squeeze(0)  # (C,H,W)
            C, H, W = latent.shape
            latent = latent.cpu().numpy()

            ### PCA ###
            latent_flat = np.transpose(latent, (1, 2, 0)).reshape(-1, C)
            pca = PCA(n_components=3, random_state=42)
            latent_pca = pca.fit_transform(latent_flat)
            latent_img = latent_pca.reshape(H, W, 3)
            latent_img = (latent_img - latent_img.min()) / (latent_img.max() - latent_img.min() + 1e-8)

            ### Original image ###
            img_disp = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)

            # -----------------------------
            # Figure 1: Original Image
            # -----------------------------
            plt.figure(figsize=(4,4))
            plt.imshow(img_disp)
            plt.axis("off")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.margins(0, 0)
            plt.show()

            # -----------------------------
            # Figure 2: Latent PCA
            # -----------------------------
            plt.figure(figsize=(4,4))
            plt.imshow(latent_img)
            plt.axis("off")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            plt.margins(0, 0)
            plt.show()
            plt.close("all")



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


def spectrum_distribution(path_to_pretrained_weights=None, config_file=None, dataset=None,
                          img_sz=None, path_to_dataset=None, bs=1, max_samples=1000, n_bins=16, delta=0.0):

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
    num_iterations = target_N // bs
    sx_chunks = []
    sz_chunks = []
    loss_sum = 0.0
    n_collected = 0

    eval_iter = iter(loader)
    for n in tqdm(range(num_iterations)):
        batch = next(eval_iter)

        with torch.no_grad():
            img = batch["images"].to(device)  # (batch, 3, img_h, img_w)
            latent = model.encode(img, scale_factor=1.0)  # mean and logvar, (batch, 8, 32, 32)
            latent = latent["posterior"]  # (batch, C, H, W)

            sx, sz, kl_loss = latent_spectral_reg_dct(
                img, latent,
                blur_ks=7, blur_sigma=1.2, n_bins=n_bins,
                loss_type="kl", center="none", remove_dc=False, return_dist=True, delta=delta
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

    print(f'px: {list(np.round(sx_mean, 4))}')
    print(f'pz: {list(np.round(sz_mean, 4))}')

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



def visualize_PSD(DCT_center=False):
    # RGB image
    px_RGB_celeba256 = [0.4057, 0.2333, 0.119, 0.0835, 0.0424, 0.0317, 0.0194, 0.0142, 0.0108, 0.0091, 0.0076, 0.006, 0.0053, 0.0044, 0.004, 0.0035]

    # PSD_delta of RGB image
    px_delta_0_6 = [0.2747, 0.1931, 0.121, 0.0989, 0.061, 0.0517, 0.0361, 0.0291, 0.0242, 0.022, 0.0194, 0.0164, 0.0153, 0.0132, 0.0124, 0.0114]
    px_delta_1_0 = [0.1961, 0.156, 0.1099, 0.0977, 0.0677, 0.0617, 0.0472, 0.0404, 0.0358, 0.034, 0.0313, 0.0278, 0.0266, 0.0238, 0.0229, 0.0212]
    px_delta_1_1 = [0.1788, 0.1466, 0.1061, 0.0961, 0.0684, 0.0633, 0.0494, 0.043, 0.0386, 0.0371, 0.0345, 0.031, 0.0299, 0.0269, 0.0261, 0.0241]
    px_delta_1_2 = [0.1627, 0.1374, 0.1021, 0.0941, 0.0687, 0.0646, 0.0515, 0.0455, 0.0414, 0.0401, 0.0377, 0.0342, 0.0333, 0.0302, 0.0294, 0.0272]
    px_delta_1_3 = [0.1479, 0.1286, 0.098, 0.0918, 0.0688, 0.0655, 0.0532, 0.0477, 0.044, 0.043, 0.0408, 0.0374, 0.0366, 0.0335, 0.0328, 0.0303]

    # SDVAE
    pz_SDVAE_300k = [0.1671, 0.1312, 0.0833, 0.071, 0.0587, 0.0538, 0.0513, 0.0476, 0.046, 0.0455, 0.0438, 0.0419, 0.0408, 0.0398, 0.0393, 0.0389]

    # dowensam
    pz_downsam_380k = [0.1666, 0.1385, 0.1134, 0.0955, 0.0789, 0.0685, 0.062, 0.0493, 0.0438, 0.0393, 0.0283, 0.0494, 0.0175, 0.0163, 0.0159, 0.017]

    # DSM, blk8
    pz_DSM_440k = [0.1638, 0.1471, 0.1202, 0.1037, 0.0774, 0.0677, 0.0529, 0.0415, 0.0377, 0.0337, 0.029, 0.0306, 0.0258, 0.0242, 0.023, 0.0216]

    # RMSC, ftVAE, 1.0
    pz_RMSC_270k = [0.1813, 0.1454, 0.111, 0.0967, 0.0755, 0.0656, 0.0551, 0.0467, 0.0405, 0.0376, 0.0331, 0.0292, 0.024, 0.0209, 0.0194, 0.018]

    # ESM, ftVAE, log001, delta 0.0
    pz_delta00_290k = [0.2705, 0.2427, 0.1507, 0.1038, 0.0567, 0.041, 0.0304, 0.0223, 0.0179, 0.0159, 0.0128, 0.0107, 0.0078, 0.0063, 0.0056, 0.0049]

    # ESM, ftVAE, log001, delta 0.4
    pz_delta04_260k = [0.3176, 0.2109, 0.1225, 0.0958, 0.0556, 0.0448, 0.0304, 0.0235, 0.0189, 0.0169, 0.0145, 0.0121, 0.0108, 0.0093, 0.0086, 0.0078]

    # ESM, ftVAE, log001, delta 0.63
    pz_delta06_280k = [0.2698, 0.1903, 0.1196, 0.0991, 0.0616, 0.0522, 0.0371, 0.0298, 0.025, 0.0229, 0.0203, 0.0173, 0.016, 0.0139, 0.0131, 0.012]

    # ESM, ftVAE, log001, delta 1.0
    pz_delta10_300k = [0.1658, 0.1483, 0.113, 0.0985, 0.0754, 0.0652, 0.0531, 0.0446, 0.0387, 0.0363, 0.0329, 0.0295, 0.026, 0.0249, 0.0246, 0.0233]

    # REPA
    px_RGB_imgnet256 = [0.4657, 0.1913, 0.0997, 0.0666, 0.0388, 0.0265, 0.0217, 0.0162, 0.0131, 0.0123, 0.0103, 0.0097, 0.0076, 0.0069, 0.0067, 0.0069]
    px_delta_0_8_imgnet = [0.2622, 0.1427, 0.0977, 0.0808, 0.0582, 0.047, 0.0438, 0.0369, 0.0334, 0.0338, 0.0307, 0.0309, 0.026, 0.0249, 0.0251, 0.0258]
    px_delta_1_0_imgnet = [0.2178, 0.1268, 0.0921, 0.0797, 0.0602, 0.0505, 0.0484, 0.042, 0.0389, 0.0401, 0.0372, 0.0379, 0.0325, 0.0314, 0.032, 0.0326]
    pz_dinov2 = [0.2288, 0.1442, 0.1042, 0.0824, 0.0629, 0.0515, 0.0458, 0.0395, 0.0353, 0.0339, 0.0335, 0.03, 0.0281, 0.0273, 0.0268, 0.0258]

    # VA-VAE
    px_imgnet256 = [0.4636, 0.1917, 0.1005, 0.0668, 0.0391, 0.0266, 0.0218, 0.0163, 0.0133, 0.0124, 0.0103, 0.0096, 0.0076, 0.0069, 0.0067, 0.0068]
    px_delta_1_0_imgnet256 = [0.2158, 0.1274, 0.0928, 0.0801, 0.0607, 0.0507, 0.0486, 0.0421, 0.0391, 0.0399, 0.0371, 0.0375, 0.0322, 0.0314, 0.032, 0.0326]
    pz_vavae = [0.1893, 0.1489, 0.1356, 0.1029, 0.0707, 0.0551, 0.0508, 0.0414, 0.0336, 0.0325, 0.0331, 0.0257, 0.0219, 0.0205, 0.0198, 0.0181]

    # x-axis
    x = np.arange(1, len(px_RGB_celeba256) + 1)

    # ----- Global style -----
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    plt.figure(figsize=(9, 5))

    # ----- Colors (carefully selected) -----
    dark_gray = "#333333"  # dark gray
    muted_blue = "#4C72B0"  # muted blue
    green = "#55A868"  # green
    strong_red = "#C44E52"  # strong red

    # ----- Plot lines -----
    plt.plot(x, px_imgnet256, linewidth=2.5, color=dark_gray, label=r"RGB Image ($\delta=0.0$)")
    plt.plot(x, px_delta_1_0_imgnet256, linewidth=2.5, color=green, linestyle="--", label=r"Power-law Target ($\delta=1.0$)")
    plt.plot(x, pz_vavae, linewidth=2.5, color=strong_red, label="VA-VAE")

    # plt.plot(x, px_RGB, linewidth=2.5, color=dark_gray, label="RGB Image")
    # plt.plot(x, pz_SDVAE_300k, linewidth=2.5, color=muted_blue, label="SD-VAE")
    # plt.plot(x, pz_downsam_380k, linewidth=2.5, color=green, label="Scale Equivariance")
    # plt.plot(x, pz_DSM_440k, linewidth=2.5, color=strong_red, label="DSM-AE")

    # ----- Labels -----
    plt.xlabel("Frequency Index")
    plt.ylabel("Probability Density")
    plt.title("Spectrum Distribution")

    plt.grid(alpha=0.25)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # visualize_latent_PCA(
    #     path_to_pretrained_weights='/home/mang/Downloads/celeba256_b48_f16_ESM_delta10_noDC_ftVAE_log001/SDVAE/checkpoint_300000/model.safetensors',
    #     config_file='configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #     path_to_dataset='/home/mang/Downloads/celeba256/celeba256_visual',
    # )

    # weight_paths = [
    #     '/home/mang/Downloads/SM/celeba256_SDVAE_bf16_b48_f16_flip_400k/SDVAE/checkpoint_300000/model.safetensors',
    #     '/home/mang/Downloads/SM/celeba256_SDVAE_b48_f16_downsam/SDVAE/checkpoint_300000/model.safetensors',
    #     '/home/mang/Downloads/SM/celeba256_b48_f16_ESM_delta10_noDC_ftVAE_log001/SDVAE/checkpoint_300000/model.safetensors',
    #     '/home/mang/Downloads/SM/celeba256_b48_f16_DSM_blk8/SDVAE/checkpoint_440000/model.safetensors',
    # ]
    #
    # visualize_latent_pca_paperstyle(
    #     weight_paths,
    #     config_file='configs/ldm_f16d16.yaml',
    #     dataset="celeba256",
    #     img_sz=256,
    #     path_to_dataset='/home/mang/Downloads/celeba256/celeba256_visual',
    #     titles=("SD-VAE", "Scale Equivariance", "ESM", "DSM"),
    #     n_fit_imgs=1024,
    #     pixel_subsample=8192,
    #     n_show=24,
    # )

    # weight_paths = [
    #     '/home/mang/Downloads/SM/celeba256_SDVAE_b48_f8/SDVAE/checkpoint_450000/model.safetensors',
    #     '/home/mang/Downloads/SM/celeba256_SDVAE_b48_f8_downsam/SDVAE/checkpoint_360000/model.safetensors',
    #     '/home/mang/Downloads/SM/celeba256_b48_f8_ESM_delta12_noDC_ftVAE_log001/SDVAE/checkpoint_250000/model.safetensors',
    #     '/home/mang/Downloads/SM/celeba256_b48_f8_DSM_blk8_81012/SDVAE/checkpoint_330000/model.safetensors',
    # ]
    #
    # visualize_latent_pca_paperstyle(
    #     weight_paths,
    #     config_file='configs/ldm_f8d4.yaml',
    #     dataset="celeba256",
    #     img_sz=256,
    #     path_to_dataset='/home/mang/Downloads/celeba256/celeba256_visual',
    #     titles=("SD-VAE", "Scale Equivariance", "ESM", "DSM"),
    #     n_fit_imgs=1024,
    #     pixel_subsample=8192,
    #     n_show=24,
    # )

    # spectrum_distribution(
    #     path_to_pretrained_weights='/leonardo_work/EUHPC_B29_014/LDM_exps/celeba256_b48_f16_ESM_delta10_noDC_ftVAE_log001/SDVAE/checkpoint_300000/model.safetensors',
    #     config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #     path_to_dataset='/leonardo_work/EUHPC_B29_014/datasets/celeba256/celeba256',
    #     bs=100, max_samples=30000, n_bins=16, delta=0.0
    # )

    visualize_PSD(DCT_center=False)

    # for ckpt in [280, 360, 400, 460]:
    #     spectrum_loss(
    #         path_to_pretrained_weights=f'/leonardo_work/EUHPC_B29_014/LDM_exps/celeba256_SM_b48_f16_16bins_log001_DSM_blk8/SDVAE/checkpoint_{ckpt}000/model.safetensors',
    #         config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #         path_to_dataset='/leonardo_work/EUHPC_B29_014/datasets/celeba256/celeba256',
    #         bs=100, max_samples=10000
    #     )

    # for i in [0, 2, 4, 6, 8]:
    #     for ckpt in [230, 290, 310]:
    #         lowpass_rFID(
    #             path_to_pretrained_weights=f'/leonardo_work/EUHPC_B29_014/LDM_exps/celeba256_SM_b48_f16_16bins_log001_ftVAE/SDVAE/checkpoint_{ckpt}000/model.safetensors',
    #             config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #             path_to_dataset='/leonardo_work/EUHPC_B29_014/datasets/celeba256/celeba256',
    #             path_to_save_imgs='/leonardo_work/EUHPC_B29_014',
    #             batch_size=100, max_samples=10000, blk_sz=8, k=i
    #         )

    # for ckpt in [280, 360, 400, 460]:
    #     rmsc_loss(
    #         path_to_pretrained_weights=f'/leonardo_work/EUHPC_B29_014/LDM_exps/celeba256_SM_b48_f16_16bins_log001_DSM_blk8/SDVAE/checkpoint_{ckpt}000/model.safetensors',
    #         config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml', dataset='celeba256', img_sz=256,
    #         path_to_dataset='/leonardo_work/EUHPC_B29_014/datasets/celeba256/celeba256',
    #         bs=100, max_samples=30000
    #     )

    # lowpass_rFID(path_to_pretrained_weights='/leonardo_work/EUHPC_B29_014/LDM_exps/imagenet256_SDVAE_bf16_b128_f16_600k/SDVAE/checkpoint_410000/model.safetensors',
    #              config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml',
    #              batch_size=100, dataset="imagenet_train", img_sz=256,
    #              path_to_dataset="/leonardo_work/EUHPC_B29_014/datasets/imagenet256/train",
    #              path_to_save_imgs='/leonardo_work/EUHPC_B29_014', num_workers=12, max_samples=50000, k=2, blk_sz=8)