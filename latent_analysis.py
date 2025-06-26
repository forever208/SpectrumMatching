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
from utils_DCT import latent_spectral_reg_dct, split_into_blocks_torch, combine_blocks_torch, dct_2d_torch_unified, idct_2d_torch_unified
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
                n_bins=n_bins, loss_type="kl", center="none", remove_dc=False, return_dist=True, delta=delta
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



def visualize_PSD(DCT_center=False):
    # RGB image
    px_RGB_celeba256 = [0.4057, 0.2333, 0.119, 0.0835, 0.0424, 0.0317, 0.0194, 0.0142, 0.0108, 0.0091, 0.0076, 0.006, 0.0053, 0.0044, 0.004, 0.0035]

    # PSD_delta of RGB image
    px_delta_0_6 = [0.2747, 0.1931, 0.121, 0.0989, 0.061, 0.0517, 0.0361, 0.0291, 0.0242, 0.022, 0.0194, 0.0164, 0.0153, 0.0132, 0.0124, 0.0114]
    px_delta_1_0 = [0.1961, 0.156, 0.1099, 0.0977, 0.0677, 0.0617, 0.0472, 0.0404, 0.0358, 0.034, 0.0313, 0.0278, 0.0266, 0.0238, 0.0229, 0.0212]
    px_delta_1_2 = [0.1627, 0.1374, 0.1021, 0.0941, 0.0687, 0.0646, 0.0515, 0.0455, 0.0414, 0.0401, 0.0377, 0.0342, 0.0333, 0.0302, 0.0294, 0.0272]

    # SDVAE
    pz_SDVAE = [0.1671, 0.1312, 0.0833, 0.071, 0.0587, 0.0538, 0.0513, 0.0476, 0.046, 0.0455, 0.0438, 0.0419, 0.0408, 0.0398, 0.0393, 0.0389]

    # downsam
    pz_downsam = [0.1666, 0.1385, 0.1134, 0.0955, 0.0789, 0.0685, 0.062, 0.0493, 0.0438, 0.0393, 0.0283, 0.0494, 0.0175, 0.0163, 0.0159, 0.017]

    # DSM, blk8
    pz_DSM = [0.1638, 0.1471, 0.1202, 0.1037, 0.0774, 0.0677, 0.0529, 0.0415, 0.0377, 0.0337, 0.029, 0.0306, 0.0258, 0.0242, 0.023, 0.0216]

    # ESM, ftVAE, log001, delta 1.0
    pz_delta10 = [0.1658, 0.1483, 0.113, 0.0985, 0.0754, 0.0652, 0.0531, 0.0446, 0.0387, 0.0363, 0.0329, 0.0295, 0.026, 0.0249, 0.0246, 0.0233]

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
    # plt.plot(x, pz_SDVAE, linewidth=2.5, color=muted_blue, label="SD-VAE")
    # plt.plot(x, pz_downsam, linewidth=2.5, color=green, label="Scale Equivariance")
    # plt.plot(x, pz_DSM, linewidth=2.5, color=strong_red, label="DSM-AE")

    # ----- Labels -----
    plt.xlabel("Frequency Index")
    plt.ylabel("Probability Density")
    plt.title("Spectrum Distribution")

    plt.grid(alpha=0.25)
    plt.legend(frameon=False)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    visualize_PSD(DCT_center=False)