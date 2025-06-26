import torch.nn.functional as F
import torch_dct as dct
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
from pathlib import Path


def split_into_blocks_torch(image: torch.Tensor, block_sz: int):
    """
    Split a 2D tensor (H, W) or batched 3D/4D tensor (B, H, W) into non-overlapping (block_sz x block_sz) blocks.

    Args:
        image (Tensor): shape (H, W) or (B, H, W) or (B, C, H, W)
        block_sz (int): block size

    Returns:
        Tensor:
            - (N_blocks, block_sz, block_sz) if input is (H, W)
            - (B, N_blocks, block_sz, block_sz) if input is (B, H, W)
    """
    if image.dim() == 2:  # (H, W)
        H, W = image.shape
        assert H % block_sz == 0 and W % block_sz == 0
        blocks = image.unfold(0, block_sz, block_sz).unfold(1, block_sz, block_sz)  # (H/b, W/b, b, b)
        return blocks.contiguous().view(-1, block_sz, block_sz)  # (N_blocks, b, b)

    elif image.dim() == 3:  # (B, H, W)
        B, H, W = image.shape
        assert H % block_sz == 0 and W % block_sz == 0
        blocks = image.unfold(1, block_sz, block_sz).unfold(2, block_sz, block_sz)  # (B, H/b, W/b, b, b)
        blocks = blocks.contiguous().view(B, -1, block_sz, block_sz)  # (B, N_blocks, b, b)
        return blocks

    elif image.dim() == 4:  # (B, C, H, W)
        B, C, H, W = image.shape
        assert H % block_sz == 0 and W % block_sz == 0
        blocks = image.unfold(2, block_sz, block_sz).unfold(3, block_sz, block_sz)  # (B, C, H/b, W/b, b, b)
        blocks = blocks.contiguous().view(B, C, -1, block_sz, block_sz)  # (B, C, N_blocks, b, b)
        return blocks

    else:
        raise ValueError(f"Input tensor must be 2D or 3D or 4D, got shape {image.shape}")


def combine_blocks_torch(blocks: torch.Tensor, height: int, width: int, block_sz: int):
    """
    Combine non-overlapping blocks into full image.

    Args:
        blocks:
            - (N, B, B) tensor (single image)
            - (batch, N, B, B) tensor (batched images)
            - (batch, C, N, B, B) tensor (batched multi-channel images)
        height: original image height
        width: original image width
        block_sz: size of each block (B)

    Returns:
        image:
            - (height, width) if input is 3D
            - (batch, height, width) if input is 4D
            - (batch, C, height, width) if input is 5D
    """
    blocks_per_row = width // block_sz
    blocks_per_col = height // block_sz

    if blocks.dim() == 3:  # (N, B, B)
        image = blocks.view(blocks_per_col, blocks_per_row, block_sz, block_sz)
        image = image.permute(0, 2, 1, 3).reshape(height, width)
        return image
    elif blocks.dim() == 4:  # (batch, N, B, B)
        B = blocks.size(0)
        image = blocks.view(B, blocks_per_col, blocks_per_row, block_sz, block_sz)
        image = image.permute(0, 1, 3, 2, 4).reshape(B, height, width)
        return image
    elif blocks.dim() == 5:  # (batch, C, N, B, B)
        B = blocks.size(0)
        C = blocks.size(1)
        image = blocks.view(B, C, blocks_per_col, blocks_per_row, block_sz, block_sz)
        image = image.permute(0, 1, 2, 4, 3, 5).reshape(B, C, height, width)
        return image
    else:
        raise ValueError(f"Expected input of shape (N, B, B) or (batch, N, B, B) or (batch, C, N, B, B), but got {blocks.shape}")


def idct_2d_torch_unified(X: torch.Tensor, center: str = "none", mean: torch.Tensor = None):
    """
    Inverse of dct_2d_torch_unified using 2D IDCT (Type-III) with ortho norm.

    X: (..., H, W)  DCT coefficients from dct_2d_torch_unified
    center:
      - "none": exact inverse if forward used center="none"
      - "mean": exact inverse if forward used center="mean" AND you pass `mean`
    mean:
      - required when center="mean": the mean that was subtracted in forward,
        shape broadcastable to (..., 1, 1)

    returns: (..., H, W)
    """
    X = X.float()

    # Invert separable 2D DCT:
    # Forward: dct(last) -> transpose -> dct(last) -> transpose back
    # Inverse: idct(last) -> transpose -> idct(last) -> transpose back
    x = dct.idct(X, norm="ortho")                          # inverse along last dim
    x = dct.idct(x.transpose(-2, -1), norm="ortho")        # inverse along second-last dim
    x = x.transpose(-2, -1)

    if center == "none":
        return x
    elif center == "mean":
        if mean is None:
            raise ValueError("center='mean' requires `mean` (the value subtracted in forward).")
        return x + mean
    else:
        raise ValueError("center must be 'mean' or 'none'")


# ============================================================
# 1) Unified 2D DCT for inputs roughly in [-1, 1]
# ============================================================
def dct_2d_torch_unified(x: torch.Tensor, center: str = "mean"):
    """
    2D DCT-II (ortho) for inputs in roughly [-1, 1], works for RGB or latents.

    x: (..., H, W)
    center:
      - "mean": subtract spatial mean per sample (recommended)
      - "none": no centering (DC can dominate)
    returns: (..., H, W)
    """
    x = x.float()
    if center == "mean":
        x = x - x.mean(dim=(-2, -1), keepdim=True)
    elif center == "none":
        pass
    else:
        raise ValueError("center must be 'mean' or 'none'")

    x = dct.dct(x, norm="ortho")                       # last dim
    x = dct.dct(x.transpose(-2, -1), norm="ortho")     # second-last
    return x.transpose(-2, -1)


# ============================================================
# 2) Gaussian blur (depthwise)
# ============================================================
def gaussian_kernel2d(kernel_size: int, sigma: float, device=None, dtype=None):
    assert kernel_size % 2 == 1, "kernel_size should be odd"
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size // 2)
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, kernel_size: int = 7, sigma: float = 1.2):
    """
    x: (B,C,H,W) -> blurred x: (B,C,H,W)
    """
    B, C, H, W = x.shape
    k = gaussian_kernel2d(kernel_size, sigma, device=x.device, dtype=x.dtype)
    k = k.view(1, 1, kernel_size, kernel_size).repeat(C, 1, 1, 1)  # (C,1,K,K)
    pad = kernel_size // 2
    return F.conv2d(x, k, padding=pad, groups=C)


def downsample_to(x: torch.Tensor, size_hw: tuple[int, int]):
    return F.interpolate(x, size=size_hw, mode="bicubic", align_corners=False)


# ============================================================
# 3) Channel-aggregated DCT power spectrum
# ============================================================
def channel_agg_power_dct_unified(x: torch.Tensor, center: str = "mean",
                                  remove_dc: bool = True, eps: float = 1e-12):
    """
    x: (B,C,H,W), values ~ [-1,1]
    returns: P: (B,H,W) = mean_c (DCT(x)^2)
    """
    B, C, H, W = x.shape
    x_ = x.reshape(B * C, H, W)                # treat channels as batch for later 2D-DCT
    X = dct_2d_torch_unified(x_, center=center)
    X = X.view(B, C, H, W)
    P = (X ** 2).mean(dim=1).clamp_min(eps)    # average over channels for each sample, (batch, H, W)
    if remove_dc:
        P[:, 0, 0] = 0.0
    return P


# ============================================================
# 4) Fast radial band pooling using scatter_add
# ============================================================
def radial_bin_map(H: int, W: int, n_bins: int, device=None):
    yy = torch.arange(H, device=device).view(H, 1).float()
    xx = torch.arange(W, device=device).view(1, W).float()
    rr = torch.sqrt(yy**2 + xx**2)                     # DC at (0,0)
    rmax = rr.max().clamp_min(1.0)
    bin_id = torch.floor(rr / rmax * n_bins).long()
    return bin_id.clamp(0, n_bins - 1)                 # (H,W)


def radial_band_energy(P: torch.Tensor, n_bins: int = 16, eps: float = 1e-8):
    """
    P: (B,H,W) -> s: (B,n_bins) mean power per radial bin
    """
    B, H, W = P.shape
    bin_id = radial_bin_map(H, W, n_bins, device=P.device).view(-1)  # (H*W,) [0, 1, 2, 3...]
    counts = torch.bincount(bin_id, minlength=n_bins).to(P.dtype).clamp_min(1.0)  # (n_bins,)

    P_flat = P.view(B, -1)                                  # (B, H*W)
    idx = bin_id.view(1, -1).expand(B, -1)                   # (B, H*W)

    s_sum = torch.zeros(B, n_bins, device=P.device, dtype=P.dtype)
    s_sum = s_sum.scatter_add(1, idx, P_flat)                # sum up the PSD in each bin
    s_mean = s_sum / (counts.view(1, -1) + eps)
    return s_mean



# ============================================================
# 5) Latent spectral regularizer
# ============================================================
def latent_spectral_reg_dct(
    x: torch.Tensor,              # (B,3,256,256) ~ [-1,1]
    z: torch.Tensor,              # (B,C,h,w)     ~ [-1,1] (or any range; DCT uses mean-centering)
    blur_ks: int = 7,
    blur_sigma: float = 1.2,
    n_bins: int = 16,
    loss_type: str = "kl",        # "l2" or "kl"
    center: str = "mean",         # DCT centering mode
    log_power: bool = True,
    remove_dc: bool = True,
    eps: float = 1e-8,
    return_dist: bool = False,
    delta: float = 0.0,
):
    B, Cz, hz, wz = z.shape

    # 1) anti-alias then downsample x to match z spatial size
    if hz == 32:
        x_blur = x
    else:
        x_blur = gaussian_blur(x, kernel_size=blur_ks, sigma=blur_sigma)
    x_ds = downsample_to(x_blur, (hz, wz))

    # 2) channel-aggregated DCT power spectra (both in [-1,1] domain)
    Px = channel_agg_power_dct_unified(x_ds, center=center, remove_dc=remove_dc, eps=eps)  # (B, hz, wz)
    Pz = channel_agg_power_dct_unified(z, center=center, remove_dc=remove_dc, eps=eps)  # (B, hz, wz)

    # 3) group spectrum into num_bins, sum the PSD into each bin
    sx = radial_band_energy(Px, n_bins=n_bins, eps=eps)  # (B, n_bins)
    sz = radial_band_energy(Pz, n_bins=n_bins, eps=eps)  # (B, n_bins)

    # flatten the PSD, which will be the target of z to match, e.g. turn sx (1/f^2) into proxy sx' (1/f^1.6) ----
    f = torch.arange(1, n_bins + 1, device=sx.device, dtype=sx.dtype).view(1, -1)
    sx = sx * (f ** delta)  # target PSD is 1/f^2 if delta=0, target PSD is 1/f^1.6 if delta=0.4

    if log_power:
        sx = torch.log(sx + eps + 1.0)
        sz = torch.log(sz + eps + 1.0)

    # 4) normalize to distributions (scale-invariant), each bin takes up how much proportion of PSD
    sx = sx.clamp_min(0.0)
    sz = sz.clamp_min(0.0)
    sx = sx / (sx.sum(dim=1, keepdim=True) + eps)
    sz = sz / (sz.sum(dim=1, keepdim=True) + eps)

    if return_dist:
        kl_loss = (sx * (torch.log(sx + eps) - torch.log(sz + eps))).sum(dim=1).mean()
        return sx, sz, kl_loss  # sx/sz has shape (B, n_bins)
    else:
        if loss_type == "l2":
            return F.mse_loss(sz, sx)
        elif loss_type == "kl":
            return (sx * (torch.log(sx + eps) - torch.log(sz + eps))).sum(dim=1).mean()
        else:
            raise ValueError("loss_type must be 'l2' or 'kl'")


def back_prop_debug():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(2, 3, 256, 256, device=device) * 2 - 1  # Fake inputs in [-1,1]
    # z_64 = torch.randn(2, 4, 64, 64, device=device, requires_grad=True).tanh()     # example latent in [-1,1]
    # z_32 = torch.randn(2, 16, 32, 32, device=device, requires_grad=True).tanh()    # example latent in [-1,1]

    z_64 = torch.randn(2, 4, 64, 64, device=device, requires_grad=True)  # example latent in [-1,1]
    z_32 = torch.randn(2, 16, 32, 32, device=device, requires_grad=True)  # example latent in [-1,1]

    # Suggested blur settings:
    loss_64 = latent_spectral_reg_dct(
        x, z_64,
        blur_ks=7, blur_sigma=1.2, n_bins=16,
        loss_type="l2", log_power=True, center="mean", remove_dc=True
    )

    loss_32 = latent_spectral_reg_dct(
        x, z_32,
        blur_ks=11, blur_sigma=2.2, n_bins=16,
        loss_type="l2", log_power=True, center="mean", remove_dc=True
    )

    total = loss_64 + loss_32
    print("loss(64x64x4):", float(loss_64))
    print("loss(32x32x16):", float(loss_32))
    print("total:", float(total))

    total.backward()
    print("backward OK")


def load_image_as_tensor_neg1_pos1(image_path: str, device: Optional[str] = None) -> torch.Tensor:
    """
    Load an image from disk and return a tensor in [-1, 1] with shape (1,3,H,W).
    """
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0               # (H,W,3) in [0,1]
    ten = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)    # (1,3,H,W)
    ten = ten * 2.0 - 1.0                                        # [-1,1]
    if device is not None:
        ten = ten.to(device)
    return ten


@torch.no_grad()
def visualize_blur_and_sx_from_path(
    image_path: str,
    blur_ks: int = 7,
    blur_sigma: float = 1.2,
    n_bins: int = 16,
    center: str = "mean",
    remove_dc: bool = True,
    eps: float = 1e-8,
    downsample_hw: Optional[Tuple[int, int]] = None,  # e.g. (64,64) to mimic latent size
    log_power_for_sx: bool = False,                   # keep False to avoid clamp->zero issues
    show_log_power_heatmap: bool = True,              # extra panel: show log(Px)
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Visualizes:
      1) original image (maybe downsampled)
      2) blurred image (maybe downsampled)
      3) (log) power spectrum heatmap Px (from blurred image)
      4) sx before/after normalization

    Requires your previously defined functions:
      - gaussian_blur, downsample_to
      - channel_agg_power_dct_unified, radial_band_energy
    """
    x = load_image_as_tensor_neg1_pos1(image_path, device=device)  # (1,3,H,W)

    # blur
    x_blur = gaussian_blur(x, kernel_size=blur_ks, sigma=blur_sigma)

    # optional downsample (to match latent spatial size)
    if downsample_hw is not None:
        x_vis = downsample_to(x, downsample_hw)
        x_blur_vis = downsample_to(x_blur, downsample_hw)
    else:
        x_vis = x
        x_blur_vis = x_blur

    # Px on the blurred (and maybe downsampled) image
    Px = channel_agg_power_dct_unified(x_blur_vis, center=center, remove_dc=remove_dc, eps=eps)  # (1,h,w)

    # sx (optional log power for sx, usually keep False)
    Px_for_sx = torch.log(Px + eps) if log_power_for_sx else Px
    sx_raw = radial_band_energy(Px_for_sx, n_bins=n_bins, eps=eps)  # (1,n_bins)
    sx_norm = sx_raw.clamp_min(0.0)
    sx_norm = sx_norm / (sx_norm.sum(dim=1, keepdim=True) + eps)

    # convert images to [0,1] for plotting
    def to_img01(t):
        t = t[0]  # (3,h,w)
        t = (t + 1.0) * 0.5
        return t.clamp(0, 1).permute(1, 2, 0).cpu()

    img0 = to_img01(x_vis)
    img1 = to_img01(x_blur_vis)

    # spectrum heatmap: use log(Px) for visibility
    Px_heat = torch.log(Px[0] + eps).detach().cpu()  # (h,w)

    # plotting: 4 panels
    fig = plt.figure(figsize=(16, 4))

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(img0)
    ax1.set_title("Original (maybe downsampled)")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(img1)
    ax2.set_title(f"Blurred (ks={blur_ks}, sigma={blur_sigma})")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 4, 3)
    im = ax3.imshow(Px_heat, origin="upper")
    ax3.set_title("log(Px) heatmap (DCT power)")
    ax3.set_xlabel("v (freq index)")
    ax3.set_ylabel("u (freq index)")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(1, 4, 4)
    bins = torch.arange(n_bins).cpu()
    ax4.plot(bins, sx_raw[0].detach().cpu(), marker="o", label="sx raw")
    ax4.plot(bins, sx_norm[0].detach().cpu(), marker="o", label="sx normalized (sum=1)")
    ax4.set_xlabel("Radial frequency bin (low → high)")
    ax4.set_ylabel("Band energy")
    ax4.set_title(f"sx (n_bins={n_bins}, log_power_for_sx={log_power_for_sx})")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {"x": x, "x_blur": x_blur, "Px": Px, "sx_raw": sx_raw, "sx_norm": sx_norm}



def read_64x64_rgb_and_dct(image_path, center="none", device="cpu"):
    """
    Read an image, convert to RGB 64x64, map to [-1, 1], then apply 2D DCT per channel.
    Returns:
        dct_coeffs: torch.Tensor of shape (3, 64, 64) on `device`, dtype float32.
    """
    image_path = Path(image_path)

    # 1) Load and enforce RGB + 64x64
    img = Image.open(image_path).convert("RGB")
    if img.size != (64, 64):
        img = img.resize((64, 64), resample=Image.BICUBIC)

    # 2) To torch: (H, W, C) uint8 -> float32 in [0, 255], then (C, H, W)
    x = torch.from_numpy(__import__("numpy").array(img))  # (64, 64, 3), uint8
    x = x.to(device=device, dtype=torch.float32)
    x = x / 127.5 - 1.0  #
    x = x.permute(2, 0, 1).contiguous()  # (3, 64, 64)

    # 3) Apply 2D DCT channel-wise: treat C as batch dim -> (..., H, W)
    coeffs = dct_2d_torch_unified(x, center=center)  # (3, 64, 64)
    print(coeffs[0, :4, :4])


def rmsc(x: torch.Tensor, patch_sz: int = 1, eps: float = 1e-12) -> torch.Tensor:
    """
    RMS Spatial Contrast (RMSC) where each patch token is formed by *stacking*
    a (patch_sz x patch_sz) block into the channel dimension (no pooling).

    Args:
        x: Tensor (B, C, H, W)
        patch_sz: 1, 2, 4, ... must divide H and W
        eps: numerical stability

    Returns:
        Tensor (B,)
    """
    if x.dim() != 4:
        raise ValueError(f"x must be 4D (B,C,H,W), got {tuple(x.shape)}")
    if patch_sz <= 0:
        raise ValueError(f"patch_sz must be positive, got {patch_sz}")

    B, C, H, W = x.shape
    if (H % patch_sz) != 0 or (W % patch_sz) != 0:
        raise ValueError(f"patch_sz={patch_sz} must divide H and W, got H={H}, W={W}")

    Hp, Wp = H // patch_sz, W // patch_sz
    T = Hp * Wp

    if patch_sz == 1:
        # tokens: (B, C, T)
        xt = x.reshape(B, C, T)
    else:
        # Split into non-overlapping patches and stack patch pixels into channel dim:
        # (B, C, H, W)
        # -> (B, C, Hp, patch_sz, Wp, patch_sz)
        # -> (B, Hp, Wp, C, patch_sz, patch_sz)
        # -> (B, Hp*Wp, C*patch_sz*patch_sz)
        # -> (B, C*patch_sz*patch_sz, T)
        xt = (
            x.view(B, C, Hp, patch_sz, Wp, patch_sz)
             .permute(0, 2, 4, 1, 3, 5)
             .reshape(B, T, C * patch_sz * patch_sz)
             .transpose(1, 2)
        )

    # L2-normalize each token vector over "channel" dim: (B, C', T)
    xt_hat = xt / xt.norm(p=2, dim=1, keepdim=True).clamp_min(eps)

    # Mean of normalized tokens across spatial locations: (B, C', 1)
    x_bar = xt_hat.mean(dim=2, keepdim=True)

    # Per-token squared L2 distance to mean: (B, T)
    sq_dist = (xt_hat - x_bar).pow(2).sum(dim=1)

    # RMSC per sample: (B,)
    return torch.sqrt(sq_dist.mean(dim=1).clamp_min(eps))


if __name__ == "__main__":
    # back_prop_debug()
    # read_64x64_rgb_and_dct("/home/mang/Downloads/ffhq256/ffhq256/00002.jpg", center="none", device="cpu")

    out = visualize_blur_and_sx_from_path(
        "/home/mang/Downloads/ffhq256/ffhq256/00000.jpg",
        blur_ks=7, blur_sigma=1.2,
        downsample_hw=(64, 64),
        n_bins=16,
        log_power_for_sx=False,
        device="cuda"
    )