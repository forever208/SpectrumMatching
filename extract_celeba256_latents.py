import os
import yaml
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader
from safetensors.torch import load_file
import numpy as np
import glob

from modules import LDMConfig, VAE
from dataset import get_dataset


def extract_latent(pretrained_weights, config_file, batch_size, dataset, path_to_dataset, path_to_latents):
    print(f"extract image latents for {dataset} from {pretrained_weights}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ### Load VAE Config ###
    with open(config_file, "r") as f:
        vae_config = yaml.safe_load(f)
        config = LDMConfig(**vae_config["vae"])

    ### Load Model and weights ###
    model = VAE(config)
    state_dict = load_file(pretrained_weights)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)

    ### Load Dataset (include non-flip and flip datasets) ###
    org_dataset, _ = get_dataset(dataset=dataset, path_to_data=path_to_dataset, train=False, random_flip_p=0.0)
    flip_dataset, _ = get_dataset(dataset=dataset, path_to_data=path_to_dataset, train=False, random_flip_p=1.0)
    combined_dataset = ConcatDataset([org_dataset, flip_dataset])
    loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                        num_workers=8, pin_memory=False, persistent_workers=True)
    samples = len(combined_dataset)
    print(f"found {samples} samples in {dataset}")

    idx = 0
    all_latents = []
    os.makedirs(path_to_latents, exist_ok=True)
    for batch in tqdm(loader):
        with torch.no_grad():
            img = batch["images"].to(device)
            moments = model.forward_enc(img)  # mean and logvar, (batch, 8, 32, 32)

            mu, logvar = torch.chunk(moments, chunks=2, dim=1)  # get mean and logvar (batch, 4, 32, 32)
            logvar = torch.clamp(logvar, min=-30.0, max=20.0)  # Clamp Logvar for numerical stability
            sigma = torch.exp(0.5 * logvar)  # std
            noise = torch.randn_like(sigma, device=sigma.device, dtype=sigma.dtype)
            latent = mu + sigma * noise  # (batch, 4, 32, 32)

            all_latents.append(latent.cpu())
            moments = moments.detach().cpu().numpy()

            for moment in moments:
                np.save(f'{path_to_latents}/{idx}.npy', moment)
                idx += 1

    all_latents = torch.cat(all_latents, dim=0)  # (N, 4, 32, 32)
    np.save(f"{path_to_latents}.npy", all_latents.numpy())
    print(f"posterior latents saved into {path_to_latents}.npy with shape {all_latents.shape}")
    print(f'saved {idx} latent npy files into {path_to_latents}')


def get_scaling_bound(latent_path=None, pct=99.9):
    # load latent posterior for statistics
    all_latents = np.load(latent_path)  # shape: (N, C, H, W)
    print(f"loading latents from {latent_path}")
    print(f"loaded all posterior latents with shape: {all_latents.shape}")

    # Flatten all values
    flat_latents = all_latents.flatten()  # (N*C*h*w)

    # Compute global statistics
    std_value =  np.std(flat_latents)
    mean_value = np.mean(flat_latents)
    min_value = np.min(flat_latents)
    max_value = np.max(flat_latents)
    upper_bound = np.percentile(flat_latents, pct)
    lower_bound = np.percentile(flat_latents, 100.0-pct)
    bound = max(abs(upper_bound), abs(lower_bound))

    print(f"Mean: {mean_value:.6f}")
    print(f"Min: {min_value:.6f}")
    print(f"Max: {max_value:.6f}")
    print(f"Std: {std_value:.6f}")
    print(f"{pct}th Percentile: {upper_bound:.6f}")
    print(f"{100.0-pct}th Percentile: {lower_bound:.6f}")
    print(f"pct scaling facor is {1 / bound}")
    print(f"std scaling facor is {1 / std_value}")


if __name__ == "__main__":
    # extract_latent(
    #     pretrained_weights='/leonardo_work/EUHPC_B29_014/LDM_exps/celeba256_SDVAE_bf16_b48_f16d16_flip_SM_16bins_KLx1/SDVAE/checkpoint_400000/model.safetensors',
    #     config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml', batch_size=32, dataset='celeba256',
    #     path_to_dataset='/leonardo_work/EUHPC_B29_014/datasets/celeba256/celeba256',
    #     path_to_latents='/leonardo_work/EUHPC_B29_014/datasets/celeba256_latents/celeba256_SM_f16_latents_16bins_noDC_KLx1_400k'
    # )

    get_scaling_bound(latent_path='/leonardo_work/EUHPC_B29_014/datasets/celeba256_latents/celeba256_SM_f16_latents_16bins_noDC_KLx1_400k.npy',
                      pct=99.9)