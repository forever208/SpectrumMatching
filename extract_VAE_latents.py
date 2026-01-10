import os
import yaml
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader
from safetensors.torch import load_file
import numpy as np
import glob
from accelerate import Accelerator
import math
from modules import LDMConfig, VAE
from dataset import get_dataset


def extract_latent(pretrained_weights, config_file, batch_size, dataset, path_to_dataset, path_to_latents, num_stat_samples):
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
    loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                        num_workers=8, pin_memory=True, persistent_workers=True)
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

            if idx <= num_stat_samples:
                all_latents.append(latent.float().cpu())
            moments = moments.detach().cpu().numpy()

            for moment in moments:
                np.save(f'{path_to_latents}/{idx}.npy', moment)
                idx += 1

    all_latents = torch.cat(all_latents, dim=0).numpy()  # (N, 4, 32, 32)
    flat_latents = all_latents.flatten() # (N*C*h*w)
    mean_value = np.mean(flat_latents)
    std_value = np.std(flat_latents)
    print(f"latent stat over {all_latents.shape[0]} samples: mean is {mean_value} std scaling facor is {1 / std_value}")
    print(f'saved {idx} m and logvar npy files into {path_to_latents}')



def extract_latent_ddp(pretrained_weights, config_file, batch_size, dataset, path_to_dataset, path_to_latents, num_stat_samples):

    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"extract image latents for {dataset} from {pretrained_weights}")
    accelerator.print(f"accelerate: world_size={accelerator.num_processes}, rank={accelerator.process_index}")

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
    loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                        num_workers=12, pin_memory=True, persistent_workers=True)

    # prepare
    model, loader = accelerator.prepare(model, loader)
    base_model = accelerator.unwrap_model(model)  # for forward_enc() inference

    if accelerator.is_main_process:
        accelerator.print(f"found {len(combined_dataset)} samples in {dataset}")
        os.makedirs(path_to_latents, exist_ok=True)
    accelerator.wait_for_everyone()

    # split the requested stat samples across processes to keep total about num_stat_samples
    per_rank_target = int(math.ceil(num_stat_samples / max(accelerator.num_processes, 1)))
    stat_taken = 0
    sum_x = torch.tensor(0.0, device=device)
    sum_x2 = torch.tensor(0.0, device=device)
    cnt = torch.tensor(0.0, device=device)

    # saving: avoid collisions by using "rank + world_size * local_i" indexing
    local_i = 0
    rank = accelerator.process_index
    world = accelerator.num_processes

    for batch in tqdm(loader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            img = batch["images"].to(device, non_blocking=True)
            class_label = batch["class_conditioning"]  # could be tensor/list/etc.

            moments = base_model.forward_enc(img)  # mean and logvar, (batch, 8, 32, 32)
            mu, logvar = torch.chunk(moments, chunks=2, dim=1)  # get mean and logvar (batch, 4, 32, 32)
            logvar = torch.clamp(logvar, min=-30.0, max=20.0)  # Clamp Logvar for numerical stability
            sigma = torch.exp(0.5 * logvar)  # std
            noise = torch.randn_like(sigma, device=sigma.device, dtype=sigma.dtype)
            latent = mu + sigma * noise  # (batch, 4, 32, 32)

            # accumulate stats on a limited number of samples (per rank)
            if stat_taken < per_rank_target:
                need = per_rank_target - stat_taken
                take = min(need, latent.shape[0])
                x = latent[:take].float().reshape(-1)
                sum_x += x.sum()
                sum_x2 += (x * x).sum()
                cnt += x.numel()
                stat_taken += take

            # save moments and labels (unique index per process)
            moments_np = moments.detach().cpu().numpy()

            if torch.is_tensor(class_label):
                labels_np = class_label.detach().cpu().numpy()  # (B, ...) or (B,)
            else:
                labels_np = np.asarray(class_label)  # try best; could be list
            assert len(moments_np) == len(labels_np), f"batch mismatch: moments={len(moments_np)} labels={len(labels_np)}"

            for i in range(len(moments_np)):
                out_idx = rank + world * local_i
                z_i = moments_np[i]
                label_i = labels_np[i]
                payload = np.array([z_i, label_i], dtype=object)  # IMPORTANT for unpacking later
                np.save(os.path.join(path_to_latents, f"{out_idx}.npy"), payload, allow_pickle=True)
                local_i += 1

    # reduce stats across all processes
    sum_x = accelerator.reduce(sum_x, reduction="sum")
    sum_x2 = accelerator.reduce(sum_x2, reduction="sum")
    cnt = accelerator.reduce(cnt, reduction="sum")

    if accelerator.is_main_process:
        mean_value = (sum_x / cnt).item() if cnt.item() > 0 else float("nan")
        var_value = (sum_x2 / cnt - (sum_x / cnt) ** 2).item() if cnt.item() > 0 else float("nan")
        std_value = float(np.sqrt(max(var_value, 0.0))) if np.isfinite(var_value) else float("nan")

        accelerator.print(f"latent stat over {num_stat_samples} samples: mean is {mean_value} std scaling factor is {1 / std_value}")
        accelerator.print(f"std and mean saved into {path_to_latents}")



if __name__ == "__main__":
    # extract_latent(
    #     pretrained_weights='/leonardo_work/EUHPC_B29_014/LDM_exps/celeba256_SDVAE_bf16_b48_f16_flip_400k/SDVAE/checkpoint_330000/model.safetensors',
    #     config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml', batch_size=100, dataset='celeba256',
    #     path_to_dataset='/leonardo_work/EUHPC_B29_014/datasets/celeba256/celeba256',
    #     path_to_latents='/leonardo_work/EUHPC_B29_014/datasets/celeba256_latents/celeba256_SM_test',
    #     num_stat_samples=50000
    # )

    # run by cmd: accelerate launch extract_VAE_latents.py
    extract_latent_ddp(
        pretrained_weights='/leonardo_work/EUHPC_B29_014/LDM_exps/imagenet256_SDVAE_bf16_b128_f16_flip_400k/SDVAE/checkpoint_430000/model.safetensors',
        config_file='/leonardo_work/EUHPC_B29_014/LDM/configs/ldm_f16d16.yaml', batch_size=100, dataset='imagenet_train',
        path_to_dataset='/leonardo_work/EUHPC_B29_014/datasets/imagenet256/train',
        path_to_latents='/leonardo_work/EUHPC_B29_014/datasets/imagenet256_latents/imagenet256_SDVAE_f16_430k',
        num_stat_samples=50000
    )