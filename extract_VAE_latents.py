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


def get_latent_scaler(pretrained_weights, config_file, batch_size, dataset, path_to_dataset, num_stat_samples,):
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"compute latent stats for {dataset} from {pretrained_weights}")
    accelerator.print(f"accelerate: world_size={accelerator.num_processes}, rank={accelerator.process_index}")

    with open(config_file, "r") as f:
        vae_config = yaml.safe_load(f)
        config = LDMConfig(**vae_config["vae"])

    model = VAE(config)
    state_dict = load_file(pretrained_weights)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)

    org_dataset, _ = get_dataset(dataset=dataset, path_to_data=path_to_dataset, train=False, random_flip_p=0.0)
    loader = DataLoader(org_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=12, pin_memory=True, persistent_workers=True,)

    model, loader = accelerator.prepare(model, loader)
    base_model = accelerator.unwrap_model(model)

    if accelerator.is_main_process:
        accelerator.print(f"found {len(org_dataset)} samples in {dataset}")
    accelerator.wait_for_everyone()

    # split total requested samples across ranks
    world = max(accelerator.num_processes, 1)
    per_rank_target = int(math.ceil(num_stat_samples / world))

    stat_taken = 0
    sum_x = torch.tensor(0.0, device=device)
    sum_x2 = torch.tensor(0.0, device=device)
    cnt = torch.tensor(0.0, device=device)

    for batch in tqdm(loader, disable=not accelerator.is_local_main_process):
        if stat_taken >= per_rank_target:
            break

        with torch.no_grad():
            img = batch["images"].to(device, non_blocking=True)
            moments = base_model.forward_enc(img)  # (batch, 8, 32, 32)
            mu, logvar = torch.chunk(moments, 2, dim=1)  # (batch, 4, 32, 32) each
            logvar = torch.clamp(logvar, min=-30.0, max=20.0)
            sigma = torch.exp(0.5 * logvar)

            noise = torch.randn_like(sigma)
            latent = mu + sigma * noise

            need = per_rank_target - stat_taken
            take = min(need, latent.shape[0])

            x = latent[:take].float().reshape(-1)
            sum_x += x.sum()
            sum_x2 += (x * x).sum()
            cnt += x.numel()
            stat_taken += take

    # reduce across ranks
    sum_x = accelerator.reduce(sum_x, reduction="sum")
    sum_x2 = accelerator.reduce(sum_x2, reduction="sum")
    cnt = accelerator.reduce(cnt, reduction="sum")

    if accelerator.is_main_process:
        cnt_val = cnt.item()
        if cnt_val > 0:
            mean_value = (sum_x / cnt).item()
            var_value = (sum_x2 / cnt - (sum_x / cnt) ** 2).item()
            std_value = float(np.sqrt(max(var_value, 0.0)))
        else:
            mean_value, std_value = float("nan"), float("nan")

        accelerator.print(
            f"latent stat over ~{num_stat_samples} samples (actual elements={int(cnt_val)}): "
            f"mean={mean_value}, std={std_value}, std scaling factor={1.0/std_value if std_value and np.isfinite(std_value) else float('nan')}"
        )


def extract_latent_ddp_for_REPA(
    pretrained_weights,
    config_file,
    batch_size,
    dataset,
    path_to_dataset,
    path_to_latents,      # == args.dest in the first script (latent root)
    path_to_images,       # == args.dest_images in the first script (image root). "" => sibling "images"
    num_workers=12,
    no_images=False,
    max_save_threads=8,   # sane cap per-rank
):
    """
    DDP/Accelerate version that matches the behavior of your first torchrun script:

    For each batch:
      - moments = base_model.forward_enc(img)  (B, 8, H', W') == cat([mu, logvar], dim=1)
      - mu, logvar = chunk(moments, 2, dim=1); sigma = exp(0.5*clamp(logvar))
      - z = cat([mu, sigma], dim=1)  (B, 8, H', W')  # mean+std
      - save image as PNG (optional)
      - save z as .npy in dest/<prefix>/img-mean-std-<idx>.npy
      - record labels for dataset.json

    At end:
      - all_gather_object labels and labels_2
      - rank0 writes dest/dataset.json and dest_images/dataset.json
    """
    import json
    import math
    from itertools import chain
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import PIL.Image
    import torch
    from accelerate import Accelerator
    from torch.utils.data import ConcatDataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler

    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    world = accelerator.num_processes

    accelerator.print(f"extract image latents for {dataset} from {pretrained_weights}")
    accelerator.print(f"accelerate: world_size={world}, rank={rank}")

    with open(config_file, "r") as f:
        vae_config = yaml.safe_load(f)
        config = LDMConfig(**vae_config["vae"])

    model = VAE(config)
    state_dict = load_file(pretrained_weights)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)

    org_dataset, _ = get_dataset(dataset=dataset, path_to_data=path_to_dataset, train=False, random_flip_p=0.0)
    loader = DataLoader(
        org_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )

    model, loader = accelerator.prepare(model, loader)
    base_model = accelerator.unwrap_model(model)

    # make dir for npy latents and org_images (will be fed into dino in REPA)
    if accelerator.is_main_process:
        os.makedirs(path_to_latents, exist_ok=True)  # latents root
    accelerator.wait_for_everyone()

    dest_images = ""
    if not no_images:
        if path_to_images is None:
            path_to_images = ""

        if path_to_images == "":
            dest_images = os.path.join(os.path.dirname(path_to_latents), "images")
        else:
            dest_images = path_to_images

        if accelerator.is_main_process:
            os.makedirs(dest_images, exist_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.print(f"[Dataset] size: {len(org_dataset)} samples total (before sharding)")

    labels = []  # [latent_relpath, class_id]
    labels_2 = []  # [image_relpath, class_id]
    total_steps = 0  # Unique global indexing, idx = step * batch_size * world + i * world + rank
    max_workers = int(max(1, min(max_save_threads, (os.cpu_count() or 8))))
    executor = ThreadPoolExecutor(max_workers=max_workers)  # Saving performance: create executor once per rank + cache dirs
    made_dirs = set()

    if accelerator.is_main_process:
        accelerator.print("len(org_dataset) =", len(org_dataset))
        accelerator.print("len(loader) =", len(loader))

    try:
        for batch in tqdm(loader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                img = batch["images"].to(device, non_blocking=True)
                class_label = batch.get("class_conditioning", None)

                moments = base_model.forward_enc(img)  # moments:(B, 8, H', W') == cat([mu, logvar], dim=1)
                mu, logvar = torch.chunk(moments, chunks=2, dim=1)

                logvar = torch.clamp(logvar, min=-30.0, max=20.0)
                sigma = torch.exp(0.5 * logvar)  # std
                z = torch.cat([mu, sigma], dim=1)  # (B, 8, H', W') == mean+std

                # Prepare labels as a simple 1D numpy array of length B
                if class_label is None:
                    labels_np = np.zeros((z.shape[0],), dtype=np.int64)  # if dataset didn't provide it, default 0
                elif torch.is_tensor(class_label):
                    labels_np = class_label.detach().cpu().numpy()
                else:
                    labels_np = np.asarray(class_label)

                if labels_np.shape[0] != z.shape[0]:
                    raise RuntimeError(f"batch mismatch: z batch={z.shape[0]}, labels={labels_np.shape[0]}")

                # Convert samples for saving (CPU-side)
                # image conversion should be based on original img tensor (normalized to [-1,1] expected)
                if not no_images:
                    img_cpu = img.detach().float().cpu()  # range assumed [-1,1]
                    img_u8 = ((img_cpu + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)  # (B,C,H,W)

                z_cpu = z.detach().cpu().numpy()  # (B, 8, H', W')
                B = z_cpu.shape[0]
                for i in range(B):
                    idx = total_steps * batch_size * world + i * world + rank
                    idx_str = f"{idx:08d}"

                    # match first script paths
                    img_rel = f"{idx_str[:5]}/img{idx_str}.png"
                    lat_rel = f"{idx_str[:5]}/img-mean-std-{idx_str}.npy"

                    # record labels (relative paths)
                    labels.append([lat_rel, int(labels_np[i])])
                    labels_2.append([img_rel, int(labels_np[i])])

                    # mkdir once per dir per rank
                    lat_dir = os.path.join(path_to_latents, os.path.dirname(lat_rel))
                    if lat_dir not in made_dirs:
                        os.makedirs(lat_dir, exist_ok=True)
                        made_dirs.add(lat_dir)

                    if not no_images:
                        img_dir = os.path.join(dest_images, os.path.dirname(img_rel))
                        if img_dir not in made_dirs:
                            os.makedirs(img_dir, exist_ok=True)
                            made_dirs.add(img_dir)

                        # build PIL image
                        x_i = img_u8[i].permute(1, 2, 0).contiguous().numpy()  # HWC uint8
                        pil_i = PIL.Image.fromarray(x_i)

                        executor.submit(pil_i.save, os.path.join(dest_images, img_rel))

                    # save latent mean+std
                    executor.submit(np.save, os.path.join(path_to_latents, lat_rel), z_cpu[i])

            total_steps += 1

    finally:
        executor.shutdown(wait=True)

    # -------------------------
    # Gather labels and write dataset.json
    # -------------------------
    gather_labels = [None for _ in range(world)]
    gather_labels_2 = [None for _ in range(world)]
    accelerator.wait_for_everyone()

    # Use torch.distributed via accelerator
    torch.distributed.all_gather_object(gather_labels, labels)
    torch.distributed.all_gather_object(gather_labels_2, labels_2)

    if accelerator.is_main_process:
        all_labels = list(chain(*gather_labels))
        all_labels_2 = list(chain(*gather_labels_2))

        all_labels = sorted(all_labels, key=lambda x: x[0])
        with open(os.path.join(path_to_latents, "dataset.json"), "w") as f:
            json.dump({"labels": all_labels}, f)

        if not no_images:
            all_labels_2 = sorted(all_labels_2, key=lambda x: x[0])
            with open(os.path.join(dest_images, "dataset.json"), "w") as f:
                json.dump({"labels": all_labels_2}, f)

        accelerator.print(f"Saved: {os.path.join(path_to_latents, 'dataset.json')}")
        if not no_images:
            accelerator.print(f"Saved: {os.path.join(dest_images, 'dataset.json')}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    extract_latent(
        pretrained_weights='PATH_TO_CKPT/model.safetensors',
        config_file='/configs/ldm_f16d16.yaml', batch_size=100, dataset='celeba256',
        path_to_dataset='PATH_TO_DATASET',
        path_to_latents='PATH_TO_SAVE_LATENTS',
        num_stat_samples=50000
    )

    # produce latents for ImageNet 256x256
    # # accelerate launch --num_processes 4 --mixed_precision no extract_VAE_latents.py
    get_latent_scaler(
        pretrained_weights="PATH_TO_CKPT/model.safetensors",
        config_file="/configs/ldm_f16d16.yaml", batch_size=100, dataset="imagenet_train",
        path_to_dataset="'PATH_TO/imagenet256/train", num_stat_samples=200000,
    )

    # accelerate launch --num_processes 4 --mixed_precision no extract_VAE_latents.py
    extract_latent_ddp_for_REPA(
        pretrained_weights="PATH_TO_CKPT/model.safetensors",
        config_file="configs/ldm_f16d16.yaml", batch_size=100, dataset="imagenet_train",
        path_to_dataset="PATH_TO/imagenet256/train",
        path_to_latents="PATH_TO/imagenet256_DSM/vae-sd",
        path_to_images="PATH_TO/imagenet256_DSM/images",
        num_workers=12,
        no_images=False,  # set True to skip PNG writing
        max_save_threads=8,  # cap per-rank save threads (recommend 4–16)
    )