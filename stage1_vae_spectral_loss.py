import os
import yaml
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
from diffusers.optimization import get_scheduler
import lpips

from utils import load_val_images, save_orig_and_generated_images, count_num_params, convert_to_PIL_imgs
from modules import VAE, LDMConfig, PatchGAN, init_weights
from modules import LPIPS as mylpips
from dataset import get_dataset
from torchmetrics import StructuralSimilarityIndexMeasure
import shutil
import torch_dct as dct


def split_into_blocks_torch(image: torch.Tensor, block_sz: int):
    """
    Split a 2D tensor (H, W) or batched 3D tensor (B, H, W) into non-overlapping (block_sz x block_sz) blocks.

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


def dct_2d_torch(x):
    # x: (N, B, B) or (B, B) float32 tensor
    # x should be in the range of [0, 255]

    # Apply 2D DCT Type-II
    x = x.float() - 128.0  # Ensure float32 and subtract 128 (OpenCV style)
    x = dct.dct(x, norm='ortho')                 # DCT along last dimension
    x = dct.dct(x.transpose(-2, -1), norm='ortho')  # DCT along second-last
    return x.transpose(-2, -1)  # (N, B, B) or (B, B)



### Load Arguments ###
def experiment_config_parser():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument("--experiment_name", required=True, type=str, metavar="experiment_name")
    parser.add_argument("--working_directory", help="where checkpoints and logs are stored", required=True, type=str, metavar="working_directory")
    parser.add_argument("--eval_dir", help="where the eval images should be saved into", required=True, type=str, metavar="eval_dir")
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction, help="log to WandB?")
    parser.add_argument("--wandb_run_name", required=True, type=str, metavar="wandb_run_name")
    parser.add_argument("--resume_from_checkpoint",  help="name of ckpt folder to resume training from", default=None, type=str, metavar="resume_from_checkpoint")
    parser.add_argument("--training_config", help="Path to config file", required=True, type=str, metavar="training_config")
    parser.add_argument("--model_config", help="Path to model config file", required=True, type=str, metavar="model_config")
    parser.add_argument("--dataset", help="dataset to train on", choices=("conceptual_captions", "imagenet", "coco", "celeba", "celeba256", "birds", "ffhq128"), required=True, type=str)
    parser.add_argument("--path_to_dataset", help="Root directory of dataset", required=True, type=str)
    parser.add_argument("--block_sz", help="block size of DCT", required=True, type=int)
    args = parser.parse_args()

    return args


def main():
    args = experiment_config_parser()

    ### Load Configs (training config and vae config) ###
    with open(args.training_config, "r") as f:
        train_cfg = yaml.safe_load(f)["training_args"]

    with open(args.model_config, "r") as f:
        vae_config = yaml.safe_load(f)["vae"]
        config = LDMConfig(**vae_config)

    assert not config.quantize, "This script only supports VAE, use stage1_vqvae_trainer.py for Quantized"

    ### Initialize Accelerator/Tracker ###
    path_to_experiment = os.path.join(args.working_directory, args.experiment_name)
    accelerator = Accelerator(
        project_dir=path_to_experiment,
        gradient_accumulation_steps=train_cfg["gradient_accumulations_steps"],
        log_with="wandb" if args.log_wandb else None
    )

    if args.log_wandb:  # init wandb with accelerator
        accelerator.init_trackers(args.experiment_name, init_kwargs={"wandb": {"name": args.wandb_run_name}})

    ### Load Model ###
    model = VAE(config).to(accelerator.device)
    latent_res = (config.img_size // (2**(len(config.vae_channels_per_block)-1)))
    accelerator.print(f"LATENT SPACE DIMENSIONS: {config.latent_channels, latent_res, latent_res}")

    ### Load LPIPS and SSIM ###
    use_lpips = False
    if train_cfg["use_lpips"]:
        use_lpips = True
        if train_cfg["use_lpips_package"]:
            lpips_loss_fn = lpips.LPIPS(net="vgg").eval()
        else:
            lpips_loss_fn = mylpips()
            lpips_loss_fn.load_checkpoint(train_cfg["lpips_checkpoint"])
        lpips_loss_fn = lpips_loss_fn.to(accelerator.device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0)).to(accelerator.device)

    ### Print Out Number of Trainable Parameters ###
    accelerator.print(f"NUMBER OF VAE PARAMETERS: {count_num_params(model)}")
    accelerator.print("Mixed precision:", accelerator.mixed_precision)

    ### Load Optimizers ###
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["optimizer_beta1"], train_cfg["optimizer_beta2"]),
        weight_decay=train_cfg["optimizer_weight_decay"]
    )

    ### Get DataLoader ###
    mini_batchsize = train_cfg["per_gpu_batch_size"] // train_cfg["gradient_accumulations_steps"]
    dataset, _ = get_dataset(
        dataset=args.dataset,
        path_to_data=args.path_to_dataset,
        num_channels=vae_config["in_channels"],
        img_size=vae_config["img_size"],
        random_resize=False,  # default as False
        interpolation=train_cfg["interpolation"],
        random_flip_p=0.0
    )
    accelerator.print("Number of Training Samples:", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=mini_batchsize,
        pin_memory=train_cfg["pin_memory"],
        num_workers=train_cfg["num_workers"],
        shuffle=False,
        persistent_workers=True,
        drop_last=False,
    )

    eval_dataloader = DataLoader(
        dataset,
        batch_size=mini_batchsize,
        pin_memory=False,
        num_workers=4,
        shuffle=False,
    )

    effective_epochs = (train_cfg["per_gpu_batch_size"] * accelerator.num_processes * train_cfg["total_training_iterations"]) / len(dataset)
    accelerator.print("Effective Epochs:", round(effective_epochs, 2))

    ### Learning Rate Scheduler ###
    lr_scheduler = get_scheduler(
            train_cfg["lr_scheduler"],
            optimizer=optimizer,
            num_training_steps=train_cfg["total_training_iterations"] * accelerator.num_processes,
            num_warmup_steps=train_cfg["lr_warmup_steps"] * accelerator.num_processes
        )

    ### Prepare Everything ###
    components = [model, optimizer, lr_scheduler, dataloader, eval_dataloader]
    if use_lpips:
        components += [lpips_loss_fn, ssim_fn]

    prepared = accelerator.prepare(*components)  # Call prepare ONCE
    model = prepared[0]
    optimizer = prepared[1]
    lr_scheduler = prepared[2]
    dataloader = prepared[3]
    eval_dataloader = prepared[4]

    if use_lpips:
        lpips_loss_fn = prepared[5]
        ssim_fn = prepared[6]

    ### Resume From Checkpoint ###
    if args.resume_from_checkpoint is not None:
        accelerator.print(f"Resuming from Checkpoint: {args.resume_from_checkpoint}")
        path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
        accelerator.load_state(path_to_checkpoint)
        global_step = int(args.resume_from_checkpoint.split("_")[-1])
    else:
        global_step = 0

    ### Training Loop ###
    for key, value in train_cfg.items():
        accelerator.print(f"{key}: {value}")

    accelerator.print(f"start loss analysing...")
    coe_losses = []
    input_dcts = []
    recon_dcts = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, disable=not accelerator.is_local_main_process)):
            pixel_values = batch["images"].to(accelerator.device)  # (batch, 3, H, W), value range [-1, 1]
            model_outputs = model(pixel_values)
            reconstructions = model_outputs["reconstruction"]  # (batch, 3, H, W)

            # scale to range [0, 255] before applying DCT
            pixel_values = (pixel_values + 1.0) * 127.5
            reconstructions = (reconstructions + 1.0) * 127.5

            # reconstruction spectral loss
            input_dct = split_into_blocks_torch(pixel_values, args.block_sz)  # (batch, C, N_blocks, B, B)
            recon_dct = split_into_blocks_torch(reconstructions, args.block_sz)  # (batch, C, N_blocks, B, B)

            batch_sz, C, N, B, _ = recon_dct.shape
            input_dct = input_dct.reshape(batch_sz * C * N, B, B)  # (batch*C*N, B, B)
            recon_dct = recon_dct.reshape(batch_sz * C * N, B, B)  # (batch*C*N, B, B)

            input_dct = dct_2d_torch(input_dct)  # (batch*C*N, B, B)
            recon_dct = dct_2d_torch(recon_dct)  # (batch*C*N, B, B)
            coe_loss = torch.abs(recon_dct - input_dct).mean(dim=0)  # shape: (B, B)

            coe_losses.append(coe_loss.detach().cpu())
            input_dcts.append(torch.abs(input_dct).mean(dim=0).detach())  # observe the magnitude of input
            recon_dcts.append(torch.abs(recon_dct).mean(dim=0).detach())  # observe the magnitude of recon

        # stack local, move to device, gather all GPUs by reduce sum and count
        coe_losses = torch.stack(coe_losses, dim=0).to(accelerator.device)  # (num_mini_batches, B, B)
        input_dcts = torch.stack(input_dcts, dim=0).to(accelerator.device)  # (num_mini_batches, B, B)
        recon_dcts = torch.stack(recon_dcts, dim=0).to(accelerator.device)  # (num_mini_batches, B, B)

        local_coe_loss = coe_losses.mean(dim=0)  # (B, B)
        local_input_dcts = input_dcts.mean(dim=0)  # (B, B)
        local_recon_dcts = recon_dcts.mean(dim=0)  # (B, B)
        local_count = torch.tensor(coe_losses.shape[0], device=accelerator.device, dtype=local_coe_loss.dtype)
        accelerator.print(f"num of mini_batches: {int(local_count.item())}")

        # Gather per-rank tensors to main (all tensors need to be on GPU before gathering)
        global_coe_loss = accelerator.gather(local_coe_loss.unsqueeze(0))  # (world_sz, B, B)
        global_input_dcts = accelerator.gather(local_input_dcts.unsqueeze(0))  # (world_sz, B, B)
        global_recon_dcts = accelerator.gather(local_recon_dcts.unsqueeze(0))  # (world_sz, B, B)

        global_coe_loss = global_coe_loss.detach().cpu()
        global_input_dcts = global_input_dcts.detach().cpu()
        global_recon_dcts = global_recon_dcts.detach().cpu()

        accelerator.print(f"{global_coe_loss.shape}")
        accelerator.print(f"{global_input_dcts.shape}")
        accelerator.print(f"{global_recon_dcts.shape}")

        if accelerator.is_main_process:
            global_coe_loss = global_coe_loss.mean(dim=0)  # (B, B)
            global_input_dcts = global_input_dcts.mean(dim=0)  # (B, B)
            global_recon_dcts = global_recon_dcts.mean(dim=0)  # (B, B)

            accelerator.print(f"avg coe loss is {global_coe_loss}")
            accelerator.print(f"avg input DCT is {global_input_dcts}")
            accelerator.print(f"avg recon DCT is {global_recon_dcts}")



if __name__ == '__main__':
    main()
