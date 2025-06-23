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

from utils import load_val_images, save_orig_and_generated_images, count_num_params
from modules import VAE, LDMConfig, PatchGAN, init_weights
from modules import LPIPS as mylpips
from dataset import get_dataset


### Load Arguments ###
def experiment_config_parser():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument("--experiment_name", required=True, type=str, metavar="experiment_name")
    parser.add_argument("--working_directory", help="where checkpoints and logs are stored", required=True, type=str, metavar="working_directory")
    parser.add_argument("--log_wandb", action=argparse.BooleanOptionalAction, help="log to WandB?")
    parser.add_argument("--wandb_run_name", required=True, type=str, metavar="wandb_run_name")
    parser.add_argument("--resume_from_checkpoint",  help="name of ckpt folder to resume training from", default=None, type=str, metavar="resume_from_checkpoint")
    parser.add_argument("--training_config", help="Path to config file", required=True, type=str, metavar="training_config")
    parser.add_argument("--model_config", help="Path to model config file", required=True, type=str, metavar="model_config")
    parser.add_argument("--dataset", help="dataset to train on", choices=("conceptual_captions", "imagenet", "coco", "celeba", "celebahq", "birds", "ffhq"), required=True, type=str)
    parser.add_argument("--path_to_dataset", help="Root directory of dataset", required=True, type=str)
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
    latent_res = (config.img_size // (len(config.vae_channels_per_block)-1)**2)
    accelerator.print(f"LATENT SPACE DIMENSIONS: {config.latent_channels, latent_res, latent_res}")

    ### Load LPIPS ###
    use_lpips = False
    if train_cfg["use_lpips"]:
        use_lpips = True
        if train_cfg["use_lpips_package"]:
            lpips_loss_fn = lpips.LPIPS(net="vgg").eval()
        else:
            lpips_loss_fn = mylpips()
            lpips_loss_fn.load_checkpoint(train_cfg["lpips_checkpoint"])
        lpips_loss_fn = lpips_loss_fn.to(accelerator.device)

    ### Load Discriminator ###
    use_disc = False
    if train_cfg["use_patchgan"]:
        use_disc = True
        discriminator = PatchGAN(
            input_channels=vae_config["in_channels"],
            start_dim=train_cfg["disc_start_dim"],
            depth=train_cfg["disc_depth"],
            kernel_size=train_cfg["disc_kernel_size"],
            leaky_relu_slope=train_cfg["disc_leaky_relu"]
        ).apply(init_weights)
        discriminator = discriminator.to(accelerator.device)

        ### If training on multiple GPUs, we need to convert BatchNorm to SyncBatchNorm ###
        if accelerator.num_processes > 1:
            discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    ### Print Out Number of Trainable Parameters ###
    accelerator.print(f"NUMBER OF VAE PARAMETERS: {count_num_params(model)}")
    if use_disc:
        accelerator.print(f"NUMBER OF DISC PARAMETERS: {count_num_params(discriminator)}")

    ### Load Optimizers ###
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        betas=(train_cfg["optimizer_beta1"], train_cfg["optimizer_beta2"]),
        weight_decay=train_cfg["optimizer_weight_decay"]
    )
    if use_disc:
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=train_cfg["disc_learning_rate"],
            betas=(train_cfg["optimizer_beta1"], train_cfg["optimizer_beta2"]),
            weight_decay=train_cfg["optimizer_weight_decay"])

    ### Get DataLoader ###
    mini_batchsize = train_cfg["per_gpu_batch_size"] // train_cfg["gradient_accumulations_steps"]
    dataset, _ = get_dataset(
        dataset=args.dataset,
        path_to_data=args.path_to_dataset,
        num_channels=vae_config["in_channels"],
        img_size=vae_config["img_size"],
        random_resize=train_cfg["random_resize"],  # default as False
        interpolation=train_cfg["interpolation"],
    )
    accelerator.print("Number of Training Samples:", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=mini_batchsize,
        pin_memory=train_cfg["pin_memory"],
        num_workers=train_cfg["num_workers"],
        shuffle=True,
        persistent_workers=True,
    )

    effective_epochs = (train_cfg["per_gpu_batch_size"] * accelerator.num_processes * train_cfg["total_training_iterations"]) / len(dataset)
    accelerator.print("Effective Epochs:", round(effective_epochs, 2))

    ### Get Learning Rate Scheduler ###
    lr_scheduler = get_scheduler(
            train_cfg["lr_scheduler"],
            optimizer=optimizer,
            num_training_steps=train_cfg["total_training_iterations"],
            num_warmup_steps=train_cfg["lr_warmup_steps"]
        )
    if use_disc:
        disc_lr_scheduler = get_scheduler(
                train_cfg["disc_lr_scheduler"],
                optimizer=disc_optimizer,
                num_training_steps=train_cfg["total_training_iterations"],
                num_warmup_steps=train_cfg["disc_lr_warmup_steps"],
            )

    ### Prepare Everything ###
    model, optimizer, lr_scheduler, dataloader = accelerator.prepare(model, optimizer, lr_scheduler, dataloader)
    if use_disc:
        discriminator, disc_optimizer, disc_lr_scheduler = accelerator.prepare(discriminator, disc_optimizer, disc_lr_scheduler)
    if use_lpips:
        lpips_loss_fn = accelerator.prepare(lpips_loss_fn)

    ### Load Validation Images (If we have a folder of them) ###
    val_images = None
    if train_cfg["val_img_folder_path"] is not None:
        val_images = load_val_images(
            path_to_image_folder=train_cfg["val_img_folder_path"],
            img_size=vae_config["img_size"],
            device=accelerator.device,
            dtype=accelerator.mixed_precision
        )

    ### Initialize Variables to Accumulate ###
    model_log = {"loss": 0, "percept_loss": 0, "recon_loss": 0, "lpips_loss": 0,
                 "kl_loss": 0, "disc_loss": 0, "adp_weight": 0}
    disc_log = {"disc_loss": 0, "logits_real": 0, "logits_fake": 0}

    ### Quick Helper to Rest Logs ###
    def reset_log(log):
        return {key: 0 for (key, _) in log.items()}

    ### Resume From Checkpoint ###
    if args.resume_from_checkpoint is not None:
        accelerator.print(f"Resuming from Checkpoint: {args.resume_from_checkpoint}")
        path_to_checkpoint = os.path.join(path_to_experiment, args.resume_from_checkpoint)
        accelerator.load_state(path_to_checkpoint)
        global_step = int(args.resume_from_checkpoint.split("_")[-1])
    else:
        global_step = 0

    progress_bar = tqdm(range(train_cfg["total_training_iterations"]), initial=global_step, disable=not accelerator.is_local_main_process)

    ### Training Loop ###
    train = True
    while train:
        model.train()
        if use_disc:
            discriminator.train()

        for i, batch in enumerate(dataloader):
            pixel_values = batch["images"].to(accelerator.device)
            model_toggle = (global_step % 2) == 0
            train_disc = (global_step >= train_cfg["disc_start"])

            ### If not using discriminator, always do generator step, and train_disc is false ###
            if not use_disc:
                generator_step = True
                train_disc = False
            else:
                if model_toggle or not train_disc:
                    generator_step = True
                else:
                    generator_step = False

            ### Pass Through Model ###
            model_outputs = model(pixel_values)
            reconstructions = model_outputs["reconstruction"]

            if generator_step:
                optimizer.zero_grad()

                with accelerator.accumulate(model):
                    ### Reconstruction Loss ###
                    if train_cfg["reconstruction_loss_fn"] == "l1":
                        reconstruction_loss = F.l1_loss(pixel_values, reconstructions)
                    elif train_cfg["reconstruction_loss_fn"] == "l2":
                        reconstruction_loss = F.mse_loss(pixel_values, reconstructions)
                    else:
                        raise ValueError(f"{train_cfg['reconstruction_loss_fn']} is not a Valid Reconstruction Loss")

                    ### LPIPS Loss ###
                    lpips_loss = torch.zeros(size=(), device=pixel_values.device)
                    if use_lpips:
                        lpips_loss = lpips_loss_fn(reconstructions, pixel_values).mean()

                    ### Perceptual_loss ###
                    perceptual_loss = reconstruction_loss + train_cfg["lpips_weight"] * lpips_loss
                    loss = perceptual_loss

                    ### Compute Discriminator Loss (incase we are training the discriminator) ###
                    gen_loss = torch.zeros(size=(), device=pixel_values.device)
                    adaptive_weight = torch.zeros(size=(), device=pixel_values.device)
                    if train_disc:
                        gen_loss = -1 * discriminator(reconstructions).mean()  # generator loss

                        ### use gradient of the last layer to construct the adaptive weight
                        last_layer = accelerator.unwrap_model(model).decoder.conv_out.weight
                        norm_grad_wrt_perceptual_loss = torch.autograd.grad(
                            outputs=loss,
                            inputs=last_layer,
                            retain_graph=True
                        )[0].detach().norm(p=2)
                        norm_grad_wrt_gen_loss = torch.autograd.grad(
                            outputs=gen_loss,
                            inputs=last_layer,
                            retain_graph=True
                        )[0].detach().norm(p=2)

                        adaptive_weight = norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clamp(min=1e-8)
                        adaptive_weight = adaptive_weight.clamp(max=1e4)
                        loss = loss + adaptive_weight * gen_loss * train_cfg["disc_weight"]

                    ### Compute KL Loss ###
                    kl_loss = model_outputs["kl_loss"].mean()
                    loss = loss + kl_loss * train_cfg["kl_weight"]

                    ### Update Model ###
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()

                    ### Create Log of Everything ###
                    log = {
                        "loss": loss,
                        "percept_loss": perceptual_loss,
                        "recon_loss": reconstruction_loss,
                        "lpips_loss": lpips_loss,
                        "kl_loss": kl_loss,
                        "disc_loss": gen_loss,
                        "adp_weight": adaptive_weight
                    }
                    for key, value in log.items():
                        model_log[key] += value.mean() / train_cfg["gradient_accumulations_steps"]

            else:
                disc_optimizer.zero_grad()
                with accelerator.accumulate(discriminator):
                    real = discriminator(pixel_values)
                    fake = discriminator(reconstructions)
                    loss = (F.relu(1 + fake) + F.relu(1 - real)).mean()  # discriminator Hinge loss

                    ### Update Discriminator Model ###
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(discriminator.parameters(), 1.0)
                    disc_optimizer.step()
                    disc_lr_scheduler.step()

                    log = {"disc_loss": loss, "logits_real": real.mean(), "logits_fake": fake.mean()}
                    for key, value in log.items():
                        disc_log[key] += value.mean() / train_cfg["gradient_accumulations_steps"]

            if accelerator.sync_gradients:
                if model_toggle or not train_disc:  # If we updated the VAE

                    ## Gather Across GPUs ###
                    model_log = {key: accelerator.gather_for_metrics(value).mean().item() for key, value in model_log.items()}
                    model_log["lr"] = lr_scheduler.get_last_lr()[0]

                    logging_string = "GEN: "
                    for k, v in model_log.items():
                        v = v.item() if torch.is_tensor(v) else v
                        if "lr" in k:
                            v = f"{v:.1e}"
                        else:
                            v = round(v, 2)
                        logging_string += f"|{k}: {v}"

                    accelerator.print(logging_string)  # Print to Console
                    accelerator.log(model_log, step=global_step)  # Push to WandB

                    ### Reset Log for Next Accumulation ###
                    model_log = reset_log(model_log)
                    model_log.pop("lr")

                ### If we updated the Discriminator ###
                else:
                    ## Gather Across GPUs ###
                    disc_log = {key: accelerator.gather_for_metrics(value).mean().item() for key, value in disc_log.items()}
                    disc_log["disc_lr"] = disc_lr_scheduler.get_last_lr()[0]

                    logging_string = "DIS: "
                    for k, v in disc_log.items():
                        v = v.item() if torch.is_tensor(v) else v
                        if "lr" in k:
                            v = f"{v:.1e}"
                        else:
                            v = round(v, 2)
                        logging_string += f"|{k}: {v}"

                    ### Print to Console ###
                    accelerator.print(logging_string)

                    ### Push to WandB ###
                    accelerator.log(disc_log, step=global_step)

                    ### Reset Log for Next Accumulation ###
                    disc_log = reset_log(disc_log)
                    disc_log.pop("disc_lr")

                global_step += 1
                progress_bar.update(1)

            ### Validation step ###
            if global_step % train_cfg["val_generation_freq"] == 0:
                if accelerator.is_main_process:
                    if val_images is None:
                        ### If we dont have a val images folder, just use the last batch as our validation images ###
                        batch_size = len(pixel_values)
                        num_random_gens = train_cfg["num_val_random_samples"]
                        if batch_size < num_random_gens:
                            num_random_gens = batch_size
                        images_to_plot = pixel_values[:num_random_gens]
                    else:
                        images_to_plot = val_images

                    model.eval()
                    with torch.no_grad():
                        reconstructions = model(images_to_plot)["reconstruction"]

                    save_orig_and_generated_images(
                        original_images=images_to_plot,
                        generated_image_tensors=reconstructions.detach(),
                        path_to_save_folder=args.working_directory,
                        step=global_step,
                        accelerator=accelerator
                    )

                    model.train()
                accelerator.wait_for_everyone()

            ### save ckpt ###
            if (global_step % train_cfg["checkpoint_iterations"] == 0) or (global_step == train_cfg["total_training_iterations"]-1):
                path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{global_step}")
                accelerator.save_state(output_dir=path_to_checkpoint)

            if global_step >= train_cfg["total_training_iterations"]:
                print("Completed Training")
                train = False
                break


if __name__ == '__main__':
    main()
