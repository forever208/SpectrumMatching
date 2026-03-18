import os
import yaml
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from tqdm import tqdm
from diffusers.optimization import get_scheduler
import lpips

from utils import load_val_images, save_orig_and_generated_images, count_num_params, convert_to_PIL_imgs
from modules import VAE, LDMConfig, PatchGAN, init_weights
from modules import LPIPS as mylpips
from dataset import get_dataset
from eval_utils.utils import calculate_psnr_between_folders
from eval_utils.fid_score import calculate_fid_given_paths
from torchmetrics import StructuralSimilarityIndexMeasure
import shutil
from utils_DCT import latent_spectral_reg_dct, split_into_blocks_torch, combine_blocks_torch, dct_2d_torch_unified, idct_2d_torch_unified


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
    parser.add_argument("--dataset", help="dataset to train on", choices=("conceptual_captions", "imagenet", "coco", "celeba256", "ffhq128", "ffhq256"), required=True, type=str)
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
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=120))
    accelerator = Accelerator(
        project_dir=path_to_experiment,
        gradient_accumulation_steps=train_cfg["gradient_accumulations_steps"],
        log_with="wandb" if args.log_wandb else None,
        kwargs_handlers = [pg_kwargs]
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

        # If training on multiple GPUs, we need to convert BatchNorm to SyncBatchNorm
        if accelerator.num_processes > 1:
            discriminator = nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    ### Print Out Number of Trainable Parameters ###
    accelerator.print(f"NUMBER OF VAE PARAMETERS: {count_num_params(model)}")
    accelerator.print("Mixed precision:", accelerator.mixed_precision)
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

    if args.dataset == 'imagenet':
        dataset, _ = get_dataset(
            dataset='imagenet_train',
            path_to_data=f'{args.path_to_dataset}/train',
            num_channels=vae_config["in_channels"],
            img_size=vae_config["img_size"],
            random_resize=train_cfg["random_resize"],  # default as False
            interpolation=train_cfg["interpolation"],
            random_flip_p=train_cfg["random_flip_p"]
        )
        accelerator.print("Number of Training Samples:", len(dataset))

        val_dataset, _ = get_dataset(
            dataset='imagenet_val',
            path_to_data=f'{args.path_to_dataset}/val',
            num_channels=vae_config["in_channels"],
            img_size=vae_config["img_size"],
            random_resize=False,  # default as False
            interpolation=train_cfg["interpolation"],
            random_flip_p=0.0
        )
        accelerator.print("Number of validation Samples:", len(val_dataset))

    else:
        dataset, _ = get_dataset(
            dataset=args.dataset,
            path_to_data=args.path_to_dataset,
            num_channels=vae_config["in_channels"],
            img_size=vae_config["img_size"],
            random_resize=train_cfg["random_resize"],  # default as False
            interpolation=train_cfg["interpolation"],
            random_flip_p=train_cfg["random_flip_p"]
        )
        accelerator.print("Number of Training Samples:", len(dataset))
        val_dataset = dataset

    dataloader = DataLoader(
        dataset,
        batch_size=mini_batchsize,
        pin_memory=train_cfg["pin_memory"],
        num_workers=train_cfg["num_workers"],
        shuffle=True,
        persistent_workers=True,
    )

    eval_dataloader = DataLoader(
        val_dataset,
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
    if use_disc:
        disc_lr_scheduler = get_scheduler(
            train_cfg["disc_lr_scheduler"],
            optimizer=disc_optimizer,
            num_training_steps=train_cfg["total_training_iterations"] * accelerator.num_processes,
            num_warmup_steps=train_cfg["disc_lr_warmup_steps"] * accelerator.num_processes,
        )

    ### Prepare Everything ###
    components = [model, optimizer, lr_scheduler, dataloader, eval_dataloader]
    if use_lpips:
        components += [lpips_loss_fn, ssim_fn]
    if use_disc:
        components += [discriminator, disc_optimizer, disc_lr_scheduler]

    prepared = accelerator.prepare(*components)  # Call prepare ONCE
    model = prepared[0]
    optimizer = prepared[1]
    lr_scheduler = prepared[2]
    dataloader = prepared[3]
    eval_dataloader = prepared[4]

    if use_lpips:
        lpips_loss_fn = prepared[5]
        ssim_fn = prepared[6]
    if use_disc:
        discriminator = prepared[7]
        disc_optimizer = prepared[8]
        disc_lr_scheduler = prepared[9]

    ### Load Validation Images (If we have a folder of them) ###
    val_images = None
    if train_cfg["val_img_folder_path"] is not None:
        val_images = load_val_images(
            path_to_image_folder=train_cfg["val_img_folder_path"],
            img_size=vae_config["img_size"],
            device=accelerator.device,
            dtype=accelerator.mixed_precision
        )

    ### Initialize log Variables ###
    model_log = {"loss": 0, "percept_loss": 0, "recon_loss": 0, "lpips_loss": 0, "kl_loss": 0, "sm_rgb": 0, "sm_delta": 0, "rmsc_loss": 0, "disc_loss": 0, "adp_weight": 0}
    disc_log = {"disc_loss": 0, "logits_real": 0, "logits_fake": 0}

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

    ### Training Loop ###
    progress_bar = tqdm(range(train_cfg["total_training_iterations"]), initial=global_step, disable=not accelerator.is_local_main_process)
    train = True
    eval_org_imgs_path = os.path.join(args.eval_dir, "eval_org_imgs")
    eval_recon_imgs_path = os.path.join(args.eval_dir, "eval_recon_imgs")
    eval_lpips = []
    eval_ssim = []
    eval_sm_rgb = []
    eval_sm_delta = []
    eval_rmsc = []

    for key, value in train_cfg.items():
        accelerator.print(f"{key}: {value}")

    while train:
        model.train()
        if use_disc:
            discriminator.train()

        for i, batch in enumerate(dataloader):
            DSM_mask = random.choice([0, 8, 10, 12,])  # DSM parameters
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

            model_outputs = model(pixel_values, DSM_mask, train_cfg["blk_sz"], delta=train_cfg["delta"])
            reconstructions = model_outputs["reconstruction"]
            pixel_values = model_outputs["img"]

            ### train the VAE ###
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

                    ### Discriminator Loss (incase we are training the discriminator) ###
                    gen_loss = torch.zeros(size=(), device=pixel_values.device)
                    adaptive_weight = torch.zeros(size=(), device=pixel_values.device)
                    if train_disc:
                        gen_loss = -1 * discriminator(reconstructions).mean()  # generator loss

                        # use gradient of the last layer to construct the adaptive weight
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

                    ### KL Loss ###
                    kl_loss = model_outputs["kl_loss"].mean()
                    loss = loss + kl_loss * train_cfg["kl_weight"]

                    ### ESM Loss ###
                    sm_loss = model_outputs["sm_delta"].mean()
                    loss = loss + sm_loss * train_cfg["esm_weight"]

                    ### RMSC Loss ###
                    rmsc_loss = model_outputs["rmsc_loss"]
                    loss = loss + rmsc_loss * train_cfg["rmsc_weight"]

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
                        "sm_rgb": model_outputs["sm_rgb"],
                        "sm_delta": model_outputs["sm_delta"],
                        "rmsc_loss": rmsc_loss,
                        "disc_loss": gen_loss,
                        "adp_weight": adaptive_weight
                    }
                    for key, value in log.items():
                        model_log[key] += value.mean() / train_cfg["gradient_accumulations_steps"]

            ### train the discriminator ###
            else:
                disc_optimizer.zero_grad()
                with accelerator.accumulate(discriminator):
                    real = discriminator(pixel_values)
                    fake = discriminator(reconstructions)
                    loss = (F.relu(1 + fake) + F.relu(1 - real)).mean()  # discriminator Hinge loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(discriminator.parameters(), 1.0)
                    disc_optimizer.step()
                    disc_lr_scheduler.step()

                    log = {"disc_loss": loss, "logits_real": real.mean(), "logits_fake": fake.mean()}
                    for key, value in log.items():
                        disc_log[key] += value.mean() / train_cfg["gradient_accumulations_steps"]

            ### output log info ###
            if accelerator.sync_gradients:
                if model_toggle or not train_disc:  # If we updated the VAE
                    model_log = {key: accelerator.gather_for_metrics(value).mean().item() for key, value in model_log.items()}
                    model_log["lr"] = lr_scheduler.get_last_lr()[0]
                    logging_string = "GEN: "
                    for k, v in model_log.items():
                        v = v.item() if torch.is_tensor(v) else v
                        if "lr" in k:
                            v = f"{v:.1e}"
                        else:
                            v = round(v, 4)
                        logging_string += f"|{k}: {v}"

                    if global_step % 100 == 0:
                        accelerator.print(f"step {global_step}: {logging_string}")  # Print to Console
                        accelerator.log(model_log, step=global_step)  # Push to WandB
                    model_log = reset_log(model_log)  # Reset Log for Next Accumulation
                    model_log.pop("lr")

                else:  # If we updated the Discriminator
                    disc_log = {key: accelerator.gather_for_metrics(value).mean().item() for key, value in disc_log.items()}
                    disc_log["disc_lr"] = disc_lr_scheduler.get_last_lr()[0]
                    logging_string = "DIS: "
                    for k, v in disc_log.items():
                        v = v.item() if torch.is_tensor(v) else v
                        if "lr" in k:
                            v = f"{v:.1e}"
                        else:
                            v = round(v, 4)
                        logging_string += f"|{k}: {v}"

                    if global_step % 100 == 1:
                        accelerator.print(f"step {global_step}: {logging_string}")  # Print to Console
                        accelerator.log(disc_log, step=global_step)  # Push to WandB
                    disc_log = reset_log(disc_log)  # Reset Log for Next Accumulation
                    disc_log.pop("disc_lr")

                global_step += 1
                progress_bar.update(1)

            ### save ckpt ###
            if (global_step % train_cfg["checkpoint_iterations"] == 0) or (
                    global_step == train_cfg["total_training_iterations"] - 1):
                path_to_checkpoint = os.path.join(path_to_experiment, f"checkpoint_{global_step}")
                accelerator.save_state(output_dir=path_to_checkpoint)

            ### Validation step ###
            if global_step % train_cfg["val_generation_freq"] == 0:
                if accelerator.is_main_process:
                    os.makedirs(eval_org_imgs_path, exist_ok=True)
                    os.makedirs(eval_recon_imgs_path, exist_ok=True)

                mini_batch_size = train_cfg["per_gpu_batch_size"]
                batch_size = mini_batch_size * accelerator.num_processes
                num_iterations = train_cfg["num_eval_images"] // batch_size + 1
                world_size = accelerator.state.num_processes
                global_rank = accelerator.process_index

                ### load eval images and save images for evaluation ###
                model.eval()
                accelerator.print(f"using {world_size} GPUs, global batch size {batch_size} ")
                accelerator.print(f"staring evaluation using {train_cfg['num_eval_images']} images...")
                eval_iter = iter(eval_dataloader)
                for j in tqdm(range(num_iterations), disable=not accelerator.is_main_process, desc='sample2dir'):
                    try:
                        mini_batch = next(eval_iter)
                    except StopIteration:
                        eval_iter = iter(eval_dataloader)  # Restart the iterator if we reach the end
                        mini_batch = next(eval_iter)

                    org_imgs = mini_batch["images"].to(accelerator.device)
                    with torch.no_grad():
                        outputs = model(org_imgs, delta=train_cfg["delta"])
                        recon_imgs = outputs["reconstruction"]
                        eval_lpips.append(lpips_loss_fn(recon_imgs, org_imgs).mean())
                        eval_ssim.append(ssim_fn(recon_imgs, org_imgs))
                        eval_sm_rgb.append(outputs["sm_rgb"])
                        eval_sm_delta.append(outputs["sm_delta"])
                        eval_rmsc.append(outputs["rmsc_loss"])

                    org_imgs = convert_to_PIL_imgs(org_imgs)  # a list PIL images
                    recon_imgs = convert_to_PIL_imgs(recon_imgs)  # a list PIL images

                    for b_id in range(mini_batch_size):  # distributed image save
                        img_id = j * mini_batch_size * world_size + global_rank * mini_batch_size + b_id
                        if img_id >= train_cfg["num_eval_images"]:
                            break
                        org_imgs[b_id].save(os.path.join(eval_org_imgs_path, f"{img_id}.jpg"))
                        recon_imgs[b_id].save(os.path.join(eval_recon_imgs_path, f"{img_id}.jpg"))

                # do visualization and metrics computation
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    accelerator.print(f'{len(os.listdir(eval_org_imgs_path))} images in {eval_org_imgs_path}')
                    accelerator.print(f'{len(os.listdir(eval_recon_imgs_path))} images in {eval_recon_imgs_path}')
                    assert len(os.listdir(eval_recon_imgs_path)) == train_cfg["num_eval_images"]

                    ### save images for visualization ###
                    if val_images is None:
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

                    ### evaluate rFID ###
                    accelerator.print(f"Evaluating rFID...")
                    fid = calculate_fid_given_paths([eval_org_imgs_path, eval_recon_imgs_path], device=pixel_values.device)

                    accelerator.print(f"Evaluating PSNR...")
                    psnr_values = calculate_psnr_between_folders(eval_org_imgs_path, eval_recon_imgs_path)
                    avg_psnr = sum(psnr_values) / len(psnr_values)

                    accelerator.print(f"Evaluating LPIPS...")
                    eval_lpips = torch.tensor(eval_lpips)
                    eval_lpips = eval_lpips.mean().item()

                    accelerator.print(f"Evaluating SSIM...")
                    eval_ssim = torch.tensor(eval_ssim)
                    eval_ssim = eval_ssim.mean().item()

                    accelerator.print(f"Evaluating sm_rgb...")
                    eval_sm_rgb = torch.tensor(eval_sm_rgb)
                    eval_sm_rgb = eval_sm_rgb.mean().item()

                    accelerator.print(f"Evaluating sm_delta...")
                    eval_sm_delta = torch.tensor(eval_sm_delta)
                    eval_sm_delta = eval_sm_delta.mean().item()

                    accelerator.print(f"Evaluating RMSC...")
                    eval_rmsc = torch.tensor(eval_rmsc)
                    eval_rmsc = eval_rmsc.mean().item()

                    with open(os.path.join(args.working_directory, f'eval.log'), 'a') as f:
                        print(f'step={global_step} rFID={fid:.5f} PSNR={avg_psnr:.5f} LPIPS={eval_lpips:.5f} SSIM={eval_ssim:.5f} sm_rgb={eval_sm_rgb:.5f}, sm_delta={eval_sm_delta:.5f}, RMSC={eval_rmsc:.5f}', file=f)

                    shutil.rmtree(eval_org_imgs_path)  # remove the image folder
                    shutil.rmtree(eval_recon_imgs_path)  # remove the image folder
                    eval_lpips = []
                    eval_ssim = []
                    eval_sm_rgb = []
                    eval_sm_delta = []
                    eval_rmsc = []
                    model.eval()

                torch.cuda.empty_cache()
                accelerator.wait_for_everyone()

                # # evaluate low-pass FID
                # num_imgs = train_cfg["num_eval_images"] // 5  # to save time
                # batch_size = mini_batch_size * accelerator.num_processes
                # num_iterations = num_imgs // batch_size + 1  # FID-10k
                # lpFID = []
                #
                # with torch.no_grad():
                #     for k in [0, 2, 4, 6, 8]:
                #         if accelerator.is_main_process:
                #             os.makedirs(eval_org_imgs_path, exist_ok=True)
                #             os.makedirs(eval_recon_imgs_path, exist_ok=True)
                #
                #         for j in tqdm(range(num_iterations), disable=not accelerator.is_main_process, desc='sample2dir'):
                #             try:
                #                 mini_batch = next(eval_iter)
                #             except StopIteration:
                #                 eval_iter = iter(eval_dataloader)  # Restart the iterator if we reach the end
                #                 mini_batch = next(eval_iter)
                #
                #             img = mini_batch["images"].to(accelerator.device)
                #             model_outputs = model(img, k, train_cfg["blk_sz"])
                #             recon_lowpass_img = model_outputs["reconstruction"]
                #             img = model_outputs["img"]
                #
                #             img = convert_to_PIL_imgs(img)  # a list PIL images
                #             recon_lowpass_img = convert_to_PIL_imgs(recon_lowpass_img)  # a list PIL images
                #
                #             for b_id in range(mini_batch_size):  # distributed image save
                #                 img_id = j * mini_batch_size * world_size + global_rank * mini_batch_size + b_id
                #                 if img_id >= num_imgs:
                #                     break
                #                 img[b_id].save(os.path.join(eval_org_imgs_path, f"{img_id}.jpg"))
                #                 recon_lowpass_img[b_id].save(os.path.join(eval_recon_imgs_path, f"{img_id}.jpg"))
                #
                #         accelerator.wait_for_everyone()
                #         if accelerator.is_main_process:
                #             accelerator.print(f"Evaluating low_pass FID{k}...")
                #             assert len(os.listdir(eval_org_imgs_path)) == num_imgs
                #             assert len(os.listdir(eval_recon_imgs_path)) == num_imgs
                #
                #             fid = calculate_fid_given_paths([eval_org_imgs_path, eval_recon_imgs_path], device=accelerator.device)
                #             lpFID.append(round(fid, 4))
                #
                #             shutil.rmtree(eval_org_imgs_path)  # remove the image folder
                #             shutil.rmtree(eval_recon_imgs_path)  # remove the image folder
                #
                #         torch.cuda.empty_cache()
                #         accelerator.wait_for_everyone()
                #
                # if accelerator.is_main_process:
                #     with open(os.path.join(args.working_directory, f'eval.log'), 'a') as f:
                #         print(f'step={global_step} low_pass_FID={lpFID}', file=f)

                accelerator.wait_for_everyone()
                model.train()

            if global_step >= train_cfg["total_training_iterations"]:
                print("Completed Training")
                train = False
                break


if __name__ == '__main__':
    main()
