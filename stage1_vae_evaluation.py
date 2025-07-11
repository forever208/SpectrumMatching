import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
import lpips
from safetensors.torch import load_file
from tqdm.auto import tqdm

from utils import load_val_images, save_orig_and_generated_images, count_num_params, convert_to_PIL_imgs
from modules import VAE, LDMConfig, PatchGAN, init_weights
from modules import LPIPS as mylpips
from dataset import get_dataset
from eval_utils.utils import calculate_psnr_between_folders
from eval_utils.fid_score import calculate_fid_given_paths
from torchmetrics import StructuralSimilarityIndexMeasure
import shutil


### Load Arguments ###
def experiment_config_parser():
    parser = argparse.ArgumentParser(description="Experiment Configuration")
    parser.add_argument("--experiment_name", required=True, type=str, metavar="experiment_name")
    parser.add_argument("--working_directory", help="where checkpoints and logs are stored", required=True, type=str, metavar="working_directory")
    parser.add_argument("--eval_dir", help="where the eval images should be saved into", required=True, type=str, metavar="eval_dir")
    parser.add_argument("--eval_checkpoint",  help="name of ckpt folder", default=None, type=str, metavar="eval_checkpoint")
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
    accelerator = Accelerator(project_dir=path_to_experiment,)

    ### Load Model ###
    model = VAE(config).to(accelerator.device)
    latent_res = (config.img_size // (2**(len(config.vae_channels_per_block)-1)))
    accelerator.print(f"LATENT SPACE DIMENSIONS: {config.latent_channels, latent_res, latent_res}")
    accelerator.print(f"NUMBER OF VAE PARAMETERS: {count_num_params(model)}")

    ### Load LPIPS and SSIM ###
    use_lpips = False
    if train_cfg["use_lpips"]:
        use_lpips = True
        if train_cfg["use_lpips_package"]:
            lpips_loss_fn = lpips.LPIPS(net="vgg").eval()
            accelerator.print(f"using pretrained VGG for LPIPS")
        else:
            lpips_loss_fn = mylpips()
            lpips_loss_fn.load_checkpoint(train_cfg["lpips_checkpoint"])
        lpips_loss_fn = lpips_loss_fn.to(accelerator.device)
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0)).to(accelerator.device)

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
    accelerator.print("Number of Samples:", len(dataset))

    eval_dataloader = DataLoader(
        dataset,
        batch_size=mini_batchsize,
        pin_memory=False,
        num_workers=4,
        shuffle=False,
    )

    ### Load Checkpoint ###
    path_to_checkpoint = os.path.join(path_to_experiment, args.eval_checkpoint, 'model.safetensors')
    accelerator.print(f"Loading from Checkpoint: {path_to_checkpoint}")
    state_dict = load_file(path_to_checkpoint)
    model.load_state_dict(state_dict)

    ### Prepare Everything ###
    model, eval_dataloader, = accelerator.prepare(model, eval_dataloader)
    eval_iter = iter(eval_dataloader)
    if use_lpips:
        lpips_loss_fn = accelerator.prepare(lpips_loss_fn)
        ssim_fn = accelerator.prepare(ssim_fn)

    ### Evaluation ###
    eval_org_imgs_path = os.path.join(args.eval_dir, "eval_org_imgs")
    eval_recon_imgs_path = os.path.join(args.eval_dir, "eval_recon_imgs")
    eval_lpips = []
    eval_ssim = []

    if accelerator.is_main_process:
        os.makedirs(eval_org_imgs_path, exist_ok=True)
        os.makedirs(eval_recon_imgs_path, exist_ok=True)

    mini_batch_size = train_cfg["per_gpu_batch_size"]
    batch_size = mini_batch_size * accelerator.num_processes
    num_iterations = train_cfg["num_eval_images"] // batch_size + 1
    world_size = accelerator.state.num_processes
    local_rank = accelerator.state.local_process_index

    model.eval()
    accelerator.print(f"staring evaluation...")
    for j in tqdm(range(num_iterations), disable=not accelerator.is_main_process, desc='sample2dir'):
        try:
            mini_batch = next(eval_iter)
        except StopIteration:
            eval_iter = iter(eval_dataloader)  # Restart the iterator if we reach the end
            mini_batch = next(eval_iter)

        org_imgs = mini_batch["images"].to(accelerator.device)
        with torch.no_grad():
            recon_imgs = model(org_imgs)
            recon_imgs = recon_imgs["reconstruction"]
            eval_lpips.append(lpips_loss_fn(recon_imgs, org_imgs).mean())
            eval_ssim.append(ssim_fn(recon_imgs, org_imgs))

        org_imgs = convert_to_PIL_imgs(org_imgs)  # a list PIL images
        recon_imgs = convert_to_PIL_imgs(recon_imgs)  # a list PIL images

        for b_id in range(mini_batch_size):  # distributed image save
            img_id = j * mini_batch_size * world_size + local_rank * mini_batch_size + b_id

            if img_id >= train_cfg["num_eval_images"]:
                break

            org_imgs[b_id].save(os.path.join(eval_org_imgs_path, f"{img_id}.jpg"))
            recon_imgs[b_id].save(os.path.join(eval_recon_imgs_path, f"{img_id}.jpg"))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(f'{len(os.listdir(eval_org_imgs_path))} images in {eval_org_imgs_path}')
        accelerator.print(f'{len(os.listdir(eval_recon_imgs_path))} images in {eval_recon_imgs_path}')
        assert len(os.listdir(eval_recon_imgs_path)) == train_cfg["num_eval_images"]

        ### evaluate rFID ###
        accelerator.print(f"Evaluating rFID...")
        fid = calculate_fid_given_paths([eval_org_imgs_path, eval_recon_imgs_path], device=accelerator.device)

        accelerator.print(f"Evaluating PSNR...")
        psnr_values = calculate_psnr_between_folders(eval_org_imgs_path, eval_recon_imgs_path)
        avg_psnr = sum(psnr_values) / len(psnr_values)

        accelerator.print(f"Evaluating LPIPS...")
        eval_lpips = torch.tensor(eval_lpips)
        eval_lpips = eval_lpips.mean().item()

        accelerator.print(f"Evaluating SSIM...")
        eval_ssim = torch.tensor(eval_ssim)
        eval_ssim = eval_ssim.mean().item()

        accelerator.print(f"rFID={fid:.5f} PSNR={avg_psnr:.5f} LPIPS={eval_lpips:.5f} SSIM={eval_ssim:.5f}")
        with open(os.path.join(args.working_directory, f'eval.log'), 'a') as f:
            print(f'evaluating ckpt={path_to_checkpoint}', file=f)
            print(f'rFID={fid:.5f} PSNR={avg_psnr:.5f} LPIPS={eval_lpips:.5f} SSIM={eval_ssim:.5f}', file=f)

        shutil.rmtree(eval_org_imgs_path)  # remove the image folder
        shutil.rmtree(eval_recon_imgs_path)  # remove the image folder

    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
