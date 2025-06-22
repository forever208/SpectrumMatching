accelerate launch stage1_vae_trainer.py \
  --experiment_name "VAETrainer" \
  --wandb_run_name "vae_celeba" \
  --working_directory "work_dir/vae_celeba" \
  --training_config "configs/stage1_vae_train.yaml" \
  --model_config "configs/ldm.yaml" \
  --dataset celebahq \
  --path_to_dataset "/mnt/datadrive/data/CelebAMask-HQ/CelebA-HQ-img" \
  --path_to_save_gens "src/celebahq_vae_gen"


