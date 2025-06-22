accelerate launch stage1_vqvae_trainer.py \
  --experiment_name "VQVAETrainer" \
  --wandb_run_name "vqvae_celeba" \
  --working_directory "work_dir/vqvae_celeba" \
  --training_config "configs/stage1_vqvae_train.yaml" \
  --model_config "configs/ldm.yaml" \
  --dataset celebahq \
  --path_to_dataset "/mnt/datadrive/data/CelebAMask-HQ/CelebA-HQ-img" \
  --path_to_save_gens "src/celebahq_vqvae_gen"
