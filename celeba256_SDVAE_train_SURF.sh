#!/bin/bash
#SBATCH --job-name=VAE
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=2  # gpus for each node --gres=gpu:8, --gpus-per-node=8
#SBATCH -t 0-52:00  # 运行总时间，天数-小时数-分钟， D-HH:MM  10, 20, 60
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/mning/LDM_exps/celeba256_SDVAE_f16d16_bf16_b48_lr5e5_flip/train_log.txt
#SBATCH --error=/home/mning/LDM_exps/celeba256_SDVAE_f16d16_bf16_b48_lr5e5_flip/train_error.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ningmang666@gmail.com
#SBATCH --account=uusei5190
#SBATCH --partition=gpu_h100

module load 2023
module load Anaconda3/2023.07-2
source activate SDVAE
nvidia-smi

accelerate launch --multi_gpu --num_processes 2 --num_cpu_threads_per_process 16 \
--mixed_precision bf16 /home/mning/LDM/stage1_vae_trainer.py \
--working_directory /home/mning/LDM_exps/celeba256_SDVAE_f16d16_bf16_b48_lr5e5_flip \
--eval_dir /home/mning/LDM_exps/celeba256_SDVAE_f16d16_bf16_b48_lr5e5_flip \
--log_wandb --experiment_name SDVAE --wandb_run_name celeba256_SDVAE_f16d16_bf16_b48_lr5e5_flip \
--training_config /home/mning/LDM/configs/train_vae_celeba256.yaml \
--model_config /home/mning/LDM/configs/ldm_f16d16.yaml \
--dataset celeba256 --path_to_dataset /projects/prjs0865/datasets/celeba256/celeba256

# sbatch celeba256_SDVAE_train_SURF.sh
# sbatch --nodelist=gcn102,gcn132 celeba256_SDVAE_train_SURF.sh