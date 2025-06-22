python compute_vae_scaling.py \
    --path_to_pretrained_weights "work_dir/vae_celeba/model.safetensors" \
    --dataset "celebahq" \
    --path_to_dataset "/mnt/datadrive/data/CelebAMask-HQ/CelebA-HQ-img/"

python compute_vae_scaling.py \
    --path_to_pretrained_weights "work_dir/vae_imagenet/model.safetensors" \
    --dataset "imagenet" \
    --path_to_dataset "/mnt/datadrive/data/ImageNet/train/" \
    --batch_size 128 \
    --num_batches 10000 \
    --num_workers 32 

python compute_vae_scaling.py \
    --path_to_pretrained_weights "work_dir/vae_cc/model.safetensors" \
    --dataset "conceptual_captions" \
    --path_to_dataset "/mnt/datadrive/data/ConceptualCaptions/hf_train" \
    --batch_size 128 \
    --num_batches 10000 \
    --num_workers 32 
