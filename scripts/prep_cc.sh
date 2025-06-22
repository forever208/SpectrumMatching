python prep_cc.py --path_to_data_root /mnt/datadrive/data/ConceptualCaptions/train \
                  --path_to_save /mnt/datadrive/data/ConceptualCaptions/hf_train_unencoded \
                  --hf_clip_model_name openai/clip-vit-large-patch14 \
                  --hf_cache_dir /mnt/datadrive/data/ConceptualCaptions/hf_cache \
                  --cpu_batch_size 512 \
                  --gpu_batch_size 256 \
                  --num_cpu_workers 32 \
                  --dtype bfloat16