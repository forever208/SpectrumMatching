img2dataset --url_list "/scratch/bbug/priyamm2/conceptual_captions/cc3m.tsv" \
            --input_format "tsv"\
            --url_col "url" \
            --caption_col "caption" \
            --output_format parquet \
            --output_folder "/scratch/bbug/priyamm2/conceptual_captions/train" \
            --processes_count 32 \
            --thread_count 128 \
            --image_size 256 \
            --resize_mode keep_ratio

