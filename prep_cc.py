import os
import torch
from PIL import Image
from io import BytesIO
from multiprocess import set_start_method
from transformers import AutoTokenizer, CLIPTextModel
from datasets import load_dataset, load_from_disk
import argparse

parser = argparse.ArgumentParser(description="Pretokenize All our Text!")

parser.add_argument("--path_to_data_root",
                    help="Point to the folder with all the Parquet files that img2dataset downloaded",
                    type=str,
                    required=True)

parser.add_argument("--path_to_save", 
                    help="Saving directory of final dataset", 
                    type=str,
                    required=True)

parser.add_argument("--hf_clip_model_name",
                    help="What Clip Backbone do you want to use to encode your text?",
                    default="openai/clip-vit-large-patch14",
                    type=str,
                    required=False)

parser.add_argument("--hf_cache_dir",
                    help="What do you want as your HF Cache Directory",
                    default=None,
                    type=str,
                    required=False)

parser.add_argument("--cpu_batch_size",
                    help="What batch size do you want to use to process your images?",
                    default=256,
                    type=int,
                    required=False)

parser.add_argument("--gpu_batch_size",
                    help="What batch size do you want to use to process your text?",
                    default=64,
                    type=int,
                    required=False)

parser.add_argument("--num_cpu_workers",
                    help="Number of CPU Cores to batch processes",
                    default=32, 
                    type=int,
                    required=False)

parser.add_argument("--dtype", 
                    default="float32",
                    type=str,
                    required=False)

parser.add_argument("--pre_encode_text",
                    default=False, 
                    action=argparse.BooleanOptionalAction)

args = parser.parse_args()

if args.dtype == "float32":
    torch_dtype = torch.float32
elif args.dtype == "float16":
    torch_dtype = torch.float16
elif args.dtype == "bfloat16":
    torch_dtype = torch.bfloat16
else:
    raise ValueError("Select from float32, float16, bfloat16")

### Load Clip Model and Tokenizer ###
tokenizer = AutoTokenizer.from_pretrained(args.hf_clip_model_name)
text_encoder = CLIPTextModel.from_pretrained(args.hf_clip_model_name, 
                                             torch_dtype=torch_dtype).eval()

def embed_text(batch, rank):
    
    ### Rank is from 0 to n_gpus, if rank is not provided then it is None and defaults to 0 ###
    device = f"cuda:{(rank or 0) % torch.cuda.device_count()}"
    text_encoder.to(device)

    try:

        captions = tokenizer(batch["caption"], 
                             padding=True, 
                             truncation=True,
                             return_tensors="pt",
                             max_length=77).to(device)

        with torch.no_grad():
            with torch.autocast(device):
                outputs = text_encoder(**captions).last_hidden_state

        ### Index Out Pad Tokens ###
        prepped_outputs = []
        for output, padding in zip(outputs, captions["attention_mask"]):
            num_non_padding = torch.sum(padding)
            output = output[:num_non_padding]
            prepped_outputs.append(output)

        batch["encoded_text"] = prepped_outputs
    
    except:
        print("Failed Batch!")
        batch["encoded_text"] = [None for _ in range(len(batch["caption"]))]

    return batch

def bin2pil(examples):

    images = []
    for image in examples["jpg"]:
        image = Image.open(BytesIO(image))
        
        images.append(image)

    examples["image"] = images

    return examples

if __name__ == "__main__":
    set_start_method("spawn")

    ### Load Dataset from Parquet ###
    parquet_files = [
        os.path.join(args.path_to_data_root, file) \
            for file in os.listdir(args.path_to_data_root) \
                if ".parquet" in file
    ]

    dataset = load_dataset("parquet", 
                            data_files={'train': parquet_files}, 
                            num_proc=args.num_cpu_workers,
                            cache_dir=args.hf_cache_dir)

    ### Remove Rows that were Unsuccessful ###
    dataset = dataset.filter(lambda example: example["status"] == "success", num_proc=args.num_cpu_workers)

    ### Remove Columns We Dont Need ###
    cols_to_rm = [col for col in dataset.column_names["train"] if col not in ["caption", "jpg"]]
    dataset = dataset.remove_columns(cols_to_rm)

    ### Convert Image Bytes to PIL ###
    dataset = dataset.map(bin2pil, batched=True, batch_size=args.cpu_batch_size, num_proc=args.num_cpu_workers)
    dataset = dataset.remove_columns("jpg")

    ### Encode All Text ###
    if args.pre_encode_text:

        print(f"Prepping on {torch.cuda.device_count()} GPUs")
        dataset = dataset.map(embed_text,
                            batched=True, 
                            batch_size=args.gpu_batch_size,
                            with_rank=True, 
                            num_proc=torch.cuda.device_count())
        dataset = dataset.remove_columns("caption")

        ### Remove Rows that were Unsuccessful ###
        dataset = dataset.filter(lambda example: example["encoded_text"] is not None, num_proc=args.num_cpu_workers)
    
    ### Final Dataset ###
    print(dataset)
    
    ### Save Dataset ###
    dataset.save_to_disk(args.path_to_save)

    ### Clear Cache ###
    num_cache_files = dataset.cleanup_cache_files()
    print(f"Cleaned {num_cache_files} Cache Files")

    ### Verify Dataset ###
    dataset = load_from_disk(args.path_to_save)
    print(dataset)