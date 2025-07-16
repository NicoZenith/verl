# Copyright 2025  –  Apache-2.0
import json, argparse, os, re, datasets
from tqdm import tqdm
from pathlib import Path
from verl.utils.hdfs_io import copy, makedirs          # optional HDFS sync

DATA_SOURCE = "mimic_grpo"        # arbitrary tag used by RewardManager
ABILITY     = "radiology"
PROMPT_SUFFIXE = " First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
TARGET_TOKENS = 1024  
MAX_PIXELS    = TARGET_TOKENS * 28 * 28    # ≈0.8 Mpx  (896×896)


def make_map_fn(split):
    def proc(example, idx):
        img_entry = {
            "image": f"file://{example['image']}",
            "resized_height": 512,           # 24×28px  →  576 tokens
            "resized_width":  512
        }
        # user prompt (keep the <image> token so Qwen knows where the pixel goes)
        prompt_msg = {
            "role": "user",
            "content": example["conversations"][0]["value"] + PROMPT_SUFFIXE
        }
        return {
            "data_source" : DATA_SOURCE,
            "prompt"      : [prompt_msg],              # same schema as Geo3K example
            "images": [img_entry],     
            "ability"     : ABILITY,
            "reward_model": {
                "style": "rule",
                "ground_truth": example["conversations"][1]["value"]      # will be parsed by reward fn
            },
            "extra_info"  : {                          # anything you want to keep
                "id":  example["id"],
                "split": split,
                "index": idx
            },
        }
    return proc

def main(json_path, local_dir, hdfs_dir=None, train_ratio=0.999):
    with open(json_path) as f:          # ← use the file path
        rows = json.load(f)             # ← module 'json' is still accessible

    ds = datasets.Dataset.from_list(rows).shuffle(seed=42)
    n  = int(len(ds) * train_ratio)

    train_ds = ds.select(range(n)).map(make_map_fn("train"), with_indices=True, num_proc=8)
    val_ds   = ds.select(range(n, len(ds))).map(make_map_fn("val"),   with_indices=True, num_proc=8)

    os.makedirs(local_dir, exist_ok=True)
    train_ds.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_ds.to_parquet  (os.path.join(local_dir, "val.parquet"))

    if hdfs_dir:
        makedirs(hdfs_dir); copy(src=local_dir, dst=hdfs_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")              # ← new name
    parser.add_argument("--local_dir", default="~/data/chexpert_mm")
    parser.add_argument("--hdfs_dir",  default=None)
    main(**vars(parser.parse_args()))
