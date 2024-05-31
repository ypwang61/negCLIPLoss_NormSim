# Databricks notebook source
#import open_clip
import torch
from clip_benchmark.datasets.builder import build_dataset
import argparse
import os
import sys
from datasets import load_dataset

import clip
import tqdm
import numpy as np
import yaml
# COMMAND ----------

import torch.nn.functional as F

from tqdm import tqdm
import clip

import time


parser = argparse.ArgumentParser(description="Generate embeddings for eval_train")
parser.add_argument("--model_name", type=str, default="ViT-L/14", help="CLIP model name")
parser.add_argument("--cache_dir", type=str, default="", help="cache directory")
parser.add_argument('--file_dir', type=str, default="", help='file directory')
parser.add_argument("--output_dir_name", type=str, default="eval_train", help="output directory name")

args = parser.parse_args()

for arg in vars(args):
    print(arg, getattr(args, arg))
    
amp = True

cache_dir = args.cache_dir

variance_device = "cuda:0"
general_device = "cuda:1"
clip_bs = 64 * 16 * 8

# define task list path
with open("eval_train.yml") as f:
    tasks = yaml.safe_load(f)
    
for task, task_info in list(tasks.items()):
    task_name = task_info.get("name", task)
    download_path = task_info.get("download", task)
    print(f'task: {task}, task_name: {task_name}, download_path: {download_path}')


print(clip.available_models())

# model, preprocess = clip.load("ViT-L/14", download_root=cache_dir)
model, preprocess = clip.load(args.model_name, download_root=cache_dir)

model.to(general_device).eval()

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

# COMMAND ----------




# specify output path for embeddings
# output_datapath = os.path.join(args.file_dir, args.output_dir_name, "variance/")
# output_datapath_standard = os.path.join(args.file_dir, args.output_dir_name, "variance_standard/")
output_feature_path = os.path.join(args.file_dir, args.output_dir_name, "features/")

if not os.path.exists(output_feature_path):
    os.makedirs(output_feature_path)
    print(f'creating output path: {output_feature_path}')

# if not os.path.exists(output_datapath):
#     os.makedirs(output_datapath)
#     print(f'creating output path: {output_datapath}')

# if not os.path.exists(output_datapath_standard):
#     os.makedirs(output_datapath_standard)
#     print(f'creating output path: {output_datapath_standard}')

def calculate_variance(
        embs: torch.Tensor,
        batch_size: int = None,
        device: str=variance_device
    ):
    # calculate variance of embeddings, i.e., embs.T @ embs / embs.shape[0]
    # embs: (N, D) where N is the number of examples and D is the dimension of the embeddings
    # returns: (D, D) matrix
    
    print(f'begin calculating variance of embeddings')
    if batch_size is None or batch_size >= embs.shape[0]:
        # device
        embs = embs.to(device)
        variance = embs.T @ embs / embs.shape[0]
    else:
        variance = torch.zeros((embs.shape[1], embs.shape[1])).to(device)
        for i in tqdm(range(0, embs.shape[0], batch_size)):
            end = min(i+batch_size, embs.shape[0])
            batch = embs[i:end].to(device)
            variance += batch.T @ batch
        
        variance /= embs.shape[0]
        
    return variance.cpu()


def calculate_standard_variance(
    embs: torch.Tensor,
    batch_size: int = None,
    device: str="cuda"
):
    # first minus mean
    embs_tmp = embs - embs.mean(dim=0)
    variance_standard = calculate_variance(embs_tmp, batch_size=batch_size, device=device)

    return variance_standard



def create_embedding_webdataset(task, data_root=None,  batch_size=None, download_path=None, device="cuda"):
    
    
    # data_root = f"https://huggingface.co/datasets/djghosh/{data_folder}/tree/main"
    
    data_root = f"https://huggingface.co/datasets/{download_path}/tree/main"

    print(f'data_root: {data_root}, begin processing task: {task}')
    start = time.time()
    
    
    dataset = build_dataset(
        dataset_name=f"wds/{task}",
        root=data_root,
        split="train",
        transform = preprocess,
        download=True,
        cache_dir=cache_dir,
    )
    
    print(f'loading {task} dataset took {time.time() - start} seconds')
    
    num_gpus =torch.cuda.device_count()
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=clip_bs,
            shuffle=False,
            num_workers=num_gpus,
        )

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    autocast = torch.cuda.amp.autocast if amp else torch.no_grad

    image_feat = []

    with torch.no_grad(), autocast():
        start = time.time()
        for images, target in tqdm(dataloader, desc=f"Processing {task}, now num is {len(image_feat) * clip_bs}"):
            images = images.to(device)
            #target = target.to(device)

            #image_features = model.get_image_features(**images)
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim = -1)
            image_feat.append(image_features.cpu())
            #text_feat.append(target.cpu())

        
        image_feat = torch.cat(image_feat).cpu()
        
        print(f'image_feat.shape {image_feat.shape}')
        image_feat_path=f"{output_feature_path}{task.replace('/','-')}_{image_feat.shape[0]}.pt"
        
        print(f'saving features to {image_feat_path}, the size that image_feat will cost is {image_feat.element_size() * image_feat.nelement() / 1024 / 1024 / 1024} GB, took {time.time() - start} seconds\n In average, {(time.time() - start) / len(image_feat) * 1110000 / 36 } A40 hours per 111M data')
        
        # store features
        torch.save(image_feat, image_feat_path)
        
        
        # # variance
        # variance = calculate_variance(image_feat, batch_size=batch_size)
        # variance_path=f"{output_datapath}{task.replace('/','-')}_{image_feat.shape[0]}.pt"
        # torch.save(variance, variance_path)
        # print(f'saving variance to {variance_path}')
        
        
        # # standard variance
        # variance_standard = calculate_standard_variance(image_feat, batch_size=batch_size)
        # variance_standard_path=f"{output_datapath_standard}{task.replace('/','-')}_{image_feat.shape[0]}.pt"
        # torch.save(variance_standard, variance_standard_path)
        # print(f'saving standard variance to {variance_standard_path}')


print(f'num of tasks: {len(tasks)}')
# for task, task_info in tqdm(list(tasks.items())[30:]): # checkpoint
for task, task_info in tqdm(list(tasks.items())):
    download_path = task_info.get("download", task)
    
    print(f'begin processing task: {task}')
    
    create_embedding_webdataset(task, data_root=None,  batch_size=10000, download_path=download_path, device=general_device)
    
    
    # if task.startswith("retrieval/"):
    #     continue

    # elif task.startswith("wilds/"):
    #     continue
    # elif task.startswith("fairness/"):
    #     continue
    # elif task.startswith("misc/"):
    #     continue
    # else:
    #     if task.startswith('imagenet'):
            # print(f"Running for task: {task}")
            # create_embedding_webdataset(task, output_datapath, data_root=None,  batch_size=None)



