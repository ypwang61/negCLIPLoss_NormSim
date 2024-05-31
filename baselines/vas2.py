import multiprocessing as mp
import os
import time
from queue import Empty
from typing import Union

import fasttext
import fsspec
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from baselines.utils import download
from scipy.stats import rankdata

fasttext.FastText.eprint = lambda x: None

import copy
import scipy
import math


import sys
sys.path.append('../../research_tools') # for the visualization
from visualization2 import ImageCaptionVisualizer, ScatterPlot

import random

# ---- debug
counter_num = 2
debug = 0


timeout = 200 # 50
# datacomp
text_name = "text" #"caption" # "text" # for datacomp
key_name = "clip_l14_similarity_score" # "clip_l14_similarity_score" # "oai-clip-vit-l14-score" # for cc12m
feature_name = "l14_img"
feature_text_name = "l14_txt"

# cc12m
# text_name = "caption"
# key_name = "oai-clip-vit-l14-score"
# feature_name = "oai-clip-vit-l14-image"

# print(f'It\'s VAS 2 !!')

def get_ranks(scores):
    arg_index = np.argsort(scores)
    ranks = np.empty_like(arg_index)
    ranks[arg_index] = np.arange(len(scores))
    return ranks

def get_small_ranks(scores):
    # max num - get_ranks(scores)
    return len(scores) - get_ranks(scores)

def get_ranks_ratios(scores):
    return get_ranks(scores) * 1.0 / len(scores)
    
    
@torch.no_grad()
def filter_by_given_uids(given_uids: np.ndarray, uids: np.ndarray, given_uids_index_in_ordered_uids_path: str) -> np.ndarray:
    """ return the index of uids that are in the given uids in a parallel way

    Args:
        given_uids (np.ndarray): given uids
        uids (np.ndarray): uids to be filtered
        
        The format of uids is 
        np.concatenate(
            [np.array(
                    [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids],
                    np.dtype("u8,u8"),
                )]
        )
        
    Returns:
        np.ndarray: index of uids that are in the given uids
    
    """
    
    # sort to accelerate the search, and obtain the sorted indices in order for align vas and css
    start = time.time()
    sort_indices = np.argsort(uids)
    uids = uids[sort_indices]
    print(f'end to sort uids, time = {time.time() - start}')
    # given_uids dont need to be sorted
    
    if given_uids_index_in_ordered_uids_path is not None and os.path.exists(given_uids_index_in_ordered_uids_path):
        indices = np.load(given_uids_index_in_ordered_uids_path)
        print(f'load given_uids_index_in_ordered_uids, len = {len(indices)}, path = {given_uids_index_in_ordered_uids_path}')
    else:
        # find the index of uids that are in the given uids
        start2 = time.time()
        print('begin to search uids')
        indices = np.searchsorted(uids, given_uids)
        print(f'end to search uids, time = {time.time() - start2}')
        indices = indices[indices < len(uids)]
        indices = indices[uids[indices] == given_uids] # filter out the uids that are not in the given uids
        
        if given_uids_index_in_ordered_uids_path is not None:
            np.save(given_uids_index_in_ordered_uids_path, indices)
            print(f'save given_uids_index_in_ordered_uids, len = {len(indices)}, path = {given_uids_index_in_ordered_uids_path}')
        
    # align the indices to the original order
    indices = sort_indices[indices]
    
    print(f'end to filter_by_given_uids, total time = {time.time() - start}')
    
    return indices

@torch.no_grad()
def filter_by_score(scores: np.ndarray, fraction: float, threshold: float, total_num: int, name: str = '') -> np.ndarray:
    """ filter the score by fraction and threshold
    
    Args:
        scores (np.ndarray): score to be filtered
        fraction (float): fraction to be filtered
        threshold (float): threshold to be filtered
        total_num (int): total number of scores
        name (str): name of the score
    
    Returns:
        np.ndarray: index of the score that are in the given uids
    """
    assert fraction is not None or threshold is not None, "fraction or threshold should be specified"
    
    if fraction is not None:
        
        n = int(total_num * fraction)
        
        print(f'The threshold for {name} is not specified, select top {fraction} fraction. begin to sort score.')
        select_indices = np.argpartition(scores, -n)[-n:] 
    
    else: # threshold is not None
        print(f'The fraction for {name} is not specified, threshold is {threshold}.')
        select_indices = np.where(scores >= threshold)[0]
    
    scores_tmp = scores[select_indices]
    print(f'After filtering {name}, the fraction = {len(scores_tmp)/total_num}, the threshold = {np.min(scores_tmp)}, mean = {np.mean(scores_tmp)}, max = {np.max(scores_tmp)}. len = {len(scores_tmp)}')
        
    return select_indices
        
@torch.no_grad()
def soft_filter_by_score(scores: np.ndarray, fraction: float, total_num: int, name: str = '', soft_type:float = 0.) -> np.ndarray:
    """
    sample based on the probability proportional to function of the scores
    
    soft_type = 0 -> hard filter
    soft_type > 0 -> scores ** soft_type
    
    """
    
    if soft_type > 0:
        prop = scores ** soft_type
    elif soft_type < 0: # exponential
        prop = np.exp(scores * -soft_type)
    else:
        raise NotImplementedError
    
    prop = prop / prop.sum()
    
    n = int(total_num * fraction)
    
    len_prop = len(prop)
    
    select_indices = np.random.choice(len_prop, n, p=prop, replace=False)
    
    scores_tmp = scores[select_indices]
    
    print(f'After filtering {name}, the fraction = {len(scores_tmp)/total_num}, the threshold = {np.min(scores_tmp)}, mean = {np.mean(scores_tmp)}, max = {np.max(scores_tmp)}. len = {len(scores_tmp)}')
    
    return select_indices
    
    


@torch.no_grad()
def get_vas_gpu(
    embeddings: torch.Tensor, target_variance: torch.Tensor, device: int
) -> torch.Tensor:
    """ calculate vas for each embeddings. VAS(i) = f_i^T S f_i, where S is the target variance matrix, f_i is the i-th embedding
    
    Args:
        embeddings (torch.Tensor): embeddings to calculate vas
        target_variance (torch.Tensor): target variance matrix
        device (int): gpu number
        
    Returns:
        torch.Tensor: variance alignment score for each embeddings
    """
    device_string = f"cuda:{device}"
    target_variance_gpu = target_variance.float().to(device_string)
    embeddings_gpu = embeddings.float().to(device_string)
    
    vas = torch.sum((embeddings_gpu @ target_variance_gpu) * embeddings_gpu, dim=1).cpu()
    
    return vas

@torch.no_grad()
def vas_filter_helper(
    device_index: int,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    arch: Union[str, None] = None,
    
    target_variance: torch.Tensor = None,
    is_vas_d: Union[bool, None] = False,
    update_image_feature_arch: bool = True,
    store_variance: bool = False,
) -> None:
    """worker function to variance alignment score filtering, pulling off a queue of tasks
    
    Args:
        target_variance (torch.Tensor): target variance matrix
        device_index (int): device on which to run the gpu processing
        in_queue (mp.Queue): task queue with fsspec, metadata path pairs
        out_queue (mp.Queue): output queue to send filtred uids
        arch: (Union[str, None]): If specified, we want to apply a threshold to arch=B/32 or L/14 clip scores. Defaults to None.
        if_vas_d (Union[bool, None]): if True,  we will return the candidate_embedding and don't return vas. Defaults to False.
    
    """
    
    while True:
        fs_root = None
        try:
            fs_root = in_queue.get(timeout=1)
        except Empty:
            # case where the queue is depleated, worker should return
            break
        
        fs, path_root = fs_root
        
        feature_name = "l14_img"
        if arch is not None:
            key = key_name
            feature_name = "l14_img"
            
            if arch == "b32":
                key = "clip_b32_similarity_score"
                if update_image_feature_arch:
                    feature_name = "b32_img"
                
                feature_name = "b32_img"
            
            
            df = pd.read_parquet(
                f"{path_root}.parquet", columns=["uid", text_name, key], filesystem=fs
            )
        else:
            df = pd.read_parquet(
                f"{path_root}.parquet", columns=["uid", text_name], filesystem=fs
            )
        # print(f'feature_name = {feature_name}, key = {key}, arch = {arch}')

        candidate_embedding = None
        with fs.open(f"{path_root}.npz") as f: # -> float32
            candidate_embedding = torch.from_numpy(np.load(f)[feature_name])#.float()

        uids = df["uid"].values
        
        uids_standard = np.array(
                            [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids],
                            np.dtype("u8,u8"),
                        )
        
        # clip scores
        if arch is not None:
            css = df[key].values
        else:
            css = None
        
        if is_vas_d or store_variance:
            out_queue.put(
                (
                    uids_standard,
                    candidate_embedding.cpu(), # will calculate VAS in the main thread
                    css,
                )
            )
        else:
            vass = get_vas_gpu(
                candidate_embedding,
                target_variance,
                device_index,
            )
            out_queue.put(
                (
                    uids_standard,
                    vass.numpy(),
                    css,
                )
            )      


@torch.no_grad()
def vas_filter_helper_vis(
    device_index: int,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    arch: Union[str, None] = None,
    
    target_variance: torch.Tensor = None,
    
) -> None:
    """worker function to variance alignment score filtering, pulling off a queue of tasks
    
    Args:
        target_variance (torch.Tensor): target variance matrix
        device_index (int): device on which to run the gpu processing
        in_queue (mp.Queue): task queue with fsspec, metadata path pairs
        out_queue (mp.Queue): output queue to send filtred uids
        arch: (Union[str, None]): If specified, we want to apply a threshold to arch=B/32 or L/14 clip scores. Defaults to None.
    
    """
    
    while True:
        fs_root = None
        try:
            fs_root = in_queue.get(timeout=1)
        except Empty:
            # case where the queue is depleated, worker should return
            break
        
        fs, path_root = fs_root
        
        feature_name = "l14_img"
        if arch is not None:
            key = key_name
            if arch == "b32":
                key = "clip_b32_similarity_score"
                feature_name = "b32_img"
            
            
            df = pd.read_parquet(
                f"{path_root}.parquet", columns=["uid", text_name, key, "url"], filesystem=fs
            )
        else:
            df = pd.read_parquet(
                f"{path_root}.parquet", columns=["uid", text_name, "url"], filesystem=fs
            )
        # print(f'feature_name = {feature_name}, key = {key}, arch = {arch}')

        candidate_embedding = None
        with fs.open(f"{path_root}.npz") as f: # -> float32
            candidate_embedding = torch.from_numpy(np.load(f)[feature_name])#.float()

        uids = df["uid"].values
        urls = df["url"].values
        captions = df[text_name].values
        
        uids_standard = np.array(
                            [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids],
                            np.dtype("u8,u8"),
                        )
        
        # clip scores
        if arch is not None:
            css = df[key].values
        else:
            css = None
        
        
        vass = get_vas_gpu(
            candidate_embedding,
            target_variance,
            device_index,
        )
        out_queue.put(
            (
                uids_standard,
                vass.numpy(),
                css,
                captions,
                urls,
            )
        )      


@torch.no_grad()
def load_all_data_vis(
        metadata_dir_path: str,
        arch: str, 
        num_gpus: int,
        
        given_uids_path: str = None,
        target_variance: torch.Tensor = None,
        downsample_rate: float = 0.001,
    ):
    """Load embeddings, UIDs, and CLIP scores from files, filter by given UIDs and CLIP score threshold, and calculate initial target variance.
    
    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        arch (str): Kind of features for CLIP filtering.
        num_gpus (int): Number of GPU workers.
        
        given_uids_path (str): Path to the given UIDs.
        target_variance (torch.Tensor): Target variance matrix.
    """
    
        
        
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    root_paths = [
        (fs, str(x.split(".parquet")[0])) for x in fs.ls(url) if ".parquet" in x
    ]

    # initializing task queues
    receive_queue = mp.Queue()
    send_queue = mp.Queue()

    for job in root_paths:
        send_queue.put(job)

    processes = []
    print("starting gpu workers")
    for worker_index in tqdm(range(num_gpus)):
        p = mp.Process(
            target=vas_filter_helper_vis,
            kwargs=dict(
                device_index=worker_index,
                in_queue=send_queue,
                out_queue=receive_queue,
                arch=arch,
                target_variance=target_variance,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.05)

    print("processing metadata with gpu workers")
    pbar = tqdm(total=len(root_paths))

    # running in main thread
    all_uids = []
    all_vass = []
    all_css = []
    all_urls = []
    all_captions = []
    
    
    def update_p(receive_queue, all_uids, all_vass, all_css, all_captions, all_urls, pbar):
        # utility function to update the progress bar and store results
        uids, embs_or_vass, css, captions, urls = receive_queue.get(timeout=10)
        all_uids.append(uids)
        all_vass.append(embs_or_vass)
        all_css.append(css)
        all_captions.append(captions)
        all_urls.append(urls)
        
        pbar.update(1)
        
        return pbar.n

    
    while True:
        # keep checking for jobs finishing and update uids
        try:
            counter = update_p(receive_queue, all_uids, all_vass, all_css, all_captions, all_urls, pbar)
            if debug == 1 and counter == counter_num: break
        except TimeoutError:
            if all(not p.is_alive() for p in processes):
                try:
                    update_p(receive_queue, all_uids, all_vass, all_css, all_captions, all_urls, pbar)
                    
                except TimeoutError:
                    raise RuntimeError("All processes dead but nothing in queue!")
        except Empty:
            pass

        if all(not p.is_alive() for p in processes):
            # case where all workers have exited
            try:
                update_p(receive_queue, all_uids, all_vass, all_css, all_captions, all_urls, pbar)
                
            except Empty:
                print("Result queue is empty and all workers have exited")
                break

    pbar.close()
    for p in processes:
        p.join(timeout=1)

    
    uids = np.concatenate(all_uids)
    
    vass = np.concatenate(all_vass).astype(np.float32)

    css = np.concatenate(all_css)
    
    captions = np.hstack(all_captions)
    
    urls = np.hstack(all_urls)
    
    print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}, css.shape = {css.shape}, captions.shape = {captions.shape}, urls.shape = {urls.shape}')
    
    # Filter by given UIDs, which may be the subet from other filtering methods
    
    if downsample_rate < 1:
        assert downsample_rate > 0, "downsample_rate should be larger than 0"
        print(f'downsample_rate = {downsample_rate}')
        num = int(len(uids) * downsample_rate)
        indices = np.random.choice(len(uids), num, replace=False)
        uids = uids[indices]
        vass = vass[indices]
        css = css[indices]
        captions = captions[indices]
        urls = urls[indices]

        print(f'================== after downsampling {downsample_rate} ==================')
        print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}, css.shape = {css.shape}, captions.shape = {captions.shape}, urls.shape = {urls.shape}')
        
        
    if given_uids_path is not None:
        given_uids = np.load(given_uids_path)
        
        given_uids_mask = np.isin(uids, given_uids)
        uids = uids[given_uids_mask]
        vass = vass[given_uids_mask] # note that tensor can take np.ndarray as index
        css = css[given_uids_mask]
        captions = captions[given_uids_mask]
        urls = urls[given_uids_mask]
        
        
        
        print(f'================== after filtering by given_uids ==================')
        print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}, css.shape = {css.shape}, captions.shape = {captions.shape}, urls.shape = {urls.shape}')
    
    # Calculate initial target variance as the sum of outer products
    return uids, vass, css, captions, urls



@torch.no_grad()
def load_all_data_vis2(
        metadata_dir_path: str,
        arch: str, 
        num_gpus: int,
        
        given_uids_path: str = None,
        target_variance: torch.Tensor = None,
        downsample_rate: float = 0.001,
    ):
    """Load embeddings, UIDs, and CLIP scores from files, filter by given UIDs and CLIP score threshold, and calculate initial target variance.
    return given_uids
    
    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        arch (str): Kind of features for CLIP filtering.
        num_gpus (int): Number of GPU workers.
        
        given_uids_path (str): Path to the given UIDs.
        target_variance (torch.Tensor): Target variance matrix.
    """
    
        
        
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    root_paths = [
        (fs, str(x.split(".parquet")[0])) for x in fs.ls(url) if ".parquet" in x
    ]

    # initializing task queues
    receive_queue = mp.Queue()
    send_queue = mp.Queue()

    for job in root_paths:
        send_queue.put(job)

    processes = []
    print("starting gpu workers")
    for worker_index in tqdm(range(num_gpus)):
        p = mp.Process(
            target=vas_filter_helper_vis,
            kwargs=dict(
                device_index=worker_index,
                in_queue=send_queue,
                out_queue=receive_queue,
                arch=arch,
                target_variance=target_variance,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.05)

    print("processing metadata with gpu workers")
    pbar = tqdm(total=len(root_paths))

    # running in main thread
    all_uids = []
    all_vass = []
    all_css = []
    all_urls = []
    all_captions = []
    
    
    def update_p(receive_queue, all_uids, all_vass, all_css, all_captions, all_urls, pbar):
        # utility function to update the progress bar and store results
        uids, embs_or_vass, css, captions, urls = receive_queue.get(timeout=10)
        all_uids.append(uids)
        all_vass.append(embs_or_vass)
        all_css.append(css)
        all_captions.append(captions)
        all_urls.append(urls)
        
        pbar.update(1)
        
        return pbar.n
                                
    while True:
        # keep checking for jobs finishing and update uids
        try:
            counter = update_p(receive_queue, all_uids, all_vass, all_css, all_captions, all_urls, pbar)
            if debug == 1 and counter == counter_num: break
        except TimeoutError:
            if all(not p.is_alive() for p in processes):
                try:
                    update_p(receive_queue, all_uids, all_vass, all_css, all_captions, all_urls, pbar)
                    
                except TimeoutError:
                    raise RuntimeError("All processes dead but nothing in queue!")
        except Empty:
            pass

        if all(not p.is_alive() for p in processes):
            # case where all workers have exited
            try:
                update_p(receive_queue, all_uids, all_vass, all_css, all_captions, all_urls, pbar)
                
            except Empty:
                print("Result queue is empty and all workers have exited")
                break

    pbar.close()
    for p in processes:
        p.join(timeout=1)

    
    uids = np.concatenate(all_uids)
    
    vass = np.concatenate(all_vass).astype(np.float32)

    css = np.concatenate(all_css)
    
    captions = np.hstack(all_captions)
    
    urls = np.hstack(all_urls)
    
    print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}, css.shape = {css.shape}, captions.shape = {captions.shape}, urls.shape = {urls.shape}')
    
    # Filter by given UIDs, which may be the subet from other filtering methods
    
    if downsample_rate < 1:
        assert downsample_rate > 0, "downsample_rate should be larger than 0"
        print(f'downsample_rate = {downsample_rate}')
        num = int(len(uids) * downsample_rate)
        indices = np.random.choice(len(uids), num, replace=False)
        uids = uids[indices]
        vass = vass[indices]
        css = css[indices]

        print(f'================== after downsampling {downsample_rate} ==================')
        print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}, css.shape = {css.shape}')
        
    if given_uids_path is not None:
        given_uids = np.load(given_uids_path)
        
        given_uids_mask = np.isin(uids, given_uids)
        given_uids = uids[given_uids_mask]
        given_vass = vass[given_uids_mask] # note that tensor can take np.ndarray as index
        given_css = css[given_uids_mask]
        
        
        
        print(f'================== after filtering by given_uids ==================')
        print(f'given_uids.shape = {given_uids.shape}, given_vass.shape = {given_vass.shape}, given_css.shape = {given_css.shape}')
    
    # Calculate initial target variance as the sum of outer products
    return uids, vass, css, given_uids, given_vass, given_css

@torch.no_grad()
def load_all_data(
        metadata_dir_path: str,
        arch: str, 
        num_gpus: int,
        
        given_uids_path: str = None,
        target_variance: torch.Tensor = None,
        is_vas_d: bool = False,
        batch_size: int = 100000,
        
        store_variance: bool = False,
        store_path: str = None,
    ):
    """Load embeddings, UIDs, and CLIP scores from files, filter by given UIDs and CLIP score threshold, and calculate initial target variance.
    
    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        arch (str): Kind of features for CLIP filtering.
        out_queue (mp.Queue): Output queue to send loaded data.
        num_gpus (int): Number of GPU workers.
        
        given_uids_path (str): Path to the given UIDs.
        target_variance (torch.Tensor): Target variance matrix.
        is_vas_d (bool): If True, we will return the candidate_embedding and don't return VAS.
    
        batch_size (int): Batch size for calculating target variance (just for VAS-D) on device.
        target_variance (torch.Tensor): Target variance matrix.
        
        store_variance (bool): If True, we will store the variance matrix for given UIDs.
    """
    
        
        
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    root_paths = [
        (fs, str(x.split(".parquet")[0])) for x in fs.ls(url) if ".parquet" in x
    ]

    # initializing task queues
    receive_queue = mp.Queue()
    send_queue = mp.Queue()

    for job in root_paths:
        send_queue.put(job)

    processes = []
    print("starting gpu workers")
    for worker_index in tqdm(range(num_gpus)):
        p = mp.Process(
            target=vas_filter_helper,
            kwargs=dict(
                device_index=worker_index,
                in_queue=send_queue,
                out_queue=receive_queue,
                arch=arch,
                target_variance=target_variance,
                is_vas_d=is_vas_d,
                store_variance=store_variance,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.05)

    print("processing metadata with gpu workers")
    pbar = tqdm(total=len(root_paths))

    # running in main thread
    all_uids = []
    all_embs_or_vass = []
    all_css = []
    
    def update_p(receive_queue, all_uids, all_embs_or_vass, all_css, pbar):
        # utility function to update the progress bar and store results
        uids, embs_or_vass, css = receive_queue.get(timeout=10)
        all_uids.append(uids)
        all_embs_or_vass.append(embs_or_vass)
        all_css.append(css)
        pbar.update(1)
        
        return pbar.n
                            
    
    while True:
        # keep checking for jobs finishing and update uids
        try:
            counter = update_p(receive_queue, all_uids, all_embs_or_vass, all_css, pbar)
            if debug == 1 and counter == counter_num: break
        except TimeoutError:
            if all(not p.is_alive() for p in processes):
                try:
                    update_p(receive_queue, all_uids, all_embs_or_vass, all_css, pbar)
                    
                except TimeoutError:
                    raise RuntimeError("All processes dead but nothing in queue!")
        except Empty:
            pass

        if all(not p.is_alive() for p in processes):
            # case where all workers have exited
            try:
                update_p(receive_queue, all_uids, all_embs_or_vass, all_css, pbar)
                
            except Empty:
                print("Result queue is empty and all workers have exited")
                break

    pbar.close()
    for p in processes:
        p.join(timeout=1)

    
    uids = np.concatenate(all_uids)
    
    if is_vas_d or store_variance: # embed -> torch.Tensor float32
        embs_or_vass = torch.cat(all_embs_or_vass)
    else: # vass -> np.ndarray
        embs_or_vass = np.concatenate(all_embs_or_vass).astype(np.float32)

    css = np.concatenate(all_css)
    
    print(f'uids.shape = {uids.shape}, embs_or_vass.shape = {embs_or_vass.shape}, css.shape = {css.shape}')
    
    # Filter by given UIDs, which may be the subet from other filtering methods
    if given_uids_path is not None:
        given_uids = np.load(given_uids_path)
        start = time.time()
        given_uids_mask = np.isin(uids, given_uids)
        print(f'filter by given_uids, np.isin use time = {time.time() - start}')
        
        uids = uids[given_uids_mask]
        embs_or_vass = embs_or_vass[given_uids_mask] # note that tensor can take np.ndarray as index
        css = css[given_uids_mask]
        print(f'================== after filtering by given_uids ==================')
        print(f'uids.shape = {uids.shape}, embs_or_vass.shape = {embs_or_vass.shape}, css.shape = {css.shape}')
        
        if store_variance:
            target_variance = cal_target_variance(embs_or_vass, num_gpus, batch_size)
            # store the variance matrix
            torch.save(target_variance, store_path)
            print(f'store the variance matrix to {store_path}')
            return 
            
    
    # Calculate initial target variance as the sum of outer products
    if is_vas_d:
        target_variance = cal_target_variance(embs_or_vass, num_gpus, batch_size)
        return uids, css, embs_or_vass, target_variance
    else:
        return uids, embs_or_vass, css
    
   
@torch.no_grad()
def load_uids_with_vas_filter(
    metadata_dir_path: str,
    files_path: str,
    num_gpus: int,
    arch: Union[str, None] = None,
    threshold: Union[float, None] = None,
    fraction: Union[float, None] = None,
    
    threshold_vas: Union[float, None] = None,
    fraction_vas: Union[float, None] = None,
    target_variance_name: str = 'imagenet-1k',
    
    given_uids_path: Union[np.ndarray, None] = None,
    if_add_more: int = 0,
    given_uids_index_in_ordered_uids_path: str = None
) -> np.ndarray:
    """vas filter
    
    Args:
        metadata_dir_path (str): directory where metadata is stored
        files_path (str): path to the files
        num_gpus (int): number of gpu workers, each of which processes parquet, npy pairs
        arch (Union[str, None], optional): kind of features for clip filtering. Defaults to None.
        threshold (Union[float, None], optional): threshold to apply to clip features. Defaults to None.
        fraction (Union[float, None], optional): top k fraction to apply to clip features. Defaults to None.
        num_workers (Union[int, None], optional): number of cpu works used to load metadata to compute threshold. Defaults to None.
        threshold_vas (Union[float, None], optional): threshold to apply to variance alignment scores. Defaults to None.
        fraction_vas (Union[float, None], optional): top k fraction to apply to variance alignment scores. Defaults to None.
        target_variance_name (str, optional): target variance name. Defaults to 'imagenet-1k'.
        
        given_uids_path (Union[np.ndarray, None], optional): if specified, we will use the given uids as the pool. Defaults to None.
    
    """

    # load target variance
    print("loading target variance")
    if target_variance_name == 'imagenet-1k' or target_variance_name == 'variance_imagenet_1k':
        target_path = os.path.join(files_path, 'variance', "variance_imagenet_1k.pt")
        target_variance = torch.load(target_path)
    elif target_variance_name == 'self':
        # raise NotImplementedError
        target_variance = None
    else:
        target_path = os.path.join(files_path, 'variance', f"variance_{target_variance_name}.pt")
        target_variance = torch.load(target_path)
    
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    root_paths = [
        (fs, str(x.split(".parquet")[0])) for x in fs.ls(url) if ".parquet" in x
    ]

    # initializing task queues
    receive_queue = mp.Queue()
    send_queue = mp.Queue()

    for job in root_paths:
        send_queue.put(job)

    processes = []
    print("starting gpu workers")
    for worker_index in tqdm(range(num_gpus)):
        p = mp.Process(
            target=vas_filter_helper,
            kwargs=dict(
                device_index=worker_index,
                in_queue=send_queue,
                out_queue=receive_queue,
                arch=arch,
                target_variance=target_variance,
                is_vas_d=False,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.05)

    print("processing metadata with gpu workers")
    pbar = tqdm(total=len(root_paths))

    # running in main thread
    all_uids = []
    all_vass = []
    all_css = []
    while True:
        # keep checking for jobs finishing and update uids
        try:
            uids, vass, css = receive_queue.get(timeout=10)
            all_uids.append(uids)
            all_vass.append(vass)
            all_css.append(css)
            
            pbar.update(1)
        except TimeoutError:
            if all(not p.is_alive() for p in processes):
                try:
                    uids, vass, css = receive_queue.get(timeout=1)
                    all_uids.append(uids)
                    all_vass.append(vass)
                    all_css.append(css)
                    
                    pbar.update(1)
                except TimeoutError:
                    raise RuntimeError("All processes dead but nothing in queue!")
        except Empty:
            pass

        if all(not p.is_alive() for p in processes):
            # case where all workers have exited
            try:
                uids, vass, css = receive_queue.get(timeout=1)
                all_uids.append(uids)
                all_vass.append(vass)
                all_css.append(css)
                
                pbar.update(1)
            except Empty:
                print("Result queue is empty and all workers have exited")
                break

    pbar.close()
    for p in processes:
        p.join(timeout=20)

    
    # ipdb.set_trace()
    
    uids = np.concatenate(all_uids)
    vass = np.concatenate(all_vass)
    css = np.concatenate(all_css)
    
    total_num = len(uids)
    
    print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}, css.shape = {css.shape}')
    print(f'uids[0] = {uids[0]}, vass[0] = {vass[0]}, css[0] = {css[0]}')
    print(f'min,max,mean of vass = {np.min(vass), np.max(vass), np.mean(vass)} || min,max,mean of css = {np.min(css), np.max(css), np.mean(css)}')
    
    if given_uids_path is not None:
        print(f'!!!!!!!!!!!!!!!!!!!!!!!! loading given_uids from {given_uids_path}')
        given_uids = np.load(given_uids_path)
        print(f'given_uids.shape = {given_uids.shape}, given_uids[0] = {given_uids[0]}')
        
        indices = filter_by_given_uids(given_uids, uids, given_uids_index_in_ordered_uids_path)
        uids_select = uids[indices]
        vass_select = vass[indices]
        css_select = css[indices]

        print(f'================== after filtering by given_uids ==================')
        print(f'uids_select.shape = {uids_select.shape}, vass_select.shape = {vass_select.shape}, css_select.shape = {css_select.shape}, given_uids.shape = {given_uids.shape}, indices.shape = {indices.shape}')
        print(f'uids_select[0] = {uids_select[0]}, vass_select[0] = {vass_select[0]}, css_select[0] = {css_select[0]}')
        print(f'min,max,mean of vass = {np.min(vass_select), np.max(vass_select), np.mean(vass_select)} || min,max,mean of css = {np.min(css_select), np.max(css_select), np.mean(css_select)}')
        
        
        
        if if_add_more == 1:
            print(f'add more uids from the complementary set')
            # complementary index
            C_indices = np.setdiff1d(np.arange(total_num), indices)
            uids_C = uids[C_indices]
            vass_C = vass[C_indices]
            css_C = css[C_indices]
            
            print(f'================== complementary index ==================')
            print(f'uids_C.shape = {uids_C.shape}, vass_C.shape = {vass_C.shape}, css_C.shape = {css_C.shape}')
            print(f'uids_C[0] = {uids_C[0]}, vass_C[0] = {vass_C[0]}, css_C[0] = {css_C[0]}')
            print(f'min,max,mean of vass_C = {np.min(vass_C), np.max(vass_C), np.mean(vass_C)} || min,max,mean of css_C = {np.min(css_C), np.max(css_C), np.mean(css_C)}')
    
    else:
        uids_select = uids
        vass_select = vass
        css_select = css
    
    
    
    if if_add_more == 0:
        ############# step 1: filter clip scores #############
        select_indices = filter_by_score(css_select, fraction, threshold, total_num, name='clip_score')
        uids_select = uids_select[select_indices]
        vass_select = vass_select[select_indices]
        
        ############# step 2: filter VAS #############
        select_indices_vas = filter_by_score(vass_select, fraction_vas, threshold_vas, total_num, name='vas')
        uids_select = uids_select[select_indices_vas]
        
        print(f'================== finish filtering by clip_score and vas ==================')
        final_uids = uids_select
        
    else: # add the good data (high VAS, high CS) from the complementary set
        ############# step 1: filter clip scores #############
        select_indices = filter_by_score(css_C, fraction, threshold, total_num, name='clip_score')
        uids_C = uids_C[select_indices]
        vass_C = vass_C[select_indices]
        
        ############# step 2: filter VAS #############
        select_indices_vas = filter_by_score(vass_C, fraction_vas, threshold_vas, total_num, name='vas')
        uids_C = uids_C[select_indices_vas]

        print(f'================== finish filtering by clip_score and vas on complementary indices ==================')
        final_uids = np.concatenate([uids_select, uids_C])
        
    
    print(f'shape of final_uids = {final_uids.shape}')
    
    return final_uids




######################################################## VAS-D ###############################################################

@torch.no_grad()
def cal_target_variance_worker(
    emb: torch.Tensor,
    send_queue: mp.Queue,
    result_queue: mp.Queue,
    gpu_index: int,
):  
    # sum_i f_i f_i^T, use gpu and vector multiplication to accelerate the calculation
    while True:
        try:
            start, end = send_queue.get(timeout=1)
            emb_chunk = emb[start:end].float().to(f'cuda:{gpu_index}') # has copied the emb to the gpu, and don't influence the main emb #.float()
            # target_variance = torch.einsum('ni,nj->ij', emb_chunk, emb_chunk).cpu()
            
            target_variance = (emb_chunk.T @ emb_chunk).cpu()
            
            result_queue.put(target_variance)
            # delete the emb_chunk to save the gpu memory
            
            del emb_chunk
            
            # gc.collect()
            # torch.cuda.empty_cache()
            
        except Empty:
            break
    
@torch.no_grad()   
def cal_target_variance(
    emb: torch.Tensor,
    num_gpus: int,
    batch_size: int,
    downsample_ratio: float = 0.1,
):
    """
        calculate the target variance using multiple queues to accelerate the calculation
        
        S = \sum_i^N f_i f_i^T / N, so we can calculate the sum of outer products in each queue, and then sum them up
        
    """
    # random sample
    if downsample_ratio < 1:
        random_idx = np.random.choice(emb.shape[0], int(emb.shape[0] * downsample_ratio), replace=False)
        emb_ds = emb[random_idx]
    else:
        emb_ds = emb
    
    emb_num = emb_ds.shape[0]
    emb_dim = emb_ds.shape[1]

    idx_chunks = []
    
    num_batches = math.ceil(emb_num / batch_size)
    
    for i in range(num_batches):
        start = i * batch_size
        end = min((i+1) * batch_size, emb_num)
        idx_chunks.append((start, end))
    
    result_queue = mp.Queue()
    send_queue = mp.Queue()
    
    for idx_pair in idx_chunks:
        send_queue.put(idx_pair)
    
    print(f'len(send_queue) = {send_queue.qsize()}, len(idx_chunks) = {len(idx_chunks)}, num_batches = {num_batches}, emb_num = {emb_num}')
    processes = []
    print("starting gpu workers")
    
    for worker_index in tqdm(range(num_gpus)):
        p = mp.Process(
            target=cal_target_variance_worker,
            kwargs=dict(
                emb=emb_ds,
                send_queue=send_queue,
                result_queue=result_queue,
                gpu_index=worker_index,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.05)
        
    target_variance = torch.zeros(emb_dim, emb_dim)
    
    print("processing target_variance with gpu workers")
    pbar = tqdm(total=len(idx_chunks))
    
    def update_target_variance(result_queue, target_variance, pbar):
        # utility function to update the progress bar and store results
        target_variance += result_queue.get()
        pbar.update(1)
        return pbar.n
    
    while True:
        try:
            # print(f'counter = {counter}, emb_num = {emb_num}, num_batches  = {num_batches}')
            # if pbar.n == num_batches: break
            # target_variance += result_queue.get() #timeout=timeout)
            # pbar.update(1)
            if update_target_variance(result_queue, target_variance, pbar) == num_batches: break
            
        except TimeoutError:
            if all(not p.is_alive() for p in processes):
                try:
                    if update_target_variance(result_queue, target_variance, pbar) == num_batches: break
                except TimeoutError:
                    raise RuntimeError("All processes dead but nothing in queue!")
        except Empty:
            pass

        if all(not p.is_alive() for p in processes):
            # case where all workers have exited
            try:
                if update_target_variance(result_queue, target_variance, pbar) == num_batches: break
            except Empty:
                print("Result queue is empty and all workers have exited")
                break
    
    pbar.close()
    
    
    for p in processes:
        p.join(timeout=1) #timeout=timeout + 10)
    
    
    target_variance /= emb_num
    
    print(f'target_variance.shape = {target_variance.shape}, mean, min, max = {target_variance.mean()}, {target_variance.min()}, {target_variance.max()}')
    
    return target_variance
    


@torch.no_grad()
def load_uids_with_vas_filter_v2(
    metadata_dir_path: str,
    files_path: str,
    
    num_gpus: int,
    arch: Union[str, None] = None,
    threshold: Union[float, None] = None,
    fraction: Union[float, None] = None,
    
    threshold_vas: Union[float, None] = None,
    fraction_vas: Union[float, None] = None,
    target_variance_name: str = 'imagenet-1k',
    given_uids_path: Union[str, None] = None,
    
    higher_is_better_vas: int = 1
) -> np.ndarray:
    """vas filter

    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        files_path (str): Path to the embedding files.
        num_gpus (int): Number of GPU workers.
        arch (Union[str, None], optional): Kind of features for CLIP filtering. Defaults to None.
        threshold (Union[float, None], optional): Threshold to apply to CLIP features. Defaults to None.
        fraction (Union[float, None], optional): Top k fraction to apply to CLIP features. Defaults to None.
        threshold_vas (Union[float, None], optional): Threshold to apply to variance alignment scores. Defaults to None.
        fraction_vas (Union[float, None], optional): Top k fraction to apply to variance alignment scores. Defaults to None.
        target_variance_name (str, optional): Target variance name. Defaults to 'imagenet-1k'.
        given_uids_path (Union[str, None], optional): Path to the given UIDs. Defaults to None.
    """
    
    
    # load target variance
    print("loading target variance")
    if target_variance_name == "self":
        target_variance = None
        print(f'name is self, set target variance to None')
    else:
        if target_variance_name == 'imagenet-1k' or target_variance_name == 'variance_imagenet_1k':
            target_path = os.path.join(files_path, 'variance', "variance_imagenet_1k.pt")
            target_variance = torch.load(target_path)
            print(f'load target_variance from {target_path}')
        else:
            target_path = os.path.join(files_path, 'variance', f"variance_{target_variance_name}.pt")
            target_variance = torch.load(target_path)
            print(f'load target_variance from {target_path}')
    
    
    uids, vass, css = load_all_data(metadata_dir_path, arch, num_gpus, given_uids_path, target_variance=target_variance, is_vas_d=False)
    total_num = len(uids)
    
    # first filter by clip score
    select_indices = filter_by_score(css, fraction, threshold, total_num, name='clip_score')
    uids = uids[select_indices]
    vass = vass[select_indices]
    
    print(f'================== after filtering by clip_score ==================')
    print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}')
    
    if not higher_is_better_vas: # lower is better
        print(f'!!! vass is lower is better, so we will change the sign of vass')
        vass = -vass

    # Perform VAS filtering
    select_indices_vas = filter_by_score(vass, fraction_vas, threshold_vas, total_num, name='vas')
    uids = uids[select_indices_vas]
    
    print(f'================== after filtering by vas ==================')
    print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}')
    
    return uids

@torch.no_grad()
def load_uids_with_vas_filter_v3(
    metadata_dir_path: str,
    files_path: str,
    
    num_gpus: int,
    arch: Union[str, None] = None,
    threshold: Union[float, None] = None,
    fraction: Union[float, None] = None,
    
    fraction_vas: Union[float, None] = None,
    target_variance_name: str = 'imagenet-1k',
    given_uids_path: Union[str, None] = None,
    
    higher_is_better_vas: int = 1,
    
    soft_type: float = 0.0,
) -> np.ndarray:
    """vas filter with soft filtering
    """
    
    
    # load target variance
    print("loading target variance")
    if target_variance_name == "self":
        target_variance = None
        print(f'name is self, set target variance to None')
    else:
        if target_variance_name == 'imagenet-1k' or target_variance_name == 'variance_imagenet_1k':
            target_path = os.path.join(files_path, 'variance', "variance_imagenet_1k.pt")
            target_variance = torch.load(target_path)
            print(f'load target_variance from {target_path}')
        else:
            target_path = os.path.join(files_path, 'variance', f"variance_{target_variance_name}.pt")
            target_variance = torch.load(target_path)
            print(f'load target_variance from {target_path}')
    
    
    uids, vass, css = load_all_data(metadata_dir_path, arch, num_gpus, given_uids_path, target_variance=target_variance, is_vas_d=False)
    total_num = len(uids)
    
    # first filter by clip score
    select_indices = filter_by_score(css, fraction, threshold, total_num, name='clip_score')
    uids = uids[select_indices]
    vass = vass[select_indices]
    
    print(f'================== after filtering by clip_score ==================')
    print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}')
    
    if not higher_is_better_vas: # lower is better
        print(f'!!! vass is lower is better, so we will change the sign of vass')
        vass = -vass

    # Perform VAS filtering
    select_indices_vas = soft_filter_by_score(vass, fraction_vas, total_num, name='vas', soft_type=soft_type)
    uids = uids[select_indices_vas]
    
    print(f'================== after filtering by vas ==================')
    print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}')
    
    return uids



@torch.no_grad()
def load_uids_with_vas_filter_curve(
    metadata_dir_path: str,
    files_path: str,
    
    num_gpus: int,
    arch: Union[str, None] = None,
    threshold: Union[float, None] = None,
    threshold_high: Union[float, None] = None,
    
    threshold_vas: Union[float, None] = None,
    
    fraction: Union[float, None] = None,
    
    target_variance_name: str = 'imagenet_1k',
    given_uids_path: Union[str, None] = None,
    
    higher_is_better_vas: int = 1,
    
    clipscore_exponent: float = 1.0,
) -> np.ndarray:
    """vas filter, use curve to combine VAS and CS together, and filter by small thresholds first

    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        files_path (str): Path to the embedding files.
        num_gpus (int): Number of GPU workers.
        arch (Union[str, None], optional): Kind of features for CLIP filtering. Defaults to None.
        threshold (Union[float, None], optional): Threshold to apply to CLIP features. Defaults to None.
        threshold_high (Union[float, None], optional): Upper bound to apply to CLIP features. Defaults to None.
        
        threshold_vas (Union[float, None], optional): Threshold to apply to variance alignment scores. Defaults to None.
        
        fraction (Union[float, None], optional): Top k fraction to apply to combined score
        target_variance_name (str, optional): Target variance name. Defaults to 'imagenet-1k'.
        given_uids_path (Union[str, None], optional): Path to the given UIDs. Defaults to None.
        
        higher_is_better_vas (int, optional): if 1, higher is better for VAS, if 0, lower is better. Defaults to 1.
        clipscore_exponent (float, optional): exponent for clip score in combining scores. Defaults to 1.0.
    """
    
    
    # load target variance
    print("loading target variance")
    
    target_variance_dir = os.path.join(files_path, 'variance')
    target_variance_path = os.path.join(target_variance_dir, f"variance_{target_variance_name}.pt") # VAS(target proxy), like VAS(imagenet_1k)
    if not os.path.exists(target_variance_path):
        if target_variance_name == 'imagenet_1k':
            # download the target variance
            download("variance_imagenet_1k", target_variance_dir)
        else:
            raise RuntimeError(f"target variance {target_variance_name} does not exist")
    
    target_variance = torch.load(target_variance_path)
    
    uids, vass, css = load_all_data(metadata_dir_path, arch, num_gpus, given_uids_path, target_variance=target_variance, is_vas_d=False)
    total_num = len(uids)
    
    # first filter by clip score
    print(f'================== filtering by clip_score ==================')
    select_indices = filter_by_score(css, None, threshold, total_num, name='clip_score')
    uids = uids[select_indices]
    vass = vass[select_indices]
    css = css[select_indices]
    print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}, css.shape = {css.shape}')
    
    # upper bound filter, remove the data with css > threshold_high
    if threshold_high is not None:
        print(f'================== filtering by clip_score_high ==================')

        neg_css = -css
        select_indices = filter_by_score(neg_css, None, -threshold_high, total_num, name='clip_score_high')
        uids = uids[select_indices]
        vass = vass[select_indices]
        css = css[select_indices]
        
        print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}, css.shape = {css.shape}')
    
        
    
    if not higher_is_better_vas: # lower is better
        print(f'!!! vass is lower is better, so we will change the sign of vass')
        vass = -vass

    # Perform VAS filtering
    print(f'================== filtering by vas ==================')
    select_indices_vas = filter_by_score(vass, None, threshold_vas, total_num, name='vas')
    uids = uids[select_indices_vas]
    vass = vass[select_indices_vas]
    css = css[select_indices_vas]

    print(f'uids.shape = {uids.shape}, vass.shape = {vass.shape}, css.shape = {css.shape}')
    
    ####################### applying curve score to combine VAS and CS #######################
    combined_score = vass * (css ** clipscore_exponent)
    print(f'combined_score.shape = {combined_score.shape}, min, max, mean = {combined_score.min()}, {combined_score.max()}, {combined_score.mean()}')
    
    print(f'================== filtering by combined_score, clipscore_exponent = {clipscore_exponent} ==================')
    select_indices_combined = filter_by_score(combined_score, fraction, None, total_num, name='combined_score')
    uids = uids[select_indices_combined]
    vass = vass[select_indices_combined]
    css = css[select_indices_combined]
    
    print(f'vass min, max, mean = {vass.min()}, {vass.max()}, {vass.mean()}')
    print(f'css min, max, mean = {css.min()}, {css.max()}, {css.mean()}')
    
    
    print(f'uids.shape = {uids.shape}')
    
    return uids






@torch.no_grad() 
def store_final_target_variance(variance, files_path, extra_str = ''):
    target_path = os.path.join(files_path, 'variance', f"variance{extra_str}.pt")
    torch.save(variance, target_path)
    print(f'save target_variance to {target_path}')
    
    return target_path

@torch.no_grad()
def vis_data_by_given_uids(
    metadata_dir_path: str,
    files_path: str,
    
    num_gpus: int,
    arch: Union[str, None] = None,
    target_variance_name: str = 'imagenet-1k',
    given_uids_path: Union[str, None] = None,
    
    note: str = '',
    
    fraction: Union[float, None] = None,
    threshold: Union[float, None] = None,
    fraction_vas: Union[float, None] = None,
    threshold_vas: Union[float, None] = None,
    
) -> np.ndarray:
    """vas filter

    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        files_path (str): Path to the embedding files.
        num_gpus (int): Number of GPU workers.
        arch (Union[str, None], optional): Kind of features for CLIP filtering. Defaults to None.
        threshold (Union[float, None], optional): Threshold to apply to CLIP features. Defaults to None.
        fraction (Union[float, None], optional): Top k fraction to apply to CLIP features. Defaults to None.
        threshold_vas (Union[float, None], optional): Threshold to apply to variance alignment scores. Defaults to None.
        fraction_vas (Union[float, None], optional): Top k fraction to apply to variance alignment scores. Defaults to None.
        target_variance_name (str, optional): Target variance name. Defaults to 'imagenet-1k'.
        given_uids_path (Union[str, None], optional): Path to the given UIDs. Defaults to None.
    """
    
    
    # load target variance
    print("loading target variance")
    if target_variance_name == "self" or target_variance_name == 'None':
        target_variance = None
        print(f'name is self, set target variance to None')
    else:
        if target_variance_name == 'imagenet-1k' or target_variance_name == 'variance_imagenet_1k':
            target_path = os.path.join(files_path, 'variance', "variance_imagenet_1k.pt")
            target_variance = torch.load(target_path)
            print(f'load target_variance from {target_path}')
        else:
            target_path = os.path.join(files_path, 'variance', f"variance_{target_variance_name}.pt")
            target_variance = torch.load(target_path)
            print(f'load target_variance from {target_path}')
    
    save_dir = os.path.join(files_path, 'vis')
    
    
    
    type = 0
    # type -1 -> store variance for given UIDs
    # type 0 -> visualize data using ImageCaptionVisualizer
    # type 1 -> visualize data using ScatterPlot
    
    
    if type == -1:
        store_path = os.path.join(files_path, 'variance', f"variance_{note}.pt")
        load_all_data(metadata_dir_path, arch, num_gpus, given_uids_path, target_variance=target_variance, is_vas_d=False, store_variance=True, store_path=store_path, batch_size=2000000)
        print(f'finish loading data')
        
    elif type == 0:
    
    
        uids, vass, css, captions, urls = load_all_data_vis(metadata_dir_path, arch, num_gpus, given_uids_path, target_variance=target_variance, downsample_rate=0.01)
        
        
        
        add_values = {}
        add_values['vas'] = vass
        add_values['cs'] = css
        
        ICV = ImageCaptionVisualizer(
            save_path=save_dir,
            urls = urls,
            captions = captions,
            add_values=add_values,
        )
        
        ICV.vis_datasets(
            note = note,
            key = 'cs',
            # ratio_list = [0.8, 0.2]
        )

        print(f'finish vis data')
        
    elif type == 1:
    
        uids, vass, css, given_uids, given_vass, given_css = load_all_data_vis2(metadata_dir_path, arch, num_gpus, given_uids_path, target_variance=target_variance, downsample_rate=0.001)
        
        # first filter by clip score
        print(f'================== filtering by clip_score ==================')
        select_indices = filter_by_score(css, fraction, threshold, len(uids), name='clip_score')
        vas_uids = uids[select_indices]
        vas_vass = vass[select_indices]
        vas_css = css[select_indices]

        print(f'uids.shape = {vas_uids.shape}, vass.shape = {vas_vass.shape}, css.shape = {vas_css.shape}')
        
        # Perform VAS filtering
        print(f'================== filtering by vas ==================')
        select_indices_vas = filter_by_score(vas_vass, fraction_vas, threshold_vas, len(uids), name='vas')
        vas_uids = vas_uids[select_indices_vas]
        vas_vass = vas_vass[select_indices_vas]
        vas_css = vas_css[select_indices_vas]
        
        print(f'uids.shape = {vas_uids.shape}, vass.shape = {vas_vass.shape}, css.shape = {vas_css.shape}')
        
        # group id can be list
        group_id = [-1] * len(uids)
        
        # if uid not in given_uids and not in vas_uids, group_id = 0
        # if uid in given_uids and not in vas_uids, group_id = 1
        # if uid in vas_uids and not in given_uids, group_id = 2
        # if uid in vas_uids and in given_uids, group_id = 3
        
        
        for i, uid in tqdm(enumerate(uids), desc='group_id'):
            if uid in given_uids:
                if uid in vas_uids:
                    group_id[i] = 3
                else:
                    group_id[i] = 1
            else:
                if uid in vas_uids:
                    group_id[i] = 2
                else:
                    group_id[i] = 0
        
        group_name_list = [
            'others',
            'only tmars',
            'only vas',
            'both'
        ]
        
        sums = [0, 0, 0, 0]
        for i in group_id:
            sums[i] += 1
        
        print(f'sums of group_id = {sums}, total_num = {len(uids)}, sum of sums = {sum(sums)}')
        
        SP = ScatterPlot(
            save_path=save_dir,
            Xs = css,
            Ys = vass,
            labels = group_id,
            X_name = 'clip_score',
            Y_name = 'vas',
            label_names = group_name_list,
        )
        
        
        SP.plot_scatter(
            note = note,
        )
        
        print(f'finish scatter plot')
    
    else:
        raise ValueError(f'invalid type = {type}')
    
    
    exit(0)



@torch.no_grad()
def load_uids_with_vas_d_filter(
    metadata_dir_path: str,
    files_path: str,
    
    num_gpus: int,
    arch: Union[str, None] = None,
    threshold: Union[float, None] = None,
    fraction: Union[float, None] = None,
    
    given_uids_path: Union[str, None] = None,
    num_iters: Union[int, None] = 100,
    fraction_vas: Union[float, None] = None,
    
    batch_size: Union[int, None] = 100000,
    batch_size_vass: Union[int, None] = None,
    
    update_image_feature_arch: bool = True,
    save_path: str = '',
):
    """Perform VAS-D filtering algorithm with initial filtering by given UIDs and CLIP score threshold.
    
    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        files_path (str): Path to the embedding files.
        num_gpus (int): Number of GPU workers.
        arch (Union[str, None], optional): Kind of features for CLIP filtering. Defaults to None.
        threshold (Union[float, None], optional): Threshold to apply to CLIP features. Defaults to None.
        fraction (Union[float, None], optional): Top k fraction to apply to CLIP features. Defaults to None.
        given_uids_path (Union[str, None], optional): Path to the given UIDs. Defaults to None.
        num_iters (Union[int, None], optional): Number of iterations for VAS-D. Defaults to 100.
        target_size (Union[int, None], optional): Target size of the final UIDs. Defaults to None.
        batch_size (Union[int, None], optional): Batch size for VAS-D on device.
    """
    
    print(f'================== start VAS-D filtering ==================')
    
    # Load all data and calculate initial target variance
    uids, css, embs, target_variance = load_all_data(metadata_dir_path, arch, num_gpus, given_uids_path, is_vas_d=True, batch_size=batch_size)
    target_size = int(fraction_vas * len(uids))
    
    # first filter by clip score
    if fraction == 1.0:
        pass
    else:
        print(f' ================== before filtering by clip_score ==================')
        select_indices = filter_by_score(css, fraction, threshold, len(uids), name='clip_score')
        uids = uids[select_indices]
        embs = embs[select_indices]

        print(f'================== after filtering by clip_score ==================')
    
    print(f'uids.shape = {uids.shape}, embs.shape = {embs.shape}, target_size = {target_size}')
    
    
    # Perform VAS-D filtering
    chunk_size = math.ceil((len(uids) - target_size) / num_iters)
    
    for i in range(num_iters):
        total_size = len(uids)
        
        print(f'======================================= iter {i}, total_size = {total_size}, batch_size = {batch_size}, chunk_size = {chunk_size}, target_size = {target_size}, batch_size_vass = {batch_size_vass}')
        
        vass = cal_vass_iter(embs, num_gpus, target_variance, batch_size_vass, total_size)
        
        if i == num_iters - 1:
            topk = target_size
        else:
            topk = total_size - chunk_size
            
        _, indices = torch.topk(vass, topk)
        uids = uids[indices]
        embs = embs[indices]
        
        current_len = len(uids)
        print(f'current_len = {current_len}')
        
        
        # save check point
        if i % (num_iters // 5) == 0 and i > 0:
            # save uids
            uids_copy = copy.deepcopy(uids)
            uids_copy.sort()
            
            
            true_save_path = save_path.replace('.npy', f'_middle_d_{current_len}.npy') # will not change the original save_path
        
            # store uids_copy
            print(f'saving uids_copy to {true_save_path} with length = {len(uids_copy)}')
            np.save(true_save_path, uids_copy)
        
        
        target_variance = cal_target_variance(embs, num_gpus, batch_size)
    
    fvas_str = f'fvas{fraction_vas}' if fraction_vas is not None else 'tvas{threshold_vas}'
    store_final_target_variance(target_variance, files_path, extra_str = f'_vas_d_{fvas_str}_{num_iters}')
    
    return uids



@torch.no_grad()
def load_uids_with_vas_d_filter_v2(
    metadata_dir_path: str,
    files_path: str,
    
    num_gpus: int,
    arch: Union[str, None] = None,
    threshold: Union[float, None] = None,
    fraction: Union[float, None] = None,
    
    given_uids_path: Union[str, None] = None,
    num_iters: Union[int, None] = 100,
    fraction_vas: Union[float, None] = None,
    
    batch_size: Union[int, None] = 100000,
    batch_size_vass: Union[int, None] = None,
    
):
    """Perform VAS-D filtering algorithm with initial filtering by given UIDs and CLIP score threshold.
    
    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        files_path (str): Path to the embedding files.
        num_gpus (int): Number of GPU workers.
        arch (Union[str, None], optional): Kind of features for CLIP filtering. Defaults to None.
        threshold (Union[float, None], optional): Threshold to apply to CLIP features. Defaults to None.
        fraction (Union[float, None], optional): Top k fraction to apply to CLIP features. Defaults to None.
        given_uids_path (Union[str, None], optional): Path to the given UIDs. Defaults to None.
        num_iters (Union[int, None], optional): Number of iterations for VAS-D. Defaults to 100.
        target_size (Union[int, None], optional): Target size of the final UIDs. Defaults to None.
        batch_size (Union[int, None], optional): Batch size for VAS-D on device.
    """
    
    
    
    # Load all data and calculate initial target variance
    uids, css, embs, target_variance = load_all_data(metadata_dir_path, arch, num_gpus, given_uids_path, is_vas_d=True, batch_size=batch_size)
    target_size = int(fraction_vas * len(uids))
    total_num = len(uids)
    
    # first filter by clip score

    print(f'================== after filtering by clip_score ==================')
    print(f'uids.shape = {uids.shape}, embs.shape = {embs.shape}, target_size = {target_size}')
    
    
    # Perform VAS-D filtering
    chunk_size = math.ceil((len(uids) - target_size) / num_iters)
    
    for i in range(num_iters):
        total_size = len(uids)
        
        print(f'======================================= iter {i}, total_size = {total_size}, batch_size = {batch_size}, chunk_size = {chunk_size}, target_size = {target_size}, batch_size_vass = {batch_size_vass}')
        
        vass = cal_vass_iter(embs, num_gpus, target_variance, batch_size_vass, total_size)
        
        if i == num_iters - 1:
            topk = target_size
        else:
            topk = total_size - chunk_size
            
        _, indices = torch.topk(vass, topk)
        uids = uids[indices]
        embs = embs[indices]
        css = css[indices]
        
        target_variance = cal_target_variance(embs, num_gpus, batch_size)
    
    # then filter by VAS
    print(f'================== after filtering by VAS ==================')
    select_indices = filter_by_score(css, fraction, threshold, total_num, name='clip_score')
    uids = uids[select_indices]
    embs = embs[select_indices]
    
    
    fvas_str = f'fvas{fraction_vas}' if fraction_vas is not None else 'tvas{threshold_vas}'
    store_final_target_variance(target_variance, files_path, extra_str = f'_vas_d_v2_{fvas_str}_{num_iters}')
    
    return uids





@torch.no_grad()
def cal_vass_iter(
    embs: torch.Tensor,
    num_gpus: int,
    target_variance: torch.Tensor,
    batch_size_vass: int,
    total_size: int,

):
    """
        calculate the VAS-D using multiple queues to accelerate the calculation
        
    """
    data_queue = mp.Queue()
        
    if batch_size_vass is not None:        
        for j in range(0, total_size, batch_size_vass):
            data_queue.put((j, j+batch_size_vass))
    else: # process in one batch
        data_queue.put((0, total_size))
    
    result_queue = mp.Queue()
    processes = []
    
    target_variance_list = []
    
    # just define the new target_variance on each device
    for worker_index in range(num_gpus):
        target_variance_list.append(target_variance.clone().to(f'cuda:{worker_index}'))
    
    print("starting gpu workers for VAS-D")
    for worker_index in tqdm(range(num_gpus)):
        p = mp.Process(
            target=vas_d_worker,
            kwargs=dict(
                emb = embs,
                in_queue=data_queue,
                out_queue=result_queue,
                device_id=worker_index,
                target_variance_list=target_variance_list,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.05)

    vass = torch.empty(total_size)
    
    print("processing VAS-D with gpu workers")
    
    # len of data_queue
    if batch_size_vass is None:
        total_pbar_size = 1
    else:
        total_pbar_size = math.ceil(total_size / batch_size_vass)
    
    pbar = tqdm(total=total_pbar_size)
    
    
    def update_vas_d(result_queue, vass, pbar):
        # utility function to update the progress bar and store results
        vas, start, end = result_queue.get() # timeout=20)
        vass[start:end] = vas
        pbar.update(0.1)
        # return the value of pbar
        return pbar.n
    
    while True:
        try:
            if update_vas_d(result_queue, vass, pbar) == total_pbar_size: break
        except TimeoutError:
            if all(not p.is_alive() for p in processes):
                try:
                    if update_vas_d(result_queue, vass, pbar) == total_pbar_size: break
                except TimeoutError:
                    raise RuntimeError("All processes dead but nothing in queue!")
        except Empty:
            pass

        if all(not p.is_alive() for p in processes):
            try:
                if update_vas_d(result_queue, vass, pbar) == total_pbar_size: break
            except Empty:
                print("Result queue is empty and all workers have exited")
                break
    
    
    pbar.close()
    for p in processes:
        p.join(0.1)
    
    print(f'mean, min, max of VAS = {vass.mean()}, {vass.min()}, {vass.max()}')
    
    return vass



@torch.no_grad()
def vas_d_worker(emb, in_queue, out_queue, device_id, target_variance_list):
    """Worker process to calculate VAS for a chunk of embeddings.
    
    Args:
        in_queue (mp.Queue): Input queue to get embedding chunk.
        out_queue (mp.Queue): Output queue to send calculated VAS.
        device_id (int): GPU device index.
        target_variance (torch.Tensor): Target variance matrix.
    """
    while True:
        try:
            start, end = in_queue.get(timeout=1)
            emb_chunk = emb[start:end].float().to(f'cuda:{device_id}') # more efficient to copy the emb to the gpu, don't leave emb_chunk results on the cpu #.float()
            
            target_variance2 = target_variance_list[device_id]
            
            # vas = torch.einsum('ni,ij,nj->n', emb_chunk, target_variance2, emb_chunk) # calculate with @, which is more efficient 
            vas = torch.sum((emb_chunk @ target_variance2) * emb_chunk, dim=1).cpu()
            
            out_queue.put((vas, start, end))
            
            # delete the emb_chunk to save the gpu memory
            del emb_chunk
            
            # gc.collect() # no for the gpu memory, just for the cpu memory
            # torch.cuda.empty_cache() # not needed frequently, delete the pointer to the emb_chunk will allow the automatic memory release
            
        except Empty:
            break
        




@torch.no_grad()
def cal_similarity_scores_norm(
    embs: torch.Tensor,
    proxy_embeddings: torch.Tensor,
    device_index: int,
    norm: int = 2,
    
    batch_size: int = None,
    
):
    """
    For proxy_embeddings G (N x D), and candidate embeddings F (M x D), calculate the similarity scores F @ G^T (M x N), then get the p norm of each row (M x 1)
    
    Note that the batch size is applying on the candidate embeddings F

    """
    device_str = f'cuda:{device_index}'
    proxy_embeddings_gpu = proxy_embeddings.float().to(device_str)
    if batch_size is None or batch_size >= embs.shape[0]:
        batch_size = embs.shape[0]
    
    num_batches = math.ceil(embs.shape[0] / batch_size)
    # print(f'norm = {norm}, batch_size = {batch_size}, num_batches = {num_batches}') # correct

    if norm >= 100:
        # infinity norm
        norm = float('inf')
    
    # for i in tqdm(range(num_batches), desc=f'cal_sim_score on {device_index}'):
    for i in range(num_batches):
        start = i * batch_size
        end = min((i+1) * batch_size, embs.shape[0])
        embs_chunk = embs[start:end].float().to(device_str)
        
        # print(f'embs_chunk.shape = {embs_chunk.shape}, proxy_embeddings_gpu.shape = {proxy_embeddings_gpu.shape}, i = {i}, num_batches = {num_batches}') # embs_chunk.shape = torch.Size([505891, 768])
        
        similarity_scores_gpu = embs_chunk @ proxy_embeddings_gpu.T
        
        del embs_chunk
        
        similarity_scores = torch.norm(similarity_scores_gpu, p=norm, dim=1).cpu()
        
        del similarity_scores_gpu

        if i == 0:
            scores = similarity_scores
        else:
            scores = torch.cat((scores, similarity_scores))
        
            
    return scores
        
@torch.no_grad()
def cal_similarity_scores_inf_column(
    embs: torch.Tensor,
    proxy_embeddings: torch.Tensor,
    device_index: int,
    norm: int = 2,
    
    batch_size: int = None,
    k: int = 200, # for top k selection, 100 can cover 34/50 for an = 5
    default_rank: int = 100000,
    average_num: int = 5,
 ):
     
    """
    For proxy_embeddings G (N x D), and candidate embeddings F (M x D), calculate the similarity scores F @ G^T (M x N), then get the rank based on each column (for every proxy) (M x N). The rank: smaller the similarity, larger the rank
    Then keep a Mx1 tensor, which is the minimum rank for each candidate embedding, finally return negative of this tensor
    
    Note that the batch size is applying on the proxy embeddings G
    """
    print(f'!!!embs.shape = {embs.shape}, proxy_embeddings.shape = {proxy_embeddings.shape}, in inf column!')
    device_str = f'cuda:{device_index}'
    embs_gpu = embs.float().to(device_str)
    
    proxy_embeddings_list = list(torch.chunk(proxy_embeddings, average_num, dim=0))
    
    
    for ip, proxy_embeddings_avg in enumerate(proxy_embeddings_list):
    
        if batch_size is None or batch_size >= proxy_embeddings_avg.shape[0]:
            batch_size = proxy_embeddings_avg.shape[0]
            
        num_batches = math.ceil(proxy_embeddings_avg.shape[0] / batch_size)
        print(f'norm = {norm}, batch_size = {batch_size}, num_batches = {num_batches}') # correct
        
        # 
        
        for i in range(num_batches):
            start = i * batch_size
            end = min((i+1) * batch_size, proxy_embeddings_avg.shape[0])
            proxy_embeddings_chunk = proxy_embeddings_avg[start:end].float().to(device_str)
            
            similarity_scores_cpu = (embs_gpu @ proxy_embeddings_chunk.T) #.to('cpu')
            
            del proxy_embeddings_chunk
            
            # get the rank of each column
            # sorted_indices = torch.argsort(similarity_scores_cpu, dim=0, descending=True)
            # tmp_ranks = torch.zeros_like(sorted_indices)
            # col_indices = torch.arange(similarity_scores_cpu.size(0)).unsqueeze(1).repeat(1, similarity_scores_cpu.size(1))
            # tmp_ranks.scatter_(0, sorted_indices, col_indices)
            
            _, indices = torch.topk(similarity_scores_cpu, k=k, dim=0, largest=True)
            tmp_ranks = torch.full_like(similarity_scores_cpu, default_rank, dtype=torch.int, device = device_str)
            topk_ranks = torch.arange(0, k, dtype=torch.int, device = device_str).view(k, 1).expand(k, indices.size(1))
            tmp_ranks.scatter_(0, indices, topk_ranks)
            
            best_ranks = tmp_ranks.min(dim=1).values
            del similarity_scores_cpu
            
            
            if i == 0:
                min_ranks = copy.deepcopy(best_ranks)
            else:
                min_ranks = torch.min(min_ranks, best_ranks)
            
            if i % 10 == 0:
                print(f'sum(rank=0) : {torch.sum(min_ranks == 0).item()}, sum(rank < k): {torch.sum(min_ranks < k).item()}, {device_str}, {embs.shape[0]}, {i}:{num_batches} | {ip}')    
            
        torch.cuda.empty_cache()
        
        if ip == 0:
            result_ranks = -min_ranks.cpu()
        else:
            result_ranks += -min_ranks.cpu()
        
        final_results = result_ranks / (ip + 1)
        
        print(f'======== sum(rank=0) : {torch.sum(-final_results == 0).item()}, sum(rank < k): {torch.sum(final_results > -k).item()}, {device_str}, {embs.shape[0]}, {ip} =========')   
    
    return result_ranks / average_num
        
 
@torch.no_grad()
def cal_exp_as(
    embs: torch.Tensor,
    proxy_embeddings: torch.Tensor,
    device_index: int,
    norm: int = 2,
    
    batch_size: int = None,
    batch_size_candidate: int = 32768,
    batch_size_small: int = 2000,
    temperature: float = 0.01,
):  
    """
    For proxy_embeddings G (N x D), and candidate embeddings F (M x D), calculate the similarity scores J = F @ G^T (M x N), then 
    1. Exp normalize for each candidate: Jn = exp(J/tem) / sum(exp(J/tem), dim=0).reshape(1, N) -> (M x N)
    2. Calculate the p-norm of the normalized scores for each candidate: Jp = ||J_n||_p -> (M x 1)
    
    
    Note that the batch size is applying on the proxy embeddings G, and the batch_size_candidate is applying on the candidate embeddings F

    """
    if norm >= 100:
        # infinity norm
        norm = float('inf')
        
    device_str = f'cuda:{device_index}'
    
    if batch_size is None or batch_size >= proxy_embeddings.shape[0]:
        batch_size = proxy_embeddings.shape[0]
    
    if batch_size_candidate is None or batch_size_candidate >= embs.shape[0]:
        batch_size_candidate = embs.shape[0]
        
    num_batches_proxy = math.ceil(proxy_embeddings.shape[0] / batch_size)
    num_batches_candidate = math.ceil(embs.shape[0] / batch_size_candidate)
    
    print(f'norm = {norm}, batch_size = {batch_size}, num_batches_proxy = {num_batches_proxy}, batch_size_candidate = {batch_size_candidate}, num_batches_candidate = {num_batches_candidate}') 
    
    # proxy embedding can get chunk by order, but candidate embedding need to be shuffled. p norm should be apply to the whole proxy embedding
    random_indices_candidate = torch.randperm(embs.shape[0])
    
    final_scores = torch.zeros(embs.shape[0])
    
    # for i in range(num_batches_candidate):
        
        
    #     start_candidate = i * batch_size_candidate
    #     end_candidate = min((i+1) * batch_size_candidate, embs.shape[0])
    #     embs_chunk = embs[random_indices_candidate[start_candidate:end_candidate]].float().to(device_str)
        
    #     # store 
        
    #     for j in range(num_batches_proxy):
    #         start_proxy = j * batch_size
    #         end_proxy = min((j+1) * batch_size, proxy_embeddings.shape[0])
    #         proxy_embeddings_chunk = proxy_embeddings[start_proxy:end_proxy].float().to(device_str)
            
    #         J = embs_chunk @ proxy_embeddings_chunk.T
            
    #         del proxy_embeddings_chunk
            
    #         exp_sim = torch.exp(J / temperature)
    #         Jn = exp_sim / torch.sum(exp_sim, dim=0).reshape(1, -1)
            
        
    #         if j == 0:
    #             emb_Jn = Jn.cpu()
    #         else: # stack over all the proxy embeddings, and then do the p norm
    #             emb_Jn = torch.cat((emb_Jn, Jn.cpu()), dim = 1)
            
    #         del J, exp_sim, Jn
            
    #         print(f'{i}, {j}, emb_Jn.shape = {emb_Jn.shape}')
    #         # clear the gpu memory
    #         torch.cuda.empty_cache()
        
                
    #     # calculate the p norm
    #     Jp = torch.norm(emb_Jn.to(device_str), p=norm, dim=1).cpu()
        
    #     del emb_Jn
    
    
    # due to the limit of memory, we should do it by two steps, first calculate the denominator, then calculate the p norm
    
    proxy_embeddings_gpu = proxy_embeddings.float().to(device_str)
    
    # print('proxy embedding cost gpu memory G : ', proxy_embeddings_gpu.element_size() * proxy_embeddings_gpu.nelement() / 1024 / 1024 / 1024, 'GB') # 4G
    
    for i in range(num_batches_candidate):
        
        
        start_candidate = i * batch_size_candidate
        end_candidate = min((i+1) * batch_size_candidate, embs.shape[0])
        embs_chunk = embs[random_indices_candidate[start_candidate:end_candidate]].float().to(device_str)
        
        ### first step, calculate the denominator
        denominators = torch.zeros(proxy_embeddings.shape[0]).to(device_str)
        
        for j in range(num_batches_proxy):
            start_proxy = j * batch_size
            end_proxy = min((j+1) * batch_size, proxy_embeddings.shape[0])
            proxy_embeddings_chunk = proxy_embeddings_gpu[start_proxy:end_proxy]
            
            J = embs_chunk @ proxy_embeddings_chunk.T
            
            
            denominator = torch.sum(torch.exp(J / temperature), dim=0)
            
            denominators[start_proxy:end_proxy] = denominator
                
            del J
        
        # # clear the gpu memory
        # torch.cuda.empty_cache()
            
        print(f'{i}, denominators.shape = {denominators.shape}')
        
        ### second step, recalculate the similarity scores and the p norm    
        
        sub_num_batches_candidate = math.ceil((end_candidate - start_candidate) / batch_size_small)
        
        # print(f'{i}, sub_num_batches_candidate = {sub_num_batches_candidate}, start_candidate = {start_candidate}, end_candidate = {end_candidate}')
        for ii in range(sub_num_batches_candidate):
            start_candidate_small = ii * batch_size_small
            end_candidate_small = min((ii+1) * batch_size_small, end_candidate - start_candidate)
            
            embs_chunk_small = embs_chunk[start_candidate_small:end_candidate_small]
            
            # here we don't need to split the proxy embeddings, just use the whole proxy embeddings
            Jn = torch.exp(embs_chunk_small @ proxy_embeddings_gpu.T / temperature) / denominators
            # print(f'embs_chunk: {embs_chunk.shape}, embs_chunk_small: {embs_chunk_small.shape}, proxy_embeddings_gpu: {proxy_embeddings_gpu.shape}, denominators: {denominators.shape}, Jn.shape = {Jn.shape}')
            
            Jp = torch.norm(Jn, p=norm, dim=1).cpu()
            
            # print(f'{start_candidate_small}, {end_candidate_small}, len(idx) = {len(random_indices_candidate[start_candidate_small:end_candidate_small])}, Jp.shape = {Jp.shape}')
            final_scores[random_indices_candidate[start_candidate_small:end_candidate_small]] = Jp
            
            del Jn
            
        print(f'{i}, Jp.shape = {Jp.shape}')
        
        # clear the gpu memory
        torch.cuda.empty_cache()
        
    return final_scores


@torch.no_grad()
def vas_filter_variant_helper(
    device_index: int,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    
    arch: Union[str, None] = None,
    
    proxy_embeddings: torch.Tensor = None,
    norm: int = 2,
    batch_size: int = None,
) -> None:
    """worker function to variance alignment score with different norm given all the proxy embeddings
    
    Args:
        device_index (int): device on which to run the gpu processing
        in_queue (mp.Queue): task queue with fsspec, metadata path pairs
        out_queue (mp.Queue): output queue to send filtred uids
        arch: (Union[str, None]): If specified, we want to apply a threshold to arch=B/32 or L/14 clip scores. Defaults to None.
        proxy_embeddings (torch.Tensor): proxy embeddings for VAS-D. Defaults to None.    
    """
    
    while True:
        fs_root = None
        try:
            fs_root = in_queue.get(timeout=1)
        except Empty:
            # case where the queue is depleated, worker should return
            break
        
        fs, path_root = fs_root
        
        feature_name = "l14_img"
        if arch is not None:
            key = key_name
            if arch == "b32":
                key = "clip_b32_similarity_score"
                feature_name = "b32_img"
            
            df = pd.read_parquet(
                f"{path_root}.parquet", columns=["uid", text_name, key], filesystem=fs
            )
        else:
            df = pd.read_parquet(
                f"{path_root}.parquet", columns=["uid", text_name], filesystem=fs
            )
        # print(f'feature_name = {feature_name}, key = {key}, arch = {arch}')

        candidate_embedding = None
        with fs.open(f"{path_root}.npz") as f: # -> float32
            candidate_embedding = torch.from_numpy(np.load(f)[feature_name])#.float()

        uids = df["uid"].values
        
        uids_standard = np.array(
                            [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids],
                            np.dtype("u8,u8"),
                        )
        
        # clip scores
        if arch is not None:
            css = df[key].values
        else:
            css = None
        
        scores = cal_similarity_scores_norm(
            candidate_embedding,
            proxy_embeddings,
            device_index,
            norm=norm,
            batch_size=batch_size,
        )
        
        out_queue.put(
            (
                uids_standard,
                scores.numpy(),
                css,
            )
        )      
            
            
@torch.no_grad()
def load_all_data_variant(
        metadata_dir_path: str,
        arch: str, 
        num_gpus: int,
        
        given_uids_path: str = None,
        
        proxy_embeddings: torch.Tensor = None,
        batch_size: int = 100000,
        norm: int = 2,
    ):
    """Load embeddings, UIDs, and CLIP scores from files, filter by given UIDs and CLIP score threshold, and calculate initial target variance.
    
    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        arch (str): Kind of features for CLIP filtering.
        out_queue (mp.Queue): Output queue to send loaded data.
        num_gpus (int): Number of GPU workers.
        
        given_uids_path (str): Path to the given UIDs.
        target_variance (torch.Tensor): Target variance matrix.
        is_vas_d (bool): If True, we will return the candidate_embedding and don't return VAS.
    
        batch_size (int): Batch size for calculating target variance (just for VAS-D) on device.
        target_variance (torch.Tensor): Target variance matrix.
        
        store_variance (bool): If True, we will store the variance matrix for given UIDs.
    """
    
    print(f'begin to load data variant, arch = {arch}')
        
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    root_paths = [
        (fs, str(x.split(".parquet")[0])) for x in fs.ls(url) if ".parquet" in x
    ]

    # initializing task queues
    receive_queue = mp.Queue()
    send_queue = mp.Queue()

    for job in root_paths:
        send_queue.put(job)

    processes = []
    print("starting gpu workers")
    for worker_index in tqdm(range(num_gpus)):
        p = mp.Process(
            target=vas_filter_variant_helper,
            kwargs=dict(
                device_index=worker_index,
                in_queue=send_queue,
                out_queue=receive_queue,
                arch=arch,
                
                proxy_embeddings=proxy_embeddings,
                norm=norm,
                batch_size=batch_size,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.05)

    print("processing metadata with gpu workers")
    pbar = tqdm(total=len(root_paths))

    # running in main thread
    all_uids = []
    all_scores = []
    all_css = []
    
    def update_p(receive_queue, all_uids, all_scores, all_css, pbar):
        # utility function to update the progress bar and store results
        uids, scores, css = receive_queue.get(timeout=10)
        all_uids.append(uids)
        all_scores.append(scores)
        all_css.append(css)
        pbar.update(1)
        
        return pbar.n
                                
    
    while True:
        # keep checking for jobs finishing and update uids
        try:
            counter = update_p(receive_queue, all_uids, all_scores, all_css, pbar)
            if debug == 1 and counter == counter_num: break
        except TimeoutError:
            if all(not p.is_alive() for p in processes):
                try:
                    update_p(receive_queue, all_uids, all_scores, all_css, pbar)
                    
                except TimeoutError:
                    raise RuntimeError("All processes dead but nothing in queue!")
        except Empty:
            pass

        if all(not p.is_alive() for p in processes):
            # case where all workers have exited
            try:
                update_p(receive_queue, all_uids, all_scores, all_css, pbar)
                
            except Empty:
                print("Result queue is empty and all workers have exited")
                break

    pbar.close()
    for p in processes:
        p.join(timeout=1)

    
    uids = np.concatenate(all_uids)
    
    scores = np.concatenate(all_scores).astype(np.float32)

    css = np.concatenate(all_css)
    
    print(f'uids.shape = {uids.shape}, scores.shape = {scores.shape}, css.shape = {css.shape}')
    
    # Filter by given UIDs, which may be the subet from other filtering methods
    if given_uids_path is not None:
        given_uids = np.load(given_uids_path)
        start = time.time()
        given_uids_mask = np.isin(uids, given_uids)
        print(f'filter by given_uids, np.isin use time = {time.time() - start}')
        
        uids = uids[given_uids_mask]
        scores = scores[given_uids_mask] # note that tensor can take np.ndarray as index
        css = css[given_uids_mask]
        print(f'================== after filtering by given_uids ==================')
        print(f'uids.shape = {uids.shape}, scores.shape = {scores.shape}, css.shape = {css.shape}')
        
            
    
    # Calculate initial target variance as the sum of outer products
    return uids, scores, css


@torch.no_grad()
def get_proxy_embeddings(
    proxy_name: str,
    proxy_path: str,
    cache_path: str,
    
    if_permutate: int = 1 # if permutate the rows of proxy embeddings
):
    # load proxy embeddings
    proxy_embeddings1 = None
    proxy_embeddings2 = None
    
    if 'imagenet_1k' in proxy_name:
        proxy_embeddings1 = torch.cat(
            [
                torch.load(download(f"in1k_clip_vit_l14_{i}", os.path.join(cache_path, 'image')))["image_features"]
                for i in tqdm(range(5))
            ],
            dim=0,
        )
        
    if 'validation' in proxy_name:
        # for all the embeddings in the validation directory, load them and concatenate them
        file_num = 0
        for file in os.listdir(proxy_path):
            if file.endswith(".pt"):
                if file_num == 0:
                    proxy_embeddings2 = torch.load(os.path.join(proxy_path, file))
                    print(f'{file_num}: load proxy_embeddings from {file}, shape = {proxy_embeddings2.shape}')
                else:
                    tmp_proxy_embeddings = torch.load(os.path.join(proxy_path, file))
                    print(f'{file_num}: load proxy_embeddings from {file}, shape = {tmp_proxy_embeddings.shape}')
                    proxy_embeddings2 = torch.cat((proxy_embeddings2, tmp_proxy_embeddings), dim=0)
                
                file_num += 1
                
    # combine the proxy embeddings
    if proxy_embeddings1 is not None and proxy_embeddings2 is not None:
        proxy_embeddings = torch.cat((proxy_embeddings1, proxy_embeddings2), dim=0)
        print(f'{proxy_name}: combine both imagenet_1k and validation proxy embeddings, proxy_embeddings.shape = {proxy_embeddings.shape}!')
    elif proxy_embeddings1 is not None:
        proxy_embeddings = proxy_embeddings1
        print(f'{proxy_name}: use imagenet_1k proxy embeddings, proxy_embeddings.shape = {proxy_embeddings.shape}!')
    elif proxy_embeddings2 is not None:
        proxy_embeddings = proxy_embeddings2
        print(f'{proxy_name}: use validation proxy embeddings, proxy_embeddings.shape = {proxy_embeddings.shape}!')
    else:
        raise ValueError(f'proxy_embeddings is None')
    
    if if_permutate:
        print(f'******** permutate proxy with {proxy_embeddings.size(0)}')
        
        return proxy_embeddings[torch.randperm(proxy_embeddings.size(0))]
    
    return proxy_embeddings
    
    


@torch.no_grad()
def load_uids_with_vas_filter_variant(
    metadata_dir_path: str,
    cache_path: str,
    
    num_gpus: int,
    arch: Union[str, None] = None,
    threshold: Union[float, None] = None,
    fraction: Union[float, None] = None,
    
    threshold_vas: Union[float, None] = None,
    fraction_vas: Union[float, None] = None,

    given_uids_path: Union[str, None] = None,
    
    higher_is_better_vas: int = 1,
    
    norm: int = 2,
    batch_size: int = None,
    
    proxy_name: str = 'imagenet_1k',
    proxy_path: str = None,
) -> np.ndarray:
    """vas filter

    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        cache_path (str): Path to the cache directory.
        
        num_gpus (int): Number of GPU workers.
        arch (Union[str, None], optional): Kind of features for CLIP filtering. Defaults to None.
        threshold (Union[float, None], optional): Threshold to apply to CLIP features. Defaults to None.
        fraction (Union[float, None], optional): Top k fraction to apply to CLIP features. Defaults to None.
        threshold_vas (Union[float, None], optional): Threshold to apply to variance alignment scores. Defaults to None.
        fraction_vas (Union[float, None], optional): Top k fraction to apply to variance alignment scores. Defaults to None.
        target_variance_name (str, optional): Target variance name. Defaults to 'imagenet-1k'.
        given_uids_path (Union[str, None], optional): Path to the given UIDs. Defaults to None.
    """
    
    
    
    # load proxy embeddings
    proxy_embeddings = get_proxy_embeddings(proxy_name, proxy_path, cache_path)
    
    uids, scores, css = load_all_data_variant(metadata_dir_path, arch, num_gpus, given_uids_path, proxy_embeddings=proxy_embeddings, batch_size=batch_size, norm=norm)
    
    
    total_num = len(uids)
    
    # first filter by clip score
    select_indices = filter_by_score(css, fraction, threshold, total_num, name='clip_score')
    uids = uids[select_indices]
    scores = scores[select_indices]
    
    print(f'================== after filtering by clip_score ==================')
    print(f'uids.shape = {uids.shape}, scores.shape = {scores.shape}')
    
    if not higher_is_better_vas: # lower is better
        print(f'!!! scores is lower is better, so we will change the sign of scores')
        scores = -scores

    # Perform VAS filtering
    select_indices_vas = filter_by_score(scores, fraction_vas, threshold_vas, total_num, name='vas')
    uids = uids[select_indices_vas]
    
    print(f'================== after filtering by vas ==================')
    print(f'uids.shape = {uids.shape}, scores.shape = {scores.shape}')
    
    return uids




def cal_new_clip_score(
    image_embedding: torch.Tensor,
    text_embedding: torch.Tensor,
    device_index: int,
    batch_size: int = None,
    score_type: int = 0,
    temperature: float = 0.01,
):
    # calculate the new version clip score, which consider batch of data for normalization
    # consider f as image_embedding, g as text_embedding
    # I1 = log[ exp(f_i^T g_i / T) / \sum_j exp(f_i^T g_j/T) ], I2 = log[ exp(f_i^T g_i/T) / \sum_j exp(f_j^T g_i/T) ], normalization contain data in the batch
    # they type 0: I = 0.5 * (I1 + I2), type 1: I = I1, type 2: I = I2
    # but here we return new clip score, sum1 and sum2
    
    device_str = f'cuda:{device_index}'
    img_emb_gpu = image_embedding.float().to(device_str)
    txt_emb_gpu = text_embedding.float().to(device_str)
    
    # print(f'img_emb_gpu.shape = {img_emb_gpu.shape}, txt_emb_gpu.shape = {txt_emb_gpu.shape}, score_type = {score_type}, temperature = {temperature}, batch_size = {batch_size}')
    # img_emb_gpu.shape = torch.Size([109662, 768]), txt_emb_gpu.shape = torch.Size([109662, 768]), score_type = 0, temperature = 0.07, batch_size = 32768
    
    if batch_size is None or batch_size >= img_emb_gpu.shape[0]:
        batch_size = img_emb_gpu.shape[0]
        
    num_batches = math.ceil(img_emb_gpu.shape[0] / batch_size)
    
    # random shuffle index
    indices = torch.randperm(img_emb_gpu.shape[0])
    
    # init 
    final_css = torch.zeros(img_emb_gpu.shape[0])
    final_norm1 = torch.zeros(img_emb_gpu.shape[0])
    final_norm2 = torch.zeros(img_emb_gpu.shape[0])
    
    for i in range(num_batches):
        start = i * batch_size
        end = min((i+1) * batch_size, img_emb_gpu.shape[0])
        batch_idx = indices[start:end]
        
        img_emb_chunk = img_emb_gpu[batch_idx]
        txt_emb_chunk = txt_emb_gpu[batch_idx]
        
        # calculate the similarity scores, the diagonal is the clip score of the same image and text
        sim_scores = img_emb_chunk @ txt_emb_chunk.T
        exp_sim_scores = torch.exp(sim_scores / temperature)
        
        clip_scores = torch.diag(sim_scores)
        
        # calculate the normalization, different normalization means different dimension for sum
        sum_1 = torch.sum(exp_sim_scores, dim=1) # sum over text
        sum_2 = torch.sum(exp_sim_scores, dim=0) # sum over image
        
        # diag_exp_sim_scores = torch.diag(exp_sim_scores)
        # calculate the new clip score
        # if score_type == 0:
        #     new_scores[batch_idx] = 0.5 * (torch.log(diag_exp_sim_scores / sum_1) + torch.log(diag_exp_sim_scores / sum_2)).cpu()
        # elif score_type == 1:
        #     new_scores[batch_idx] = torch.log(diag_exp_sim_scores / sum_1).cpu()
        # elif score_type == 2:
        #     new_scores[batch_idx] = torch.log(diag_exp_sim_scores / sum_2).cpu()
        # else:
        #     raise ValueError(f'invalid score_type = {score_type}')
        
        final_css[batch_idx] = clip_scores.cpu()
        final_norm1[batch_idx] = 0.5 * temperature * torch.log(sum_1).cpu()
        final_norm2[batch_idx] = 0.5 * temperature * torch.log(sum_2).cpu()
        
        
    return final_css, final_norm1, final_norm2
        
        
        
        


@torch.no_grad()
def cs_new_helper(
    device_index: int,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    arch: Union[str, None] = None,
    
    batch_size: int = None,
    score_type: int = 0,
    temperature: float = 0.01,
    average_num: int = 1,
    
    target_variance: torch.Tensor = None,
    
    proxy_embeddings: torch.Tensor = None,
    norm: int = 2,
    batch_size_vass: int = None,
    
    vas_inf_type: int = 0,
    arch_ncl: Union[str, None] = None,
) -> None:
    """worker function to new clip score filtering
    
    Args:
        device_index (int): device on which to run the gpu processing
        in_queue (mp.Queue): task queue with fsspec, metadata path pairs
        out_queue (mp.Queue): output queue to send filtred uids
        arch: (Union[str, None]): If specified, we want to apply a threshold to arch=B/32 or L/14 clip scores. Defaults to None.
        
        batch_size (int): Batch size for calculating the new clip score on device.
        score_type (int): 0: I = 0.5 * (I1 + I2), 1: I = I1, 2: I = I2
        temperature (float): temperature for the new clip score. should be set to be 0.01
        average_num (int): average the new clip score for multiple different batch of data
        
        target_variance (torch.Tensor): Target variance matrix for VAS
        
        # part for VAS variant
        proxy_embeddings (torch.Tensor): proxy embeddings for VAS. Defaults to None. If specified, we will calculate the similarity scores with the proxy embeddings.
        norm (int): norm for the similarity scores. 
        batch_size_vass (int): Batch size for calculating target variance (just for VAS-D)on device.
        vas_inf_type (int): 0: norm, 1: inf column
        
        arch_ncl: arch for calculate negative CLIPLoss
        
    """
    
    # if arch_ncl is None:
    #     arch_ncl = arch
    
    while True:
        fs_root = None
        try:
            fs_root = in_queue.get(timeout=1)
        except Empty:
            # case where the queue is depleated, worker should return
            break
        
        fs, path_root = fs_root
        
        
        ##################### arch #######################
        if arch == "l14":
            feature_name = "l14_img"
            feature_text_name = "l14_txt"
            key = key_name
            text_name = "text" 
            
        elif arch == "b32":
            if arch_ncl == 'dfn_p': # regenerate with dfn_p
                feature_name = "oai-clip-vit-b32-image"
                feature_text_name = "oai-clip-vit-b32-text"
                key = "oai-clip-vit-b32-score"
                text_name = "caption"
            else:
                feature_name = "b32_img"
                feature_text_name = "b32_txt"   
                key = "clip_b32_similarity_score"
                text_name = "text"     
        
        elif arch == "dfn_p":
            key = "dfn-p-clip-vit-b32-score"
            feature_name = "dfn-p-clip-vit-b32-image"
            feature_text_name = "dfn-p-clip-vit-b32-text"
            text_name = "caption"
            
        else:
            raise ValueError(f'invalid arch = {arch}')
        
        ##################### arch_ncl #######################
        # ncl just influence feature_ncl
        if arch_ncl is not None:
            if arch_ncl == "l14":
                feature_ncl_name = "l14_img"
                feature_ncl_text_name = "l14_txt"
            elif arch_ncl == "b32":
                feature_ncl_name = "b32_img"
                feature_ncl_text_name = "b32_txt"
            elif arch_ncl == "dfn_p":
                feature_ncl_name = "dfn-p-clip-vit-b32-image"
                feature_ncl_text_name = "dfn-p-clip-vit-b32-text"
            else:
                raise ValueError(f'invalid arch_ncl = {arch_ncl}')
            
        # print(f'ncl: {feature_ncl_name}, {feature_ncl_text_name}; vas: {feature_name}, {feature_text_name}; key = {key}, arch = {arch}, arch_ncl = {arch_ncl}')
        
        df = pd.read_parquet(
            f"{path_root}.parquet", columns=["uid", text_name, key, "url"], filesystem=fs
        )
        
        # print(f'feature_name = {feature_name}, key = {key}, arch = {arch}')

        
        image_embedding, text_embedding = None, None
        image_embedding_ncl, text_embedding_ncl = None, None
        
        with fs.open(f"{path_root}.npz") as f: # -> float32
            ff = np.load(f)
            # show the keys
            # print(f'keys = {ff.keys()}') # keys = KeysView(NpzFile 'object' with keys: b32_img, b32_txt, l14_img, l14_txt, dedup) 
            image_embedding = torch.from_numpy(ff[feature_name])#.float()
            text_embedding = torch.from_numpy(ff[feature_text_name])
            if arch_ncl is not None:
                # print(f'load the ncl features, feature_ncl_name = {feature_ncl_name}, feature_ncl_text_name = {feature_ncl_text_name}')
                image_embedding_ncl = torch.from_numpy(ff[feature_ncl_name])
                text_embedding_ncl = torch.from_numpy(ff[feature_ncl_text_name])
            else:
                print(f'arch_ncl is None, so image_embedding_ncl and text_embedding_ncl are the same as image_embedding and text_embedding')
                image_embedding_ncl = image_embedding
                text_embedding_ncl = text_embedding

        
        uids = df["uid"].values
        
        uids_standard = np.array(
                            [(int(uid[:16], 16), int(uid[16:32], 16)) for uid in uids],
                            np.dtype("u8,u8"),
                        )
        
        # original clip score
        css = df[key].values
            
        # urls and captions
        urls = df["url"].values
        captions = df[text_name].values
        
        
        ncl_css = torch.zeros(image_embedding.shape[0])
        ncl_norm1 = torch.zeros(image_embedding.shape[0])
        ncl_norm2 = torch.zeros(image_embedding.shape[0])
        
        for i in range(average_num):
            css_i, norm1_i, norm2_i = cal_new_clip_score(
                # image_embedding,
                # text_embedding,
                image_embedding_ncl,
                text_embedding_ncl,
                device_index,
                batch_size=batch_size,
                score_type = score_type,
                temperature = temperature,
            )
            ncl_css += css_i
            ncl_norm1 += norm1_i
            ncl_norm2 += norm2_i
        
        if average_num > 1: # 0 or 1 -> no need to average
            ncl_css /= average_num
            ncl_norm1 /= average_num
            ncl_norm2 /= average_num
            
        # print(f'ncl_css[:10] = {ncl_css[:10]}\nncl_norm1[:10] = {ncl_norm1[:10]}\nncl_norm2[:10] = {ncl_norm2[:10]}\ncss[:10] = {css[:10]}')
        # exit(0)
        
        if proxy_embeddings is None and target_variance is None:
            vass = torch.zeros(image_embedding.shape[0])
        else:
            if proxy_embeddings is not None:
                # print(f'calculate the similarity scores with proxy_embeddings, norm = {norm}, batch_size_vass = {batch_size_vass}, vas_inf_type = {vas_inf_type}')
                if vas_inf_type == 0:
                    vass = cal_similarity_scores_norm(
                        image_embedding,
                        proxy_embeddings,
                        device_index,
                        norm=norm,
                        batch_size=batch_size_vass,
                        
                    )
                elif vas_inf_type == 99:
                    tem_as = 0.07 if arch == 'dfn_p' else 0.01
                    vass = cal_exp_as(
                        image_embedding,
                        proxy_embeddings,
                        device_index,
                        norm=norm,
                        batch_size=batch_size_vass,
                        temperature=tem_as,
                    )
                else:
                    vass = cal_similarity_scores_inf_column(
                        image_embedding,
                        proxy_embeddings,
                        device_index,
                        norm=norm,
                        batch_size=batch_size_vass,
                    )
            else:
                print(f'calculate the VAS with target_variance')
                vass = get_vas_gpu(
                    image_embedding,
                    target_variance,
                    device_index,
                )
        
        
        out_queue.put(
            (
                uids_standard,
                ncl_css.numpy(),
                ncl_norm1.numpy(),
                ncl_norm2.numpy(),
                vass.numpy(),
                css,
                urls,
                captions,
            )
        )  
        
        
@torch.no_grad()
def load_all_data_cs_new(
        metadata_dir_path: str,
        arch: str, 
        num_gpus: int,
        
        given_uids_path: str = None,
        
        batch_size: int = 100000,
        score_type: int = 0,
        temperature: float = 0.01,
        average_num: int = 1,
        
        target_variance: torch.Tensor = None,
        
        proxy_embeddings: torch.Tensor = None,
        norm: int = 2,
        batch_size_vass: int = None,
        
        vas_inf_type: int = 0,
        downsampling_ratio: float = 1.0,
        
        arch_ncl: Union[str, None] = None,
    ):
    """load uids, new clip scores, and original clip scores
    
    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        arch (str): Kind of features for CLIP filtering.
        out_queue (mp.Queue): Output queue to send loaded data.
        num_gpus (int): Number of GPU workers.
        
        given_uids_path (str): Path to the given UIDs.
        
        batch_size (int): Batch size for calculating target variance (just for VAS-D) on device.
        score_type (int): 0: I = 0.5 * (I1 + I2), 1: I = I1, 2: I = I2
        temperature (float): temperature for the new clip score
        average_num (int): average the new clip score for the same image and text
        
        target_variance (torch.Tensor): Target variance matrix for VAS
        
        
    """
    
    print(f'begin load_all_data_cs_new, arch = {arch}, metadata_dir_path = {metadata_dir_path}')
        
    fs, url = fsspec.core.url_to_fs(metadata_dir_path)
    root_paths = [
        (fs, str(x.split(".parquet")[0])) for x in fs.ls(url) if ".parquet" in x
    ]
    if downsampling_ratio < 1.0:
        
        root_paths = list(random.sample(root_paths, int(len(root_paths) * downsampling_ratio)))
        print(f'!! downsampling_ratio = {downsampling_ratio}, len of root_paths is {len(root_paths)}')
        
    # initializing task queues
    receive_queue = mp.Queue()
    send_queue = mp.Queue()

    for job in root_paths:
        send_queue.put(job)

    processes = []
    print("starting gpu workers")
    for worker_index in tqdm(range(num_gpus)):
        p = mp.Process(
            target=cs_new_helper,
            kwargs=dict(
                device_index=worker_index,
                in_queue=send_queue,
                out_queue=receive_queue,
                arch=arch,
                
                batch_size=batch_size,
                score_type=score_type,
                temperature=temperature,
                average_num=average_num,
                
                target_variance=target_variance,
                
                proxy_embeddings=proxy_embeddings,
                norm=norm,
                batch_size_vass=batch_size_vass,
                
                vas_inf_type=vas_inf_type,
                arch_ncl=arch_ncl,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.05)

    print("processing metadata with gpu workers")
    pbar = tqdm(total=len(root_paths))

    # running in main thread
    all_uids = []
    all_scores = []
    all_norm1s = []
    all_norm2s = []
    
    all_vass = []
    all_css = []
    all_urls = []
    all_captions = []
    
    def update_p(receive_queue, all_uids, all_scores, all_norm1s, all_norm2s, all_vass, all_css, all_urls, all_captions, pbar):
        # utility function to update the progress bar and store results
        uids, scores, norm1s, norm2s, vass, css, urls, captions = receive_queue.get(timeout=10)
        all_uids.append(uids)
        all_scores.append(scores)
        all_norm1s.append(norm1s)
        all_norm2s.append(norm2s)
        
        all_vass.append(vass)
        all_css.append(css)
        all_urls.append(urls)
        all_captions.append(captions)
        
        pbar.update(1)
        
        return pbar.n
                                
    
    while True:
        # keep checking for jobs finishing and update uids
        try:
            counter = update_p(receive_queue, all_uids, all_scores, all_norm1s, all_norm2s, all_vass, all_css, all_urls, all_captions, pbar)
            if debug == 1 and counter == counter_num: break
        except TimeoutError:
            if all(not p.is_alive() for p in processes):
                try:
                    update_p(receive_queue, all_uids, all_scores, all_norm1s, all_norm2s, all_vass, all_css, all_urls, all_captions, pbar)
                    
                except TimeoutError:
                    raise RuntimeError("All processes dead but nothing in queue!")
        except Empty:
            pass

        if all(not p.is_alive() for p in processes):
            # case where all workers have exited
            try:
                update_p(receive_queue, all_uids, all_scores, all_norm1s, all_norm2s, all_vass, all_css, all_urls, all_captions, pbar)            
            except Empty:
                print("Result queue is empty and all workers have exited")
                break

    pbar.close()
    for p in processes:
        p.join(timeout=1)

    
    uids = np.concatenate(all_uids)
    scores = np.concatenate(all_scores).astype(np.float32)
    norm1s = np.concatenate(all_norm1s).astype(np.float32)
    norm2s = np.concatenate(all_norm2s).astype(np.float32)
    
    vass = np.concatenate(all_vass).astype(np.float32)
    css = np.concatenate(all_css)
    urls = np.hstack(all_urls)
    captions = np.hstack(all_captions)
    
    print(f'uids.shape = {uids.shape}, scores.shape = {scores.shape}, norm1s.shape = {norm1s.shape}, norm2s.shape = {norm2s.shape}, vass.shape = {vass.shape}, css.shape = {css.shape}, urls.shape = {urls.shape}, captions.shape = {captions.shape}')
    
    # Filter by given UIDs, which may be the subet from other filtering methods
    if given_uids_path is not None:
        given_uids = np.load(given_uids_path)
        start = time.time()
        given_uids_mask = np.isin(uids, given_uids)
        print(f'filter by given_uids, np.isin use time = {time.time() - start}')
        
        uids = uids[given_uids_mask]
        scores = scores[given_uids_mask] # note that tensor can take np.ndarray as index
        norm1s = norm1s[given_uids_mask]
        norm2s = norm2s[given_uids_mask]
        vass = vass[given_uids_mask]
        css = css[given_uids_mask]
        urls = urls[given_uids_mask]
        captions = captions[given_uids_mask]
        print(f'================== after filtering by given_uids ==================')
        print(f'uids.shape = {uids.shape}, scores.shape = {scores.shape}, norm1s.shape = {norm1s.shape}, norm2s.shape = {norm2s.shape}, vass.shape = {vass.shape}, css.shape = {css.shape}, urls.shape = {urls.shape}, captions.shape = {captions.shape}')
        
            
    
    # Calculate initial target variance as the sum of outer products
    return uids, scores, norm1s, norm2s, vass, css, urls, captions

@torch.no_grad()
def get_target_variance(
    target_variance_name: str,
    files_path: str,
):
    # get target variance
    print("loading target variance")
    if target_variance_name == "self" or target_variance_name == None:
        target_variance = None
        print(f'name is self, set target variance to None')
    else:
        if target_variance_name == 'imagenet-1k' or target_variance_name == 'variance_imagenet_1k':
            target_path = os.path.join(files_path, 'variance', "variance_imagenet_1k.pt")
            target_variance = torch.load(target_path)
            print(f'load target_variance from {target_path}')
        else:
            target_path = os.path.join(files_path, 'variance', f"variance_{target_variance_name}.pt")
            target_variance = torch.load(target_path)
            print(f'load target_variance from {target_path}')
            
    return target_variance
            
    

@torch.no_grad()
def load_uids_with_cs_new(
    metadata_dir_path: str,
    files_path: str,
    
    num_gpus: int,
    arch: Union[str, None] = None,
    threshold: Union[float, None] = None,
    fraction: Union[float, None] = None,
    
    target_variance_name: str = 'imagenet_1k',
    threshold_vas: Union[float, None] = None,
    fraction_vas: Union[float, None] = None,

    given_uids_path: Union[str, None] = None,
    
    # higher_is_better_vas: int = 1,
    
    batch_size: int = 32768,
    
    score_type: int = 0,
    temperature: float = 0.01,
    
    average_num: int = 1,

    proxy_name: str = None,
    proxy_path: str = None,
    cache_path: str = None,
    
    norm: int = 2,
    batch_size_vass: int = None,
    
    if_use_old_cs: int = 0,
    
    vas_inf_type: int = 0,
    downsampling_ratio: float = 0.03, # 1.0, #0.03,
    
    save_path: str = None,
    arch_ncl: Union[str, None] = None,
) -> np.ndarray:
    """cs_new filter

    Args:
        metadata_dir_path (str): Directory path containing metadata files.
        cache_path (str): Path to the cache directory.
        
        num_gpus (int): Number of GPU workers.
        arch (Union[str, None], optional): Kind of features for CLIP filtering. Defaults to None. can be dfn_p now
        threshold (Union[float, None], optional): Threshold to apply to CLIP features. Defaults to None.
        fraction (Union[float, None], optional): Top k fraction to apply to CLIP features. Defaults to None.
        threshold_vas (Union[float, None], optional): Threshold to apply to variance alignment scores. Defaults to None.
        fraction_vas (Union[float, None], optional): Top k fraction to apply to variance alignment scores. Defaults to None.
        target_variance_name (str, optional): Target variance name. Defaults to 'imagenet-1k'.
        given_uids_path (Union[str, None], optional): Path to the given UIDs. Defaults to None.
        
        batch_size (int): Batch size for ccalculating the new clip score on device. Defaults to 32768.
        score_type (int): 0: I = 0.5 * (I1 + I2), 1: I = I1, 2: I = I2
        temperature (float): temperature for the new clip score. Defaults to 0.01.
        
        average_num (int): average the new clip score for multiple times. Defaults to 1.
    """
    
    # if arch_ncl == 'dfn_p' or (arch_ncl is None and arch == 'dfn_p'):
    #     assert temperature == 0.07, f'temperature should be 0.07 for dfn_p, but got {temperature}'
    # else:
    #     assert temperature == 0.01, f'temperature should be 0.01, but got {temperature}'
        
    # get target variance
    print(f'**************** temperature = {temperature}, average_num = {average_num}, proxy {proxy_name}, target var {target_variance_name}, arch {arch}, arch_ncl {arch_ncl} ****************')
    
    if fraction_vas == fraction:
        print(f'fraction_vas = fraction = {fraction}, so we will not filter by vas')
        target_variance = None
        proxy_embeddings = None
    else:
        target_variance = get_target_variance(target_variance_name, files_path)
        proxy_embeddings = None if proxy_name is None else get_proxy_embeddings(proxy_name, proxy_path, cache_path)
    
    if fraction == 1.0: # no need for new clip loss
        assert average_num == 0, f'fraction = 1.0, so average_num should be 0, but got {average_num}'
    
    uids, also_css, norm1s, norm2s, vass, css, urls, captions = load_all_data_cs_new(
            metadata_dir_path, arch, num_gpus, given_uids_path, batch_size=batch_size, 
            score_type=score_type, temperature=temperature, average_num=average_num, 
            target_variance=target_variance,
            proxy_embeddings=proxy_embeddings, batch_size_vass=batch_size_vass, norm=norm,
            vas_inf_type=vas_inf_type,
            downsampling_ratio = downsampling_ratio,
            arch_ncl=arch_ncl,
        )
    uids_copy = copy.deepcopy(uids)
    
    normalization = norm1s + norm2s
    new_css = also_css - normalization
    
    # show the correlation between new clip score and clip score
    print(f'also_css.shape = {also_css.shape}, css.shape = {css.shape}, vass.shape = {vass.shape}, norm1s.shape = {norm1s.shape}, norm2s.shape = {norm2s.shape}, new_css.shape = {new_css.shape}')
    print(f'correlation between new_css and css = {scipy.stats.spearmanr(new_css, css)}') # correlation between new_css and css = SignificanceResult(statistic=0.9642641277898771, pvalue=0.0)
    
    # bs 32768
    # t: 0.07 -> correlation between new_css and css = SignificanceResult(statistic=0.9669697452192089, pvalue=0.0)
    # t: 0.2 -> correlation between new_css and css = SignificanceResult(statistic=0.9339138174256968, pvalue=0.0)
    # t: 0.01 -> statistic=0.9748452266488377
    
    ################################ first visualize the results ################################
    vis_name = f'kk_an_{average_num}_f{fraction}_fvas{fraction_vas}_arch_{arch}_arch_ncl_{arch_ncl}_old{if_use_old_cs}_norm{norm}_inf_type{vas_inf_type}_temp{temperature}'
    if given_uids_path is not None:
        # get the final uid_name by parsing
        g_name = given_uids_path.split('/')[-1]
        g_name = g_name.split('.')[0]
        vis_name += g_name
        
    save_dir = os.path.join(files_path, 'vis', vis_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # first get the ranks for new clip score and old clip score, for example, if cs = [0.2, 0.3, 0.1, 0.4], then we get the rank of each element as [2, 3, 1, 4]
    
    # new clip score * temperature = old_cs - normalization
    
    
    rank_old_cs = get_ranks_ratios(css) # the smaller the cs, the smaller rank
    rank_new_cs = get_ranks_ratios(new_css)
    rank_vas = get_ranks_ratios(vass)
    rank_normalization = get_ranks_ratios(normalization)
    rank_norm1s = get_ranks_ratios(norm1s)
    rank_norm2s = get_ranks_ratios(norm2s)
    
    relative_rank = rank_new_cs - rank_old_cs
    sum_rank_new_cs_vass = rank_new_cs + rank_vas
    
    
    # add values
    add_values = {
        # 'new_cs': new_css,
        'rank_old_cs': rank_old_cs,
        'rank_new_cs': rank_new_cs,
        'rank_vas': rank_vas,
        # 'rank_norm': rank_normalization,
        # 'rank_norm1': rank_norm1s,
        # 'rank_norm2': rank_norm2s,
        # 'relative_rank': relative_rank,
        'cs': css,
        'vas': vass,
        'new_cs': new_css,
        # 'sum_rank': sum_rank_new_cs_vass,
    }
    
    ICV = ImageCaptionVisualizer(
        save_path=save_dir,
        urls = urls,
        captions = captions,
        add_values=add_values,
    )
    
    # ICV.vis_datasets(
    #     note = f'_standard',
    #     key = 'sum_rank',
    #     # key = 'new_cs',
    #     # ratio_list = [0.8, 0.2]
    #     repeat_num = 2
    # )
    
    # ICV.vis_datasets(
    #     note = f'_standard',
    #     key = 'relative_rank',
    #     # key = 'new_cs',
    #     ratio_list = [1.0, 0.99, 0.98, 0.02, 0.01, 0.0],
    #     repeat_num = 5 #5
    # )
    
    # ICV.vis_datasets(
    #     note = f'_standard',
    #     key = 'rank_norm',
    #     # key = 'new_cs',
    #     # ratio_list = [0.8, 0.2]
    #     repeat_num = 2
    # )
    
    # VAS
    
    
    ICV.vis_datasets(
        note = f'_standard',
        key = 'rank_new_cs',
        # key = 'new_cs',
        ratio_list = [1.0, 0.9, 0.8, 0.5, 0.2, 0.1, 0.0],
        repeat_num = 5,
    )
    
    ICV.vis_datasets(
        note = f'_standard',
        key = 'rank_vas',
        ratio_list = [1.0, 0.9, 0.8, 0.5, 0.2, 0.1, 0.0],
        repeat_num = 5,
        with_caption=False,
    )
    
    # if given_uids_path is not None:
    #     exit(0)
    # exit(0)
    
    ####################################################################################
    
    
    total_num = len(uids)
    
    
    
    ###########  normal case -> first filter by old/new clipscore, then filter by VAS
    # first filter by new clip score
    if if_use_old_cs == 0:
        print(f'================== filtering by NEW clip_score ==================')
        select_indices = filter_by_score(new_css, fraction, threshold, total_num, name='new_clip_score')
    else:
        print(f'================== filtering by OLD clip_score ==================')
        assert average_num == 0, f'average_num should be 0 when using old clip score'
        select_indices = filter_by_score(css, fraction, threshold, total_num, name='clip_score')
        
    uids = uids[select_indices]
    new_css = new_css[select_indices]
    vass = vass[select_indices]
    css = css[select_indices]
    
    print(f'uids.shape = {uids.shape}, new_css.shape = {new_css.shape}, css.shape = {css.shape}, vass.shape = {vass.shape}')
    
    
    # then filter by VAS
    if (threshold_vas is None and fraction_vas is None) or fraction_vas == fraction:
        print(f'no need to filter by vas')
        return uids
    
    print(f'================== filtering by vas ==================')
    select_indices_vas = filter_by_score(vass, fraction_vas, threshold_vas, total_num, name='vas')
    uids = uids[select_indices_vas]
    new_css = new_css[select_indices_vas]
    vass = vass[select_indices_vas]
    css = css[select_indices_vas]
    
    # if norm == 103: # filter values with too high
    
    print(f'uids.shape = {uids.shape}, new_css.shape = {new_css.shape}, css.shape = {css.shape}, vass.shape = {vass.shape}')
    
    
    # #### re visualize
    # rank_old_cs = get_ranks_ratios(css) # the smaller the cs, the smaller rank
    # rank_new_cs = get_ranks_ratios(new_css)
    # rank_vas = get_ranks_ratios(vass)
    
    # re_add_values = {
    #     # 'new_cs': new_css,
    #     'rank_old_cs': rank_old_cs,
    #     'rank_new_cs': rank_new_cs,
    #     'rank_vas': rank_vas,
    #     'cs': css,
    #     'vas': vass,
    #     'new_cs': new_css,
    # }
    
    # save_dir = os.path.join(files_path, 'vis', vis_name + '_final')
    # ICV = ImageCaptionVisualizer(
    #     save_path=save_dir,
    #     urls = urls,
    #     captions = captions,
    #     add_values=re_add_values,
    # )
    
    # ICV.vis_datasets(
    #     note = f'_standard',
    #     key = 'rank_new_cs',
    #     ratio_list = [1.0, 0.99, 0.98, 0.95, 0.9, 0.8, 0.5, 0.1, 0.0],
    #     repeat_num = 1
    # )
    
    # # VAS
    # ICV.vis_datasets(
    #     note = f'_standard',
    #     key = 'rank_vas',
    #     # ratio_list = [0.8, 0.2]
    #     ratio_list = [1.0, 0.99, 0.98, 0.95, 0.9, 0.8, 0.5, 0.1, 0.0],
    #     repeat_num = 1
    # )
    
    
        
    
    if norm == 101:
        # also filter by the sum_rank
        print(f'================== filtering by sum_rank: f{fraction_vas} no threshold ==================')
        select_indices = filter_by_score(sum_rank_new_cs_vass, fraction_vas, None, total_num, name='sum_rank')
        uids_copy = uids_copy[select_indices]
        
        # sort
        uids_copy.sort()
        
        # change the save_path: add '_sum_rank' before '.npy' in the save_path
        true_save_path = save_path.replace('.npy', '_sum_rank.npy') # will not change the original save_path
        
        # store uids_copy
        print(f'saving uids_copy to {true_save_path} with length = {len(uids_copy)}')
        np.save(true_save_path, uids_copy)
        
    
    
    return uids