# Reference: https://github.com/JRyanShue/NFD/blob/main/nfd/neural_field_diffusion/metrics/evaluate_pc.py
import argparse
from pathlib import Path
import os

from pytorch3d.loss import chamfer_distance
from evaluation_metrics import compute_all_pc_metrics
from PyTorchEMD.emd import earth_mover_distance
from ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist

import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
import torch


def normalize_point_clouds(pcs: torch.Tensor, return_shift_scale=False, padding=0.0):
    # refactored version for batched processing
    pcs = pcs.clone()
    pc_max = pcs[..., :3].amax(dim=1, keepdim=True)  # (B, 1, 3)
    pc_min = pcs[..., :3].amin(dim=1, keepdim=True)
    shift = (pc_min + pc_max) / 2
    scale = 2 / (pc_max - pc_min).amax(dim=-1, keepdim=True) * (1 - padding)
    pcs[..., :3] = (pcs[..., :3] - shift) * scale

    if return_shift_scale:
        return pcs, shift, scale
    else:
        return pcs


def get_pcs(data_dir, type, interval, only_test=False):
    assert type == "gt" or type == "gen" or type == "gen_cond"
    model_ids, cat_ids = None, None
    if type == "gt":
        pc_files, model_ids, cat_ids = [], [], []
        for path in Path(data_dir).glob(f"**/*.npy"):
            if only_test:
                if path.parts[-3] == "test":
                    pc_files.append(path)
                    model_ids.append(path.parts[-2])
                    cat_ids.append(path.parts[-4])
            else:
                pc_files.append(path)
                model_ids.append(path.parts[-2])
                cat_ids.append(path.parts[-4])
            pc_files = pc_files[:interval]
            model_ids = model_ids[:interval]
            cat_ids = cat_ids[:interval]
    elif type == "gen_cond":
        pc_files, model_ids = [], []
        for path in Path(data_dir).glob(f"**/*.npy"):
            pc_files.append(path)
            model_ids.append(path.stem)
    else:
        pc_files = [path for path in Path(data_dir).glob(f"**/*.npy")]

    pcs = []
    for pc_file in tqdm(pc_files):
        pc = np.load(str(pc_file), allow_pickle=True)
        pcs.append(pc)
    normalized_pcs = normalize_point_clouds(torch.from_numpy(np.vstack(pcs).astype(np.float32).reshape(-1, pc.shape[0], pc.shape[1])).type(torch.float32))
    return normalized_pcs, model_ids, cat_ids


def main(args):
    print("path to generated point clouds: ", args.gen_dir)
    if args.task_type == "uncond":
        gen_pcs, _, _ = get_pcs(args.gen_dir, "gen", args.gt_interval)
        gt_pcs, gt_model_ids, gt_cat_ids = get_pcs(args.gt_dir, "gt", args.gt_interval)
        results = {}
        results = compute_all_pc_metrics(gen_pcs, gt_pcs, args.batch_size)
        print(results)
    
    # if args.task_type == "cond":
    #     gen_pcs, gen_model_ids, _ = get_pcs(args.gen_dir, "gen", args.gt_interval)
    #     gt_pcs, gt_model_ids, gt_cat_ids = get_pcs(args.gt_dir, "gt", args.gt_interval)
    #     results = {}
        
    results = {k: float(v) for k, v in results.items()}
    return results
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gt-interval", type=int, default=100)
    parser.add_argument("--task_type", type=str, default="uncond")
    args = parser.parse_args()

    main(args)