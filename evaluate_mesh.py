# Reference: https://github.com/JRyanShue/NFD/blob/main/nfd/neural_field_diffusion/metrics/evaluate_mesh.py
import argparse
import glob
import os
from pathlib import Path

from evaluation_metrics import compute_all_mesh_metrics

import numpy as np
from tqdm import tqdm
import torch
import trimesh

# def get_meshes(data_dir):
#     pc_files = [
#         os.path.join(data_dir, f) for f in os.listdir(data_dir)
#         if f.endswith("npy")
#     ]

#     pcs = []
#     for pc_file in tqdm(pc_files):
#         pc = np.load(pc_file)
#         pcs.append(pc)

#     return torch.from_numpy(
#         np.vstack(pcs).reshape(-1, pc.shape[0],
#                                pc.shape[1])).type(torch.float32)


def get_meshes(data_dir, suffix, type, interval):
    mesh_files = [
        path for path in Path(data_dir).glob(f"**/*{suffix}")
        # os.path.join(data_dir, f) for f in glob.glob(f"{data_dir}/**/*{suffix}", recursive=True)
    ]
    # if type == "gt":
    #     mesh_files = [mf for i, mf in enumerate(mesh_files) if i % interval == 0]
    mesh_files = sorted(mesh_files[:interval], key=lambda file: str(file))
    file_suffix = mesh_files[0].suffix

    meshes = [
        trimesh.load_mesh(mesh_file, file_type=file_suffix)
        for mesh_file in tqdm(mesh_files)
    ]

    return meshes


def main(args):
    gen_meshes = get_meshes(args.gen_mesh_dir, args.suffix, "gen", args.gt_interval)
    gt_pcs = get_meshes(args.gt_mesh_dir, args.suffix, "gt", args.gt_interval)

    results = compute_all_mesh_metrics(gen_meshes, gt_pcs, args.batch_size)
    print(results)
    results = {k: float(v) for k, v in results.items()}
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-mesh-dir", type=str, required=True)
    parser.add_argument("--gen-mesh-dir", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="off")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gt-interval", type=int, default=100)
    args = parser.parse_args()

    main(args)