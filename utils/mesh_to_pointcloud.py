# Reference: https://github.com/JRyanShue/NFD/blob/main/nfd/neural_field_diffusion/metrics/mesh_to_pointcloud.py
import argparse
import glob
import os

from render_utils import scale_to_unit_sphere

import numpy as np
from tqdm import tqdm
import trimesh


def main(args):

    mesh_files = [
        os.path.join(args.mesh_dir, f) for f in glob.glob(f"{args.mesh_dir}/**/*{args.suffix}", recursive=True)
    ]

    for i, mesh_file in tqdm(enumerate(mesh_files, start=1)):
        mesh = trimesh.load_mesh(mesh_file, file_type=args.suffix)
        mesh = scale_to_unit_sphere(mesh)
        pc = trimesh.sample.sample_surface(mesh, count=args.num_points)
        pc = np.array(pc[0])

        np.save(os.path.join(args.pc_dir, f"point_samples_{i}.npy"), pc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", type=str, required=True)
    parser.add_argument("--pc-dir", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="off")
    parser.add_argument("--num-points", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.pc_dir, exist_ok=True)
    main(args)