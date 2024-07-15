# Reference: https://github.com/JRyanShue/NFD/blob/main/nfd/neural_field_diffusion/metrics/render_fid.py
import argparse
import glob
import os
from pathlib import Path
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# # os.environ["PYOPENGL_PLATFORM"] = "osmesa"

from render_utils import Render, create_pose, scale_to_unit_sphere

import numpy as np
from tqdm import tqdm
import torch
import torchvision
import trimesh


def get_meshes(data_dir, type, suffix, train_model_ids=None): # , cat_id_loc=None):
    if type == "gt":
        assert train_model_ids is not None
        mesh_files = [
            path for path in Path(data_dir).glob(f"**/*{suffix}") if path.parts[-3] in train_model_ids
            # os.path.join(data_dir, f) for f in glob.glob(f"{data_dir}/**/*{suffix}", recursive=True)
        ]
        # cat_ids = [path.parts[-4] for path in mesh_files]
        model_ids = [path.parts[-3] for path in mesh_files]
        file_suffix = mesh_files[0].suffix
    
    else:
        mesh_files, model_ids = [], []
        for path in Path(data_dir).glob(f"**/*{suffix}"):
            mesh_files.append(path)
            model_ids.append(str(path.stem).split("_")[0])
        file_suffix = mesh_files[0].suffix

    # # Check if the mesh files are loaded correctly
    # print("Loading meshes...")
    # meshes = []
    # for mesh_idx, mesh_file in tqdm(enumerate(mesh_files), total=len(mesh_files)):
    #     meshes.append(trimesh.load_mesh(mesh_file, file_type=file_suffix))
    #     try:
    #         print(meshes[mesh_idx].vertices.shape)
    #     except:
    #         print(f"Error occurs when loading {mesh_file}.")

    meshes = [
        trimesh.load_mesh(mesh_file, file_type=file_suffix)
        for mesh_file in tqdm(mesh_files, total=len(mesh_files))
    ]

    return meshes, model_ids


FrontVector = (np.array(
    [[0.52573, 0.38197, 0.85065], [-0.20081, 0.61803, 0.85065],
     [-0.64984, 0.00000, 0.85065], [-0.20081, -0.61803, 0.85065],
     [0.52573, -0.38197, 0.85065], [0.85065, -0.61803, 0.20081],
     [1.0515, 0.00000, -0.20081], [0.85065, 0.61803, 0.20081],
     [0.32492, 1.00000, -0.20081], [-0.32492, 1.00000, 0.20081],
     [-0.85065, 0.61803, -0.20081], [-1.0515, 0.00000, 0.20081],
     [-0.85065, -0.61803, -0.20081], [-0.32492, -1.00000, 0.20081],
     [0.32492, -1.00000, -0.20081], [0.64984, 0.00000, -0.85065],
     [0.20081, 0.61803, -0.85065], [-0.52573, 0.38197, -0.85065],
     [-0.52573, -0.38197, -0.85065], [0.20081, -0.61803, -0.85065]])) * 2


def render_mesh(mesh,
                resolution=1024,
                index=5,
                background=None,
                scale=1,
                no_fix_normal=True,
                light=True):

    camera_pose = create_pose(FrontVector[index] * scale)

    render = Render(size=resolution,
                    camera_pose=camera_pose,
                    background=background,
                    light=light)

    triangle_id, rendered_image, normal_map, depth_image, p_images = render.render(
        path=None, clean=True, mesh=mesh, only_render_images=no_fix_normal)
    return rendered_image


def render_for_fid(mesh, root_dir, mesh_idx, num_views, light=True):
    render_resolution = 299
    mesh = scale_to_unit_sphere(mesh)
    for j in range(num_views):
        image = render_mesh(mesh, index=j, resolution=render_resolution, light=light) / 255
        torchvision.utils.save_image(
            torch.from_numpy(image.copy()).permute(2, 0, 1),
            f"{root_dir}/view_{j}/{mesh_idx}.png")


def main(args):
    # Generate images of generated meshes
    train_ids = None
    if args.type == "gt":
        assert args.split_path is not None
        train_split_path = Path(args.split_path) / args.cat_id / "model_ids_train.txt"
        with open(train_split_path, "r") as file:
            train_ids = [line.strip() for line in file]

    gen_meshes, gen_model_ids = get_meshes(args.gen_dir, args.type, args.suffix, train_model_ids=train_ids)
    print("Number of meshes: ", len(gen_meshes))
    for i in range(args.num_views):
        os.makedirs(os.path.join(args.gen_out_dir, f"view_{i}"), exist_ok=True)
    
    if args.process_with_model_id:
        for mesh_idx, gen_mesh in tqdm(enumerate(gen_meshes), total=len(gen_meshes)):
            render_for_fid(gen_mesh, args.gen_out_dir, gen_model_ids[mesh_idx], args.num_views, light=not args.no_light)
    else:
        for mesh_idx, gen_mesh in tqdm(enumerate(gen_meshes), total=len(gen_meshes)):
            render_for_fid(gen_mesh, args.gen_out_dir, mesh_idx, args.num_views, light=not args.no_light)

    # # Generate images of GT meshes
    # if args.gt_dir is not None:
    #     gt_meshes, gt_model_ids = get_meshes(args.gt_dir)
    #     for i in range(args.num_views):
    #         os.mkdir(os.path.join(args.gt_out_dir, f"view_{i}"))
    #     if args.process_with_model_id:
    #         for mesh_idx, gt_mesh in tqdm(enumerate(gt_meshes), total=len(gt_meshes)):
    #             render_for_fid(gt_mesh, args.gt_out_dir, gt_model_ids[mesh_idx], args.num_views)
    #     else:
    #         for mesh_idx, gt_mesh in tqdm(enumerate(gt_meshes), total=len(gt_meshes)):
    #             render_for_fid(gt_mesh, args.gt_out_dir, mesh_idx, args.num_views)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gt_dir", type=str, default=None)
    parser.add_argument("--gen_dir", type=str, required=True) # gt-dir here when preprocessing
    # parser.add_argument("--gt_out_dir", type=str, default=None)
    parser.add_argument("--gen_out_dir", type=str, required=True) # gt-out-dir here when preprocessing
    parser.add_argument("--num_views", type=int, default=20)
    parser.add_argument("--type", type=str, default="gt") # gt or gen
    parser.add_argument("--suffix", type=str, default="off")
    parser.add_argument("--process_with_model_id", action="store_true") # model_ids for the output file names, instead of mesh indices to process gt renderings
    parser.add_argument("--no_light", action="store_true") # defaults to light == True
    parser.add_argument("--split_path", type=str, default=None)
    parser.add_argument("--cat_id", type=str, default="03001627")
    args = parser.parse_args()

    main(args)