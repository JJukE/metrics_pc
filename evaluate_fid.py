# Reference: https://github.com/JRyanShue/NFD/blob/main/nfd/neural_field_diffusion/metrics/evaluate_fid.py 
import argparse
import os
import sys
from pathlib import Path

import tqdm

from pytorch_fid.fid_score import calculate_fid_given_paths


# def get_common_stems(list1, list2):
#     stems1 = set(Path(path).stem for path in list1)
#     stems2 = set(Path(path).stem for path in list2)
#     common_stems = stems1 & stems2

#     common_list1 = [path for path in list1 if Path(path).stem in common_stems]
#     common_list2 = [path for path in list2 if Path(path).stem in common_stems]
#     return common_list1, common_list2


def main(args):
    # gt_paths = [path for path in Path(args.gt_dir).glob("**/*.png")]
    # gen_paths = [path for path in Path(args.gen_dir).glob("**/*.png")]
    # print("len(gt_paths): ", len(gt_paths))
    # print("len(gen_paths): ", len(gen_paths))

    # gt_paths, gen_paths = get_common_stems(gt_paths, gen_paths)
    # print("len(gt_paths): ", len(gt_paths))
    # print("len(gen_paths): ", len(gen_paths))

    fid = 0
    for view_index in tqdm.tqdm(range(args.num_views)):
        # gt_paths_ = sorted([path for path in gt_paths if f"view_{view_index}" in path])
        # gen_paths_ = [path for path in gen_paths if f"view_{view_index}" in gen_paths_]
        # print(gt_paths_)

        # if f"view_{view_index}" in gt_paths:
        gt_view_dir = os.path.join(args.gt_dir, f"view_{view_index}")
        gen_view_dir = os.path.join(args.gen_dir, f"view_{view_index}")
        view_score = calculate_fid_given_paths([gt_view_dir, gen_view_dir],
                                               args.batch_size,
                                               args.device,
                                               dims=2048,
                                               num_workers=8)
        
        fid += view_score
    fid /= args.num_views
    print(f"FID: {fid}")
    return {"FID": float(fid)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", type=str, required=True)
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--num_views", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    main(args)