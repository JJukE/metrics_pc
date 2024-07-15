import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import trange
from tqdm.auto import tqdm

from ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist

chamfer_dist = chamfer_3DDist()


#============================================================
# Point Clouds
#============================================================
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


# from https://github.com/alexzhou907/PVD
def pairwise_cd(xs, ys, b):
    n, m = len(xs), len(ys)
    cd_all = []
    for i in trange(n, ncols=100, file=sys.stdout, desc="pairwise_cd"):
        x = xs[i, None].cuda(non_blocking=True) # (1, n, 3)
        cd_lst = []
        for j in range(0, m, b):
            b_ = min(m - j, b)
            y = ys[j:j + b_].cuda(non_blocking=True) # (b, n, 3)
            dl, dr, _, _ = chamfer_dist(x.repeat(b_, 1, 1).contiguous(), y.contiguous()) # (b, n), (b, n)
            cd = (dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1) # (1, b)
            cd_lst.append(cd)
        cd_lst = torch.cat(cd_lst, dim=1) # (1, m)
        cd_all.append(cd_lst)
    cd_all = torch.cat(cd_all, dim=0) # (n, m)
    return cd_all


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        "lgan_mmd": mmd,
        "lgan_cov": cov,
        "lgan_mmd_smp": mmd_smp,
    }


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    dev = Mxx.device

    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0, device=dev), torch.zeros(n1, device=dev)))
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float("inf")
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1, device=dev))).topk(k, 0, False)

    count = torch.zeros(n0 + n1, device=dev)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1, device=dev)).float()

    s = {
        "tp": (pred * label).sum(),
        "fp": (pred * (1 - label)).sum(),
        "fn": ((1 - pred) * label).sum(),
        "tn": ((1 - pred) * (1 - label)).sum(),
    }

    s.update(
        {
            "precision": s["tp"] / (s["tp"] + s["fp"] + 1e-10),
            "recall": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_t": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "acc_f": s["tn"] / (s["tn"] + s["fp"] + 1e-10),
            "acc": torch.eq(label, pred).float().mean(),
        }
    )
    return s


def compute_metric_cd(xs, ys, b):
    rs = pairwise_cd(ys, xs, b)
    rr = pairwise_cd(ys, ys, b)
    ss = pairwise_cd(xs, xs, b)
    
    results = {}
    res_mmd = lgan_mmd_cov(rs.T)
    results.update({"%s-CD" % k: v for k, v in res_mmd.items()})
    res_1nn = knn(rr, rs, ss, 1, sqrt=False)
    results.update({"1-NN-CD-%s" % k: v for k, v in res_1nn.items() if "acc" in k})
    return results


def get_pcs(data_dir, type, interval=None):
    # pc_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("npy")]
    # pc_files = [
    #     os.path.join(data_dir, f) for f in glob.glob(f"{data_dir}/**/*.npy", recursive=True)
    # ]
    pc_files = [path for path in Path(data_dir).glob(f"**/*.npy")]
    if type == "gt":
        pc_files = [path for path in Path(data_dir).glob(f"**/*.npy")][:interval]

    pcs = []
    for pc_file in tqdm(pc_files):
        pc = np.load(str(pc_file), allow_pickle=True)
        pcs.append(pc)
    pcs = np.vstack(pcs).astype(np.float32).reshape(-1, pc.shape[0], pc.shape[1])
    return pcs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_dir", type=str)
    parser.add_argument("gen_dir", type=str)
    parser.add_argument("batch_size", type=int, defualt=32)
    parser.add_argument("gt_interval", type=int, default=100)
    args = parser.parse_args()

    gen_pcs = torch.tensor(get_pcs(args.gen_dir, "gen"), device="cuda")
    gt_pcs = torch.tensor(get_pcs(args.gt_dir, "gt"), device="cuda")
    
    metric_res = compute_metric_cd(normalize_point_clouds(gen_pcs), normalize_point_clouds(gt_pcs), args.batch_size)
    for k, v in metric_res.items():
        print(f"{k}: {v}") # lgan-mmd-cov * 10^3, lgan-mmd-