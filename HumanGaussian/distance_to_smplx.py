import numpy as np
import matplotlib.pyplot as plt
from animation import Skeleton
from types import SimpleNamespace
import os

def main():
    # ---------- paths ----------
    ply_path = "./content/save_ply/A_boy_with_a_beanie_wearing_black_leather_shoes-T-pose.ply"
    smplx_path = "./models/models_smplx_v1_1/models"

    # ---------- init ----------
    opt = SimpleNamespace()
    opt.ply = ply_path
    opt.motion = "./content/py3d_smpl_walk_motion.npz"

    sk = Skeleton(opt)
    sk.load_smplx(smplx_path)

    mapping_dist = sk.mapping_dist  # (N,)

    # ---------- basic statistics ----------
    print("\n=== Mapping distance statistics ===")
    print(f"Num Gaussians : {mapping_dist.shape[0]}")
    print(f"Max           : {mapping_dist.max():.6f}")
    print(f"Min           : {mapping_dist.min():.6f}")
    print(f"Mean          : {mapping_dist.mean():.6f}")
    print(f"Std           : {mapping_dist.std():.6f}")

    # ---------- histogram statistics ----------
    bins = 100
    counts, bin_edges = np.histogram(mapping_dist, bins=bins)

    print("\n=== Mapping distance histogram (range : count) ===")
    for i in range(len(counts)):
        left  = bin_edges[i]
        right = bin_edges[i + 1]
        print(f"[{left:.6f}, {right:.6f}): {counts[i]}")

    # ---------- plot & save ----------
    plt.figure(figsize=(8, 5))
    plt.hist(mapping_dist, bins=bins, edgecolor="black")
    plt.xlabel("Distance")
    plt.ylabel("Number of Gaussians")
    plt.title("Histogram of Gaussian–Mesh Face Distances")
    plt.grid(True)

    ply_name = os.path.splitext(os.path.basename(ply_path))[0]
    save_path = f"./{ply_name}_mapping_dist_hist.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"\nHistogram image saved to: {save_path}")

if __name__ == "__main__":
    main()
