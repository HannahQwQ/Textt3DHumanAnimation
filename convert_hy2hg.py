import argparse
import numpy as np


def convert_hy_to_hg(input_npz, output_npz, fps=30):
    data = np.load(input_npz, allow_pickle=True)

    poses_hy = data["poses"]       # (T, 156)
    trans = data["trans"]          # (T, 3)
    betas_hy = data["betas"]       # (1, 16)
    gender = data["gender"][0]     # string

    T, D = poses_hy.shape
    if D < 72:
        raise ValueError(f"Invalid HY pose dim: {D}, expected at least 72 (24 joints × 3)")

    # reshape to (T, 24, 3)
    poses = poses_hy[:, :72].reshape(T, 24, 3).astype(np.float64)

    trans = trans.astype(np.float64)
    betas = betas_hy.reshape(-1)[:10].astype(np.float64)

    hg_data = {
        "poses": poses,
        "trans": trans,
        "betas": betas,
        "gender": np.array(gender, dtype="<U8"),
        "mocap_framerate": np.array(fps, dtype=np.int64),
    }

    np.savez(output_npz, **hg_data)

    print("✅ Converted HY-Motion → HumanGaussian legacy motion format")
    print(f"   Input : {input_npz}")
    print(f"   Output: {output_npz}")
    print(f"   Frames: {T}, FPS: {fps}")
    print(f"   Shapes:")
    print(f"     poses : {poses.shape}")   # (T,24,3)
    print(f"     trans : {trans.shape}")   # (T,3)
    print(f"     betas : {betas.shape}")   # (10,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HY-Motion npz to HumanGaussian legacy motion format"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()
    convert_hy_to_hg(args.input, args.output, fps=args.fps)
