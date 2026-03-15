import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse

def rot6d_to_axis_angle(rot6d, flip_z=False):
    """
    Convert 6D rotation representation to axis-angle (3D) vector.
    Optionally flip the Z-axis direction to match HG coordinate.
    """
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_matrix = np.stack([b1, b2, b3], axis=-1)  # (...,3,3)

    if flip_z:
        # 在坐标系上反转 Z 轴
        flip_mat = np.diag([1, 1, -1])
        rot_matrix = flip_mat @ rot_matrix @ flip_mat

    r = R.from_matrix(rot_matrix)
    return r.as_rotvec()

def convert_mdm_to_hg_rotations(input_path, output_path, flip_z=True):
    """
    Convert MDM SMPL motion .npy to HumanGaussian .npz format
    """
    data = np.load(input_path, allow_pickle=True).item()
    motion_6d = data['thetas']              # (joint_num,6,frames)
    root_trans = data['root_translation']   # (3,frames)
    num_frames = int(data['length'])
    joint_num, _, frame_num = motion_6d.shape

    poses_list = []
    for f in range(frame_num):
        frame_rot6d = motion_6d[:, :, f]
        frame_axis_angle = rot6d_to_axis_angle(frame_rot6d, flip_z=flip_z)
        poses_list.append(frame_axis_angle)
    poses = np.stack(poses_list, axis=0)

    # 同步修正根平移
    if flip_z:
        root_trans[2, :] *= -1

    hg_data = {
        "poses": poses.astype(np.float64),
        "trans": root_trans.T.astype(np.float64),
        "betas": np.zeros(10, dtype=np.float64),
        "gender": np.array("male", dtype="<U4"),
        "mocap_framerate": np.array(30, dtype=np.int64)
    }

    np.savez(output_path, **hg_data)
    print(f"✅ Saved converted motion to {output_path} (flip_z={flip_z})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MDM motion file to HumanGaussian SMPL rotations")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npy file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npz file")
    parser.add_argument("--no_flip", action="store_true", help="Disable Z-axis flip")
    args = parser.parse_args()

    convert_mdm_to_hg_rotations(args.input, args.output, flip_z=not args.no_flip)
