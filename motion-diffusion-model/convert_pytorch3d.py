import numpy as np
import torch
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
import argparse

def rot6d_to_axis_angle(rot6d):
    """
    Convert 6D rotation representation to axis-angle (3D) vector using PyTorch3D.
    Always flips the Z-axis to match HumanGaussian coordinate.
    """
    # Convert to torch tensor
    rot6d_t = torch.tensor(rot6d, dtype=torch.float32)  # (...,6)

    # (joint_num,6) -> (joint_num,3,3)
    rot_matrix = rotation_6d_to_matrix(rot6d_t)

    # Always flip Z axis
    # flip_mat = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
    # rot_matrix = flip_mat @ rot_matrix @ flip_mat

    # Convert to axis-angle representation (...,3)
    rot_axis_angle = matrix_to_axis_angle(rot_matrix)

    return rot_axis_angle.numpy()

def convert_mdm_to_hg_rotations(input_path, output_path):
    """
    Convert MDM SMPL motion .npy to HumanGaussian .npz format.
    Always applies Z-axis flip.
    """
    data = np.load(input_path, allow_pickle=True).item()
    motion_6d = data['thetas']              # (joint_num,6,frames)
    root_trans = data['root_translation']   # (3,frames)
    joint_num, _, frame_num = motion_6d.shape

    poses_list = []
    for f in range(frame_num):
        frame_rot6d = motion_6d[:, :, f]
        frame_axis_angle = rot6d_to_axis_angle(frame_rot6d)
        poses_list.append(frame_axis_angle)
    poses = np.stack(poses_list, axis=0)

    # Flip root translation Z component
    # root_trans[2, :] *= -1

    hg_data = {
        "poses": poses.astype(np.float64),
        "trans": root_trans.T.astype(np.float64),
        "betas": np.zeros(10, dtype=np.float64),
        "gender": np.array("male", dtype="<U4"),
        "mocap_framerate": np.array(30, dtype=np.int64)
    }

    np.savez(output_path, **hg_data)
    print(f"✅ Saved converted motion to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MDM motion file to HumanGaussian SMPL rotations")
    parser.add_argument("--input", type=str, required=True, help="Path to input .npy file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npz file")
    args = parser.parse_args()

    convert_mdm_to_hg_rotations(args.input, args.output)
