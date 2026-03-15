import numpy as np
import os

def convert_mdm_to_hg(input_path, output_dir):
    print(f"Loading MDM motion data from: {input_path}")
    data = np.load(input_path, allow_pickle=True).item()

    motions = data["motion"]  # (N, 22, 3, T)
    num_samples = motions.shape[0]
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_samples):
        motion = motions[i]  # (22, 3, T)
        motion = np.transpose(motion, (2, 0, 1))  # (T, 22, 3)

        # 坐标轴转换: MDM(Y-up,Z-forward) -> HG(Z-up,Y-forward)
        hg_motion = np.zeros_like(motion)
        hg_motion[..., 0] = motion[..., 0]   # x 保持不变
        hg_motion[..., 1] = motion[..., 2]   # z -> y
        hg_motion[..., 2] = -motion[..., 1]  # -y -> z

        # 根节点轨迹作为平移
        root_trans = hg_motion[:, 0, :]  # (T, 3)

        # 生成伪造的SMPL参数
        hg_npz = {
            "poses": np.zeros((hg_motion.shape[0], 55, 3), dtype=np.float32),
            "trans": root_trans.astype(np.float32),
            "betas": np.zeros(10, dtype=np.float32),
            "gender": "male",
            "mocap_framerate": 30,
        }

        save_path = os.path.join(output_dir, f"motion_{i:02d}.npz")
        np.savez(save_path, **hg_npz)
        print(f"Saved: {save_path}")

    print("✅ All motions converted successfully.")

if __name__ == "__main__":
    convert_mdm_to_hg("save/humanml_trans_dec_512_bert/samples_humanml_trans_dec_512_bert_000200000_seed10_the_person_walked_forward_and_is_picking_up_his_toolbox/results.npy", "converted_hg")
