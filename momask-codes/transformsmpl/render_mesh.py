import argparse
import os
import shutil
import glob
import numpy as np
from tqdm import tqdm
import vis_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="MoMask generated mp4 file, e.g. generation/exp1/animations/0/sample0_repeat0_len196_ik.mp4"
    )
    parser.add_argument("--cuda", type=bool, default=True)
    parser.add_argument("--device", type=int, default=0)
    params = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. 基本检查
    # ------------------------------------------------------------------
    assert params.input_path.endswith('.mp4')
    assert os.path.exists(params.input_path)

    mp4_path = params.input_path
    basename = os.path.splitext(os.path.basename(mp4_path))[0]

    # ------------------------------------------------------------------
    # 2. 自动定位 exp 目录
    # ------------------------------------------------------------------
    anim_dir = os.path.dirname(mp4_path)           # .../animations/0
    exp_dir = os.path.dirname(os.path.dirname(anim_dir))  # .../exp1

    joints_root = os.path.join(exp_dir, "joints")
    assert os.path.exists(joints_root), f"Cannot find joints directory: {joints_root}"

    # 默认取 joints 下的第一个子目录（通常是 0）
    joint_subdir = sorted(os.listdir(joints_root))[0]
    joints_dir = os.path.join(joints_root, joint_subdir)

    # 找到对应的 joints npy
    npy_candidates = glob.glob(os.path.join(joints_dir, f"{basename}.npy"))
    if len(npy_candidates) == 0:
        raise FileNotFoundError(f"Cannot find joints npy for {basename} in {joints_dir}")
    joints_npy_path = npy_candidates[0]

    # ------------------------------------------------------------------
    # 3. 读取 joints，并生成完整 results.npy 格式
    # ------------------------------------------------------------------
    joints = np.load(joints_npy_path)   # (T, J, 3)
    assert joints.ndim == 3, f"Unexpected joints shape: {joints.shape}"
    T, J, _ = joints.shape

    # 生成 transformsmpl 兼容的 motion 格式
    motion = joints.transpose(1, 2, 0)[None, ...]  # (1, J, 3, T)
    results_dict = {
        "motion": motion.astype(np.float32),
        "lengths": np.array([T], dtype=np.int32),
        "text": ["placeholder text"],  # 可改为实际描述
        "num_samples": 1,
        "num_repetitions": 1
    }

    tmp_results_path = os.path.join(exp_dir, f"{basename}_results_tmp.npy")
    np.save(tmp_results_path, results_dict)

    sample_i = 0
    rep_i = 0

    # ------------------------------------------------------------------
    # 4. 输出路径
    # ------------------------------------------------------------------
    out_npy_path = mp4_path.replace('.mp4', '_smpl_params.npy')
    results_dir = mp4_path.replace('.mp4', '_obj')

    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.makedirs(results_dir)

    # ------------------------------------------------------------------
    # 5. 调用原有 SMPL 回归与导出逻辑
    # ------------------------------------------------------------------
    npy2obj = vis_utils.npy2obj(
        tmp_results_path,
        sample_i,
        rep_i,
        device=params.device,
        cuda=params.cuda
    )

    print(f"Saving obj files to [{os.path.abspath(results_dir)}]")
    for frame_i in tqdm(range(npy2obj.real_num_frames)):
        npy2obj.save_obj(
            os.path.join(results_dir, f'frame{frame_i:03d}.obj'),
            frame_i
        )

    print(f"Saving SMPL params to [{os.path.abspath(out_npy_path)}]")
    npy2obj.save_npy(out_npy_path)

    # ------------------------------------------------------------------
    # 6. 清理临时文件
    # ------------------------------------------------------------------
    if os.path.exists(tmp_results_path):
        os.remove(tmp_results_path)

    print("Done.")
