import numpy as np
import os

# ==== 设置文件路径 ====
path = "hy2hg_toolbox.npz"

if not os.path.exists(path):
    print(f"[Error] Motion file not found: {path}")
    exit()

# ==== 载入数据 ====
print(f"\n========== Inspecting HumanGaussian motion file ==========")
print(f"Path: {path}\n")

data = np.load(path, allow_pickle=True)
print(f"Keys in file: {list(data.keys())}\n")

poses = data["poses"]       # (frames, joints, 3)
trans = data["trans"]       # (frames, 3)
betas = data["betas"]       #
print(f"Poses shape: {poses.shape}")
print(f"Trans shape: {trans.shape}")
print(f"Betas shape: {betas.shape}")
print("============================================================\n")

# ==== 查看一帧的所有关节 ====
# 修改这个参数即可选择不同帧
frame_id = 0  # 比如第0帧、第10帧、第50帧
if frame_id >= poses.shape[0]:
    print(f"[Error] frame_id {frame_id} 超出范围 (共有 {poses.shape[0]} 帧)")
    exit()

print(f"========== Frame {frame_id} Joint Coordinates ==========")
frame_pose = poses[frame_id]  # shape = (joints, 3)

for j, (x, y, z) in enumerate(frame_pose):
    print(f"Joint {j:02d}:  x={x: .6f},  y={y: .6f},  z={z: .6f}")

print("============================================================\n")

# ==== 可选：输出 root 平移量 ====
root_trans = trans[frame_id]
print(f"Root translation (frame {frame_id}): x={root_trans[0]:.6f}, y={root_trans[1]:.6f}, z={root_trans[2]:.6f}")
print("============================================================\n")
