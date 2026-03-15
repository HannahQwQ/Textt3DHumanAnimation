from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import sys
import subprocess
import re
import glob

# Python 在 fork 后再次使用 tokenizers 并行导致的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------ 1. Prompt 拆分 ------------------

# 本地路径
model_dir = "./qwen"

# 指定环境中可见的GPU，单GPU内存足够不会OOM
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 加载 tokenizer 与模型
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
device = "cuda:1"
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    # device_map="auto" if torch.cuda.is_available() else None,      #会默认将所有GPU投入使用
    device_map={"": 1},    # 明确指定放在 GPU 1
    low_cpu_mem_usage=True,
)

# 输入原始 prompt
# original_prompt = "A woman wearing a short jean skirt, a cropped top, and a white sneaker performs a crazy dance move."
if len(sys.argv) > 1:
    original_prompt = " ".join(sys.argv[1:]).strip()
else:
    original_prompt = input("请输入要拆分的英文 prompt:\n> ").strip()

if not original_prompt:
    print("错误：输入为空，请重新运行脚本。")
    exit()

# 让模型进行拆分
task_prompt = (
    f"""
    Split the prompt into:

    Sentence 1 (appearance):
    - Keep ONLY nouns, adjectives, and prepositional phrases describing the person.
    - Do NOT add, invent, or replace any words.
    - REMOVE any verbs and any words describing objects or items that are not part of the person's appearance.

    Sentence 2 (action):
    - Keep ONLY the verbs and their objects describing what the person is doing.
    - REMOVE all appearance words.
    - Use ONLY words from the original prompt.

    No explainations.

    Reconstruct two complete phrases/sentences without adding any new words.

    Output exactly:
    Sentence 1: ...
    Sentence 2: ...

    Prompt: {original_prompt}

    Output:
    """
)


# 编码输入
inputs = tokenizer(task_prompt, return_tensors="pt").to(model.device)

# 生成输出
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.0001,   #temperature > 0 会增加随机性和创意
        top_p=1.0,         #top_p < 1 会让模型倾向选择概率较高的“合理”补全
    )

# 解码输出
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

# 从模型输出解析两个句子
appearance_prompt = None
action_prompt = None
for line in result.splitlines():
    line = line.strip()
    if line.lower().startswith("sentence 1:"):
        appearance_prompt = line[len("sentence 1:"):].strip()
    elif line.lower().startswith("sentence 2:"):
        action_prompt = line[len("sentence 2:"):].strip()

# 确认已经获取到最后一次赋值
if not appearance_prompt or not action_prompt:
    raise ValueError("无法解析拆分结果")

# ========== 释放显存 ==========
del model
del tokenizer

# torch.cuda.empty_cache()
# torch.cuda.synchronize()
# =============================


# ------------------ 2. 人体生成 ------------------

# GPU编号与服务器一致
hg_cmd = [
    "python",
    "launch.py",
    "--config",
    "configs/test.yaml",
    "--train",
    "--gpu",
    "2",    
    f"system.prompt_processor.prompt={appearance_prompt}"
]
subprocess.run(hg_cmd, check=True, cwd="./HumanGaussian")


# ------------------ 3. 动作生成 ------------------

# 这里的device编号0从3号GPU开始
motion_cmd = [
    "python",
    "-m",
    "sample.generate",
    "--model_path",
    "./save/humanml_trans_dec_512_bert/model000200000.pt",  
    "--device",
    "1",
    f"--text_prompt={action_prompt}"
]

# 环境变量传递：给子进程传递 HF 镜像环境变量
env = os.environ.copy()
env["HF_ENDPOINT"] = "https://hf-mirror.com"
env["HF_HUB_URL"] = "https://hf-mirror.com"  # 兼容旧版 HF 库

subprocess.run(motion_cmd, check=True, cwd="./motion-diffusion-model", env=env )


# ------------------ 4. 找到动作生成输出视频 & 进行SMPL参数的转换渲染------------------

# 1. 动作生成输出根目录
mdm_root = os.path.abspath(os.path.join(os.getcwd(), 'save', 'mdm'))

# 2. 找到最新生成的动作目录（按时间排序）
subdirs = [d for d in glob.glob(os.path.join(mdm_root, '*')) if os.path.isdir(d)]
if not subdirs:
    raise RuntimeError("No MDM output folders found under save/mdm/")
latest_dir = max(subdirs, key=os.path.getmtime)
print("[INFO] Latest MDM folder:", latest_dir)

# 3. 找 samples_00_to_00.mp4
mp4_path = os.path.join(latest_dir, "samples_00_to_00.mp4")
if not os.path.isfile(mp4_path):
    raise RuntimeError(f"File not found: {mp4_path}")
print("[INFO] MP4 for SMPL conversion:", mp4_path)

# 4. 构建 SMPL 渲染命令
smpl_cmd = [
    "python",
    "-m",
    "visualize.render_mesh",
    "--device",
    "1",  # 希望使用的 GPU 编号，从3号显卡开始对应
    "--input_path",
    mp4_path
]


# # 环境变量传递：给子进程传递 HF 镜像环境变量
env = os.environ.copy()
env["HF_ENDPOINT"] = "https://hf-mirror.com"
env["HF_HUB_URL"] = "https://hf-mirror.com"  # 兼容旧版 HF 库

# 5. 在 motion-diffusion-model 下执行
subprocess.run(smpl_cmd, check=True, cwd="./motion-diffusion-model", env=env)
print("[INFO] SMPL conversion finished!")


# ------------------5. 自动调用 convert_pytorch3d.py 生成 NPZ 文件-------------------

# 找到刚生成的 SMPL npy
smpl_npy_path = mp4_path.replace(".mp4", "_smpl_params.npy")

if not os.path.isfile(smpl_npy_path):
    raise RuntimeError(f"SMPL params .npy not found: {smpl_npy_path}")

# NPZ 输出名称
# HG 内容目录
hg_content_dir = os.path.abspath(os.path.join("save", "HGmodel", "content"))
os.makedirs(hg_content_dir, exist_ok=True)

# 使用 prompt 生成安全的动作名称
latest_dir = os.path.dirname(mp4_path)  # /.../Flying_a_kite@20251117-164517
base_dirname = os.path.basename(latest_dir)  # Flying_a_kite@20251117-164517

npz_output_path = os.path.join(
    "save", "HGmodel", "content", base_dirname + "_hg_motion.npz"
)

print("[INFO] Converting SMPL npy → HG npz:")
print("       input :", smpl_npy_path)
print("       output:", npz_output_path)

convert_cmd = [
    "python",
    "convert_pytorch3d.py",
    "--input", smpl_npy_path,
    "--output", npz_output_path
]

subprocess.run(convert_cmd, check=True, cwd=".", env=env)

print("[INFO] HG npz conversion finished!")


# ------------------6. 调用HG的animation脚本进行动画化-------------------

# -------- 配置 GPU --------
gpu_id = "4"  # 可根据需要修改

# -------- HumanGaussian 目录 --------
animate_cwd = "./HumanGaussian"  # animation.py 所在目录

# -------- 自动找到最新的 HG PLY 文件 --------
hg_root = os.path.join("save", "HGmodel")  # 你的 HGmodel 根目录
model_root = os.path.join(hg_root, "name-of-this-experiment-run")
hg_subdirs = [d for d in glob.glob(os.path.join(model_root, '*')) if os.path.isdir(d)]
if not hg_subdirs:
    raise RuntimeError(f"No HGmodel subfolders found under {model_root}")

latest_hg_dir = max(hg_subdirs, key=os.path.getmtime)
ply_path = os.path.join(latest_hg_dir, "save", "last.ply")

if not os.path.isfile(ply_path):
    raise RuntimeError(f"PLY file not found: {ply_path}")

ply_path = os.path.join("..", latest_hg_dir, "save", "last.ply")

print(f"[INFO] Using latest HG model: {ply_path}")

# -------- 自动找到最新的动作 NPZ 文件 --------
motion_root = os.path.join(hg_root, "content")
motion_files = glob.glob(os.path.join(motion_root, "*.npz"))
if not motion_files:
    raise RuntimeError(f"No NPZ files found under {motion_root}")

latest_motion = max(motion_files, key=os.path.getmtime)

latest_motion = os.path.join("..", latest_motion)
print(f"[INFO] Using latest motion NPZ: {latest_motion}")

save_dir = os.path.abspath(os.path.join(animate_cwd, "../save/animated"))

# -------- 构建 animation.py 命令 --------
animation_cmd = [
    "python",
    "animation.py",
    "--ply", ply_path,
    "--motion", latest_motion,
    "--save", save_dir,
    "--play"
]

# -------- 环境变量 --------
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = gpu_id

# -------- 输出路径提示 --------
print("[INFO] Animation will be saved to:", save_dir)

# -------- 最终执行动画子进程 --------
print("[INFO] Running animation.py ...")
subprocess.run(animation_cmd, check=True, cwd=animate_cwd, env=env)
print("[INFO] Animation finished successfully!")
