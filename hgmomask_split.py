import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import sys
import subprocess
import re
import time
import glob
import json

# Python 在 fork 后再次使用 tokenizers 并行导致的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 输入参数解析器
def parse_args():
    parser = argparse.ArgumentParser(
        description="Split text prompt and run HG-MoMask pipeline"
    )

    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU id to use (e.g., 0, 1, 2...)"
    )

    parser.add_argument(
        "--text_prompt",
        type=str,
        required=True,
        help='Input text prompt, e.g. "A person walks forward"'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # -------- GPU --------
    use_cuda = torch.cuda.is_available()
    device = f"cuda:{args.gpu_id}" if use_cuda else "cpu"
    gpu_id = args.gpu_id

    # -------- Prompt --------
    original_prompt = args.text_prompt.strip()
    if not original_prompt:
        print(
            "Error: --text_prompt is empty.\n"
            "Usage:\n"
            '  python hgmomask_split.py --gpu_id 1 --text_prompt "A person walks forward"'
        )
        return

    print(f"[INFO] Using device: {device}")
    print(f"[INFO] Prompt: {original_prompt}")

    # ------------------ 1. Prompt 拆分 ------------------

    # 本地路径
    model_dir = "./qwen"

    # 加载 tokenizer 与模型
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        # device_map="auto" if torch.cuda.is_available() else None,      #会默认将所有GPU投入使用
        device_map={"": gpu_id},
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
        "1",    # gpu_id+1也是可以的，指定下一个gpu
        f"system.prompt_processor.prompt={appearance_prompt}"
    ]
    subprocess.run(hg_cmd, check=True, cwd="./HumanGaussian")


    # ------------------ 3. 动作生成 ------------------

    def make_ext_from_text(text, max_len=40):
        safe_text = re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip())
        safe_text = safe_text[:max_len]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return f"{safe_text}_{timestamp}"

    ext_name = make_ext_from_text(action_prompt)

    # 这里的device编号0从3号GPU开始
    motion_cmd = [
        "python",
        "gen_t2m.py",
        "--gpu_id",
        str(gpu_id),
        "--ext",
        ext_name,
        "--text_prompt",
        action_prompt
    ]

    # 继承当前环境变量（MoMask 一般不依赖 HF）
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HUB_URL"] = "https://hf-mirror.com"  # 兼容旧版 HF 库

    subprocess.run(motion_cmd, check=True, cwd="./momask-codes")

    # ------------------ 4. 定位 MoMask 输出 & 进行 SMPL 参数转换渲染 ------------------

    # MoMask 动作生成根目录
    momask_root = os.path.abspath(
        os.path.join(os.getcwd(), "save", "momask", ext_name)
    )

    if not os.path.isdir(momask_root):
        raise RuntimeError(f"MoMask output folder not found: {momask_root}")

    print("[INFO] MoMask output folder:", momask_root)

    # animations/0 目录
    anim_dir = os.path.join(momask_root, "animations", "0")
    joint_dir = os.path.join(momask_root, "joints", "0")

    if not os.path.isdir(anim_dir):
        raise RuntimeError(f"Animations folder not found: {anim_dir}")
    if not os.path.isdir(joint_dir):
        raise RuntimeError(f"Joints folder not found: {joint_dir}")

    # 优先查找 *_ik.mp4
    mp4_candidates = sorted(
        glob.glob(os.path.join(anim_dir, "*_ik.mp4"))
    )

    if not mp4_candidates:
        raise RuntimeError("No *_ik.mp4 found in animations/0")

    mp4_path = mp4_candidates[0]
    print("[INFO] Using IK-optimized MP4:", mp4_path)

    # 对应的 *_ik.npy（用于后续 SMPL / HG 驱动）
    npy_candidates = sorted(
        glob.glob(os.path.join(joint_dir, "*_ik.npy"))
    )

    if not npy_candidates:
        raise RuntimeError("No *_ik.npy found in joints/0")

    npy_path = npy_candidates[0]
    print("[INFO] Using IK-optimized joint file:", npy_path)

    # ------------------ SMPL 渲染命令 ------------------

    smpl_cmd = [
        "python",
        "transformsmpl/render_mesh.py",
        "--input_path",
        mp4_path,
        "--device",
        str(gpu_id)   # 使用你给定的 GPU
    ]

    # MoMask 一般不依赖 HF，但保持环境一致性
    env = os.environ.copy()

    # 在 MoMask 根目录下执行
    subprocess.run(smpl_cmd, check=True, cwd="./momask-codes", env=env)

    print("[INFO] SMPL conversion & rendering finished!")


    # ------------------ 5. 调用 convert_pytorch3d.py 生成 HG 可用的 NPZ ------------------

    # MoMask 当前动作的 animations/0 目录
    anim_dir = os.path.join(
        "save", "momask", ext_name, "animations", "0"
    )

    if not os.path.isdir(anim_dir):
        raise RuntimeError(f"Animations folder not found: {anim_dir}")

    # 查找 *_ik_smpl_params.npy
    smpl_param_candidates = sorted(
        glob.glob(os.path.join(anim_dir, "*_ik_smpl_params.npy"))
    )

    if not smpl_param_candidates:
        raise RuntimeError("No *_ik_smpl_params.npy found in animations/0")

    smpl_npy_path = smpl_param_candidates[0]

    print("[INFO] Found SMPL param file:")
    print("       ", smpl_npy_path)

    # HG motion 输出目录
    hg_content_dir = os.path.abspath(
        os.path.join("save", "HGmodel", "content")
    )
    os.makedirs(hg_content_dir, exist_ok=True)

    # 使用 ext_name 作为动作唯一标识
    npz_output_path = os.path.join(
        hg_content_dir,
        f"{ext_name}_hg_motion.npz"
    )

    print("[INFO] Converting SMPL npy → HG npz:")
    print("       input :", smpl_npy_path)
    print("       output:", npz_output_path)

    # convert_pytorch3d.py 位于与 save/ 同级
    convert_cmd = [
        "python",
        "convert_pytorch3d.py",
        "--input",
        smpl_npy_path,
        "--output",
        npz_output_path
    ]

    subprocess.run(convert_cmd, check=True, cwd=".", env=env)

    print("[INFO] HG npz conversion finished!")


    # ------------------ 6. 调用 HumanGaussian 的 animation.py 进行动画化 ------------------

    # -------- GPU 配置: 编号为3的GPU对应gpu_id=0 --------
    # -------- HumanGaussian 目录 --------
    animate_cwd = "./HumanGaussian"  # animation.py 所在目录

    # -------- 读取 HG launch 导出的实验路径 --------
    hg_meta_path = os.path.join("save", "HGmodel", "last_trial.json")
    if not os.path.isfile(hg_meta_path):
        raise RuntimeError(
            "HG trial metadata not found. "
            "Make sure launch.py exports cfg.trial_dir."
        )

    with open(hg_meta_path, "r") as f:
        hg_meta = json.load(f)

    hg_trial_dir = hg_meta["trial_dir"]  # 绝对路径或相对路径，取决于你导出时

    # -------- 构建 PLY 路径 --------
    ply_path_abs = os.path.join(hg_trial_dir, "save", "last.ply")
    if not os.path.isfile(ply_path_abs):
        raise RuntimeError(f"PLY file not found: {ply_path_abs}")

    # animation.py 在 HumanGaussian/ 下运行，需要相对路径
    ply_path = os.path.relpath(ply_path_abs, start="HumanGaussian")

    print("[INFO] Using HG model:", ply_path)

    # -------- 动作 NPZ（Stage 5 生成） --------
    # 调试用：ext_name="He_picks_up_his_toolbox__20251224_105725"

    motion_npz_abs = os.path.join(
        "save",
        "HGmodel",
        "content",
        f"{ext_name}_hg_motion.npz"
    )

    if not os.path.isfile(motion_npz_abs):
        raise RuntimeError(f"Motion NPZ not found: {motion_npz_abs}")

    motion_npz = os.path.join("..", motion_npz_abs)

    print("[INFO] Using motion NPZ:", motion_npz)

    # -------- 动画输出目录 --------
    save_dir_abs = os.path.abspath(os.path.join("save", "animated"))
    os.makedirs(save_dir_abs, exist_ok=True)

    save_dir = os.path.join("..", save_dir_abs)

    print("[INFO] Animation will be saved to:", save_dir_abs)

    # -------- 构建 animation.py 命令 --------
    animation_cmd = [
        "python",
        "animation.py",
        "--ply", ply_path,
        "--motion", motion_npz,
        "--save", save_dir,
        "--play"
    ]

    # -------- 环境变量 --------
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # -------- 执行动画 --------
    print("[INFO] Running animation.py ...")
    subprocess.run(
        animation_cmd,
        check=True,
        cwd=animate_cwd,
        env=env
    )

    print("[INFO] Animation finished successfully!")

if __name__ == "__main__":
    main()
