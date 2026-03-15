#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import argparse

def run_command(cmd, cwd=None, env=None):
    """Run a shell command and check return code."""
    print(f"\n[Running] {cmd} (cwd={cwd})")
    result = subprocess.run(cmd, shell=True, cwd=cwd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def split_prompt(full_prompt):
    """
    Split full prompt into appearance and action prompts.
    Here we assume 'performs' marks the action part.
    """
    if "performs" in full_prompt:
        parts = full_prompt.split("performs", 1)
        appearance = parts[0].strip()
        action = "A person performs" + parts[1].strip()
    else:
        # fallback: everything is appearance
        appearance = full_prompt
        action = "A person performs an action."
    return appearance, action

def main():
    parser = argparse.ArgumentParser(description="Three-stage prompt automation script")
    parser.add_argument("--prompt", type=str, required=True, help="Full input prompt")
    parser.add_argument("--output_root", type=str, default="./output", help="Root output folder")
    parser.add_argument("--gpu_launch", type=str, default="2", help="GPU for launch.py")
    parser.add_argument("--gpu_anim", type=str, default="4", help="GPU for animation.py")
    args = parser.parse_args()

    # ===== Define project directories =====
    projects = {
        "launch": "/path/to/project1",       # launch.py 所在目录
        "generate": "/path/to/project2",     # sample.generate 所在目录
        "animation": "/path/to/project3",    # animation.py 所在目录
    }

    # ===== Create output directories =====
    os.makedirs(args.output_root, exist_ok=True)
    stage1_out = os.path.join(args.output_root, "stage1_launch")
    stage2_out = os.path.join(args.output_root, "stage2_generate")
    stage3_out = os.path.join(args.output_root, "stage3_animation")
    for d in [stage1_out, stage2_out, stage3_out]:
        os.makedirs(d, exist_ok=True)

    # ===== Split prompt =====
    appearance_prompt, action_prompt = split_prompt(args.prompt)
    print(f"[Prompt split] Appearance: {appearance_prompt}")
    print(f"[Prompt split] Action: {action_prompt}")

    # ===== Stage 1: launch.py =====
    cmd1 = f'python launch.py --config configs/test.yaml --train --gpu {args.gpu_launch} "system.prompt_processor.prompt= {appearance_prompt}"'
    run_command(cmd1, cwd=projects["launch"])

    # ===== Stage 2: sample.generate =====
    # 默认model_path可以自己修改
    model_path = os.path.join(projects["generate"], "save/humanml_trans_dec_512_bert/model000200000.pt")
    cmd2 = f'python -m sample.generate --model_path "{model_path}" --text_prompt "{action_prompt}"'
    run_command(cmd2, cwd=projects["generate"])

    # ===== Stage 3: animation.py =====
    # 假设stage2生成的PLY和motion文件位于stage2_out
    ply_path = os.path.join(stage2_out, "women.ply")            # 需要根据实际生成文件修改
    motion_path = os.path.join(stage2_out, "smpl_walk_motion.npz")  # 需要根据实际生成文件修改
    env_anim = os.environ.copy()
    env_anim["CUDA_VISIBLE_DEVICES"] = args.gpu_anim
    cmd3 = f'python animation.py --ply "{ply_path}" --motion "{motion_path}" --play'
    run_command(cmd3, cwd=projects["animation"], env=env_anim)

    print("\nAll stages completed successfully!")

if __name__ == "__main__":
    main()
