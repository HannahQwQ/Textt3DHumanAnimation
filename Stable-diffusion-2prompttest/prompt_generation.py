import os
from pathlib import Path
from diffusers import StableDiffusionPipeline
import torch

# 手动输入 prompt
prompt = input("input prompt: \n ").strip()

# 合法化文件名
prompt_tag = "_".join(prompt.lower().split())
save_dir = "./saved"
os.makedirs(save_dir, exist_ok=True)
filename = f"{prompt_tag}.png"

# 模型路径（你可替换为 HuggingFace 名字或本地路径）
model_path = "./models/hf/stable-diffusion-2-base"

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

# 设置 negative prompt（来自你的配置）
negative_prompt = (
    "shadow, dark face, colorful hands, eyeglass, glasses, "
    "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), "
    "text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, "
    "morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, "
    "deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, "
    "gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, "
    "too many fingers, long neck"
)

# 生成图像
image = pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=7.5).images[0]

# 保存图像
save_path = os.path.join(save_dir, filename)
image.save(save_path)

print(f"generation saved as {save_path}")
