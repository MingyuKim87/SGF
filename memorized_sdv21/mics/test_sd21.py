# sd21_smoketest.py
import torch
from pathlib import Path
from diffusers import DiffusionPipeline

# 🔒 로컬 모델 루트(여기에 model_index.json이 있어야 함)
MODEL_DIR = "/home/mgyukim/scratch/hf_models/sd21"

pipe = DiffusionPipeline.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    safety_checker=None,
    local_files_only=True,   # 허브 접속 시도 금지
    use_safetensors=True,
)
pipe = pipe.to("cuda")

prompt = "a photo of tench"
g = torch.Generator(device="cuda").manual_seed(42)
img = pipe(prompt, height=512, width=512, num_inference_steps=30,
           guidance_scale=7.5, generator=g).images[0]

out = Path("sd21_smoketest.png")
img.save(out)
print(f"[OK] saved -> {out.resolve()}")

args.pretrained_model_name_or_path