import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE = "Qwen/Qwen3.5-2B"
PROMPT = open("prompt.txt").read()


def load(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    out = []
    for i in np.linspace(0, len(frames) - 1, 8).astype(int):
        h, w = frames[i].shape[:2]
        f = cv2.resize(frames[i], (512, round(h * 512 / w)))
        out.append(Image.fromarray(f[..., ::-1]))
    return out


root, out_dir = Path(sys.argv[1]), sys.argv[2]
processor = AutoProcessor.from_pretrained(BASE)
model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.float16).cuda()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()

videos = [(p, 1) for p in (root / "Fight").glob("*")] + [(p, 0) for p in (root / "NonFight").glob("*")]
random.seed(0)
random.shuffle(videos)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-4, total_steps=len(videos) // 8 + 1, pct_start=0.1)
scaler = torch.amp.GradScaler()
model.train()

losses = []
for i, (path, label) in enumerate(videos):
    images = load(path)
    content = [{"type": "text", "text": PROMPT}] + [{"type": "image"}] * len(images)
    prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                           add_generation_prompt=True, enable_thinking=False)
    answer = '{"fight": ' + ("true" if label else "false")
    batch = processor(text=[prompt + answer], images=images, return_tensors="pt").to("cuda")
    k = batch["input_ids"].shape[1] - processor(text=[prompt], images=images, return_tensors="pt")["input_ids"].shape[1]

    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(**batch, logits_to_keep=k + 1).logits[:, :-1].float()
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch["input_ids"][0, -k:])
    scaler.scale(loss / 8).backward()
    losses.append(loss.item())

    if (i + 1) % 8 == 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sched.step()
        if (i + 1) % 80 == 0:
            print(f"{i + 1}/{len(videos)} loss {np.mean(losses[-80:]):.3f}", flush=True)

model = model.merge_and_unload().half()
model.save_pretrained(out_dir)
processor.save_pretrained(out_dir)
print("saved to", out_dir)
