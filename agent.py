import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import requests
import torch
from openai import OpenAI

from model import FRAMES, SECONDS, STEP, load_model, to_tensor

THRESHOLD = 0.3
COOLDOWN = 60
PROMPT = open("prompt.txt").read()
TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT = os.getenv("TG_CHAT")

vlm = OpenAI(base_url=os.getenv("VLM_URL", "http://localhost:8000/v1"), api_key="none")


def crops(frame):
    h, w = frame.shape[:2]
    s = 224 / min(h, w)
    frame = cv2.resize(frame, (max(224, round(w * s)), max(224, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    if w >= h:
        return [frame[:224, x:x + 224] for x in (0, (w - 224) // 2, w - 224)]
    return [frame[y:y + 224, :224] for y in (0, (h - 224) // 2, h - 224)]


@torch.inference_mode()
def score(model, device, clip):
    x = torch.stack([to_tensor(np.stack([c[i] for c in clip])) for i in range(3)]).to(device)
    return torch.softmax(model(x), 1)[:, 1].max().item()


def ask_vlm(frames):
    content = [{"type": "text", "text": PROMPT}]
    for i in np.linspace(0, len(frames) - 1, 8).astype(int):
        h, w = frames[i].shape[:2]
        f = cv2.resize(frames[i], (512, round(h * 512 / w)))
        img = base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1]).decode()
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}})

    name = os.getenv("VLM_MODEL") or vlm.models.list().data[0].id
    text = vlm.chat.completions.create(
        model=name,
        messages=[{"role": "user", "content": content}],
        max_tokens=200,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    ).choices[0].message.content

    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group()) if m else {"fight": True, "description": text}


def save_clip(frames, fps, path):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


def check(source, t, frames, fps):
    try:
        res = ask_vlm(frames[-int(fps * 5):])
    except Exception as e:
        res = {"fight": True, "description": f"vlm error {e}"}

    with open("alerts.jsonl", "a") as f:
        f.write(json.dumps({"source": source, "t": round(t, 1), "time": time.ctime(), **res}, ensure_ascii=False) + "\n")

    if not res.get("fight"):
        print(f"[{t:.1f}s] false alarm: {res.get('description')}")
        return

    print(f"[{t:.1f}s] FIGHT: {res.get('description')}")
    os.makedirs("alerts", exist_ok=True)
    path = f"alerts/{int(time.time())}.mp4"
    save_clip(frames, fps, path)

    if TG_TOKEN and TG_CHAT:
        with open(path, "rb") as f:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendVideo",
                          data={"chat_id": TG_CHAT, "caption": f"🔴 Драка: {source}\n{res.get('description')}"},
                          files={"video": f}, timeout=60)


def watch(source):
    model, device = load_model()
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not 0 < fps < 120:
        fps = 25
    print(f"watching {source}")

    raw = deque(maxlen=int(fps * 8))
    small = deque(maxlen=int(fps * SECONDS) + 1)
    scores = deque(maxlen=5)
    n = 0
    last = -COOLDOWN

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        raw.append(frame)
        small.append(crops(frame))
        t = n / fps
        if t < SECONDS or n % round(fps * STEP):
            continue

        idx = np.linspace(0, len(small) - 1, FRAMES).astype(int)
        scores.append(score(model, device, [small[i] for i in idx]))

        if sum(s >= THRESHOLD for s in scores) >= 3 and t - last > COOLDOWN:
            last = t
            print(f"[{t:.1f}s] x3d thinks fight, asking vlm")
            threading.Thread(target=check, args=(source, t, list(raw), fps)).start()


if __name__ == "__main__":
    for src in sys.argv[1:]:
        threading.Thread(target=watch, args=(src,)).start()
