import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorchvideo.models.hub import x3d_m

WEIGHTS = "models/fight_x3d_m.pt"
FRAMES = 16
SECONDS = 2.0
STEP = 0.5
THRESHOLD = 0.6


def build_model(pretrained=False):
    model = x3d_m(pretrained=pretrained)
    head = model.blocks[-1]
    head.proj = nn.Linear(head.proj.in_features, 2)
    head.activation = nn.Identity()
    return model


def load_model(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), device


def prepare_frame(frame):
    h, w = frame.shape[:2]
    scale = 256 / min(h, w)
    frame = cv2.resize(frame, (max(224, round(w * scale)), max(224, round(h * scale))), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    y, x = (h - 224) // 2, (w - 224) // 2
    return frame[y:y + 224, x:x + 224]


def read_video(path, max_seconds=120):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        raise ValueError("can't read video")
    frames = []
    while len(frames) < max_seconds * fps:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(prepare_frame(frame))
    cap.release()
    if len(frames) / fps < SECONDS:
        raise ValueError("video is shorter than 2 seconds")
    return frames, fps


def to_tensor(clip):
    x = torch.from_numpy(np.ascontiguousarray(clip[..., ::-1])).float() / 255
    x = (x - 0.45) / 0.225
    return x.permute(3, 0, 1, 2)


@torch.inference_mode()
def check_video(model, device, path, threshold=THRESHOLD):
    frames, fps = read_video(path)
    times = np.arange(len(frames)) / fps
    ends = np.arange(SECONDS, times[-1] + 1e-6, STEP)

    clips = []
    for end in ends:
        wanted = np.linspace(end - SECONDS, end, FRAMES)
        idx = np.clip(np.searchsorted(times, wanted), 0, len(frames) - 1)
        clips.append(to_tensor(np.stack([frames[i] for i in idx])))

    scores = []
    for i in range(0, len(clips), 8):
        batch = torch.stack(clips[i:i + 8]).to(device)
        scores += torch.softmax(model(batch), 1)[:, 1].tolist()

    alert_at = None
    for i in range(len(scores)):
        if sum(s >= threshold for s in scores[max(0, i - 4):i + 1]) >= 3:
            alert_at = float(ends[i])
            break

    return ends.tolist(), scores, alert_at
