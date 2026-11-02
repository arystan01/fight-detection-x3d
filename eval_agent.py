import sys
from pathlib import Path

import cv2
import numpy as np

from agent import THRESHOLD, ask_vlm, crops, score
from model import FRAMES, SECONDS, STEP, load_model


def read(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames, fps


def first_alert(model, device, frames, fps, th):
    times = np.arange(len(frames)) / fps
    small = [crops(f) for f in frames]
    scores = []
    for end in np.arange(SECONDS, times[-1] + 1e-6, STEP):
        idx = np.clip(np.searchsorted(times, np.linspace(end - SECONDS, end, FRAMES)), 0, len(frames) - 1)
        scores.append(score(model, device, [small[i] for i in idx]))
        if sum(s >= th for s in scores[-5:]) >= 3:
            return end
    return None


def show(name, preds, labels):
    tp = sum(p and l for p, l in zip(preds, labels))
    fp = sum(p and not l for p, l in zip(preds, labels))
    fn = sum(l and not p for p, l in zip(preds, labels))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    print(f"{name:10} precision {prec:.1%}  recall {rec:.1%}  f1 {2 * prec * rec / (prec + rec):.1%}  (tp {tp} fp {fp} fn {fn})")


root = Path(sys.argv[1])
th = float(sys.argv[2]) if len(sys.argv) > 2 else THRESHOLD
model, device = load_model()

labels, x3d, agent = [], [], []
for label, folder in [(1, "Fight"), (0, "NonFight")]:
    for path in sorted((root / folder).glob("*")):
        frames, fps = read(path)
        if len(frames) / fps < SECONDS:
            continue
        t = first_alert(model, device, frames, fps, th)
        ok = t is not None and ask_vlm(frames[max(0, int((t - 4) * fps)):int((t + 1) * fps)]).get("fight")
        labels.append(label)
        x3d.append(t is not None)
        agent.append(bool(ok))

show("x3d", x3d, labels)
show("x3d + vlm", agent, labels)
