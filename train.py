import glob
import random
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import FRAMES, SECONDS, build_model, prepare_frame, to_tensor

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
EPOCHS = 10
BATCH = 16
LR = 1e-4


def find_videos(split):
    items = []
    for label, name in enumerate(["NonFight", "Fight"]):
        for path in glob.glob(f"{DATA}/{split}/{name}/*.avi"):
            items.append((path, label))
    return items


class Videos(Dataset):
    def __init__(self, items, train):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        need = int(SECONDS * fps)
        if len(frames) > need:
            start = random.randint(0, len(frames) - need) if self.train else (len(frames) - need) // 2
            frames = frames[start:start + need]
        idx = np.linspace(0, len(frames) - 1, FRAMES).round().astype(int)
        clip = np.stack([prepare_frame(frames[j]) for j in idx])

        if self.train and random.random() < 0.5:
            clip = clip[:, :, ::-1]
        return to_tensor(clip), label


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


device = "cuda"
train_loader = DataLoader(Videos(find_videos("train"), True), batch_size=BATCH, shuffle=True, num_workers=4)
val_loader = DataLoader(Videos(find_videos("val"), False), batch_size=BATCH, num_workers=4)

model = build_model(pretrained=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
loss_fn = torch.nn.CrossEntropyLoss()

best = 0
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        loss = loss_fn(model(x.to(device)), y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    acc = evaluate(model, val_loader, device)
    print(f"epoch {epoch + 1}: val acc {acc:.4f}")
    if acc > best:
        best = acc
        torch.save({"model_state": model.state_dict(), "class_to_idx": {"NonFight": 0, "Fight": 1},
                    "window_sec": SECONDS, "epoch": epoch}, "models/my_model.pt")
