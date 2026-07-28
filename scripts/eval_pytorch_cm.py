#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class VBXSensorModel(nn.Module):
    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def main():
    ap = argparse.ArgumentParser(description="Compute confusion matrix for VBXSensorModel on X.npy/y.npy")
    ap.add_argument("--x", default="bearing_vbx/X.npy")
    ap.add_argument("--y", default="bearing_vbx/y.npy")
    ap.add_argument("--ckpt", default="vbx_sensor_256.pth")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    x_path = Path(args.x)
    y_path = Path(args.y)
    ckpt_path = Path(args.ckpt)

    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    print(f"X {X.shape} {X.dtype}  y {y.shape} {y.dtype}")
    print(f"y0={(y==0).sum()} y1={(y==1).sum()}")

    model = VBXSensorModel()
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    cm = [[0, 0], [0, 0]]
    pred_cnt = [0, 0]
    true_cnt = [0, 0]
    total = 0
    correct = 0

    with torch.no_grad():
        for i in range(0, X.shape[0], args.batch):
            xb = torch.from_numpy(np.asarray(X[i : i + args.batch], dtype=np.float32))
            yb = torch.from_numpy(np.asarray(y[i : i + args.batch], dtype=np.int64))
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
            for t, p in zip(yb.tolist(), pred.tolist()):
                cm[t][p] += 1
                true_cnt[t] += 1
                pred_cnt[p] += 1

    acc = correct / total if total else 0.0
    print(f"total={total} acc={acc:.6f} ({correct}/{total})")
    print(f"cm(rows=true, cols=pred)={cm}")
    print(f"true_cnt={true_cnt} pred_cnt={pred_cnt}")


if __name__ == "__main__":
    main()

