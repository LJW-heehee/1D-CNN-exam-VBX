"""
Train a VectorBlox-friendly sensor CNN on NASA Bearing data, then export ONNX.

Uses X.npy / y.npy from prepare_bearing_vbx.py.
Flow: train -> save .pth -> export ONNX.

GAP uses fixed AvgPool2d (not AdaptiveAvgPool2d) so onnx2tf/TFLite becomes
AVERAGE_POOL_2D -> MEAN axes [1,2], which VectorBlox SDK 3.1 accepts.
AdaptiveAvgPool often becomes MEAN [2,1] and breaks 3.1 compilation.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler


def _encoder_spatial_hw(input_length: int):
    """H,W after three Conv2d(k=(1,3), s=(1,2), p=(0,1)). H stays 1."""
    w = input_length
    for _ in range(3):
        w = (w + 2 * 1 - 3) // 2 + 1
    return 1, w


class VBXSensorModel(nn.Module):
    def __init__(self, input_channels=1, input_length=256):
        super(VBXSensorModel, self).__init__()
        self.input_length = input_length

        # 1D time axis as Conv2D width: kernel (1, k) on input (N, C, H=1, W=256)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.ReLU()
        )
        # Fixed AvgPool -> (B, 64, 1, 1). Prefer over AdaptiveAvgPool for VBX 3.1 MEAN [1,2].
        h_out, w_out = _encoder_spatial_hw(input_length)
        self.gap = nn.AvgPool2d(kernel_size=(h_out, w_out))
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.encoder(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def load_bearing_data(data_dir: str):
    """Load X.npy / y.npy produced by prepare_bearing_vbx.py."""
    data_dir = os.path.abspath(data_dir)
    X_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")

    if not os.path.isfile(X_path) or not os.path.isfile(y_path):
        raise FileNotFoundError(
            f"[-] X.npy or y.npy not found in: {data_dir}\n"
            "Run prepare_bearing_vbx.py first."
        )

    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    # X is already (N, 1, 1, 256) == PyTorch NCHW
    return X, y


def train_and_export(
    data_dir: str = "./bearing_vbx",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    seed: int = 42,
    out_onnx: str = "vbx_sensor_256.onnx",
    out_ckpt: str = "vbx_sensor_256.pth",
    no_train: bool = False,
    class_weight: bool = True,
    oversample: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    model = VBXSensorModel(input_channels=1, input_length=256)

    if no_train:
        # Skip training: load checkpoint and export ONNX only
        if os.path.isfile(out_ckpt):
            model.load_state_dict(torch.load(out_ckpt, map_location="cpu"))
            print(f"[*] Loaded checkpoint: {out_ckpt}")
        model.eval()
    else:
        X, y = load_bearing_data(data_dir)
        print(f"[*] Loaded: X {X.shape}, y {y.shape} (normal={np.sum(y==0)}, anomaly={np.sum(y==1)})")

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

        # Stratified split (keep class ratio)
        rng = np.random.default_rng(seed)
        idx0 = np.flatnonzero(y == 0)
        idx1 = np.flatnonzero(y == 1)
        rng.shuffle(idx0)
        rng.shuffle(idx1)

        n0_val = int(len(idx0) * val_ratio)
        n1_val = int(len(idx1) * val_ratio)

        val_idx = np.concatenate([idx0[:n0_val], idx1[:n1_val]], axis=0)
        train_idx = np.concatenate([idx0[n0_val:], idx1[n1_val:]], axis=0)
        rng.shuffle(val_idx)
        rng.shuffle(train_idx)

        train_ds = Subset(dataset, train_idx.tolist())
        val_ds = Subset(dataset, val_idx.tolist())

        if oversample:
            # Oversample anomaly class
            y_train = y[train_idx]
            w = np.where(y_train == 1, (y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0).astype(np.float64)
            sampler = WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
            print(f"[*] Oversample enabled: train_n={len(train_idx)} val_n={len(val_idx)} "
                  f"(train y0={int((y_train==0).sum())}, y1={int((y_train==1).sum())})")
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            print(f"[*] Oversample disabled: train_n={len(train_idx)} val_n={len(val_idx)}")

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = model.to(device)

        if class_weight:
            y_train_np = y[train_idx]
            n0 = int((y_train_np == 0).sum())
            n1 = int((y_train_np == 1).sum())
            w0 = 1.0
            w1 = (n0 / max(n1, 1)) if n1 > 0 else 1.0
            weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
            print(f"[*] Class weight enabled: n0={n0}, n1={n1}, weight=[{w0:.4f}, {w1:.4f}]")
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Keep best checkpoint by anomaly F1
        best_f1_anom = -1.0
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            model.eval()
            correct, total = 0, 0
            cm00 = cm01 = cm10 = cm11 = 0

            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = model(bx).argmax(dim=1)
                    correct += (pred == by).sum().item()
                    total += by.size(0)

                    for t, p_ in zip(by.tolist(), pred.tolist()):
                        if t == 0 and p_ == 0: cm00 += 1
                        elif t == 0 and p_ == 1: cm01 += 1
                        elif t == 1 and p_ == 0: cm10 += 1
                        elif t == 1 and p_ == 1: cm11 += 1

            val_acc = correct / total if total else 0.0

            prec1 = cm11 / (cm11 + cm01) if (cm11 + cm01) else 0.0
            rec1 = cm11 / (cm11 + cm10) if (cm11 + cm10) else 0.0
            f1_1 = (2 * prec1 * rec1 / (prec1 + rec1)) if (prec1 + rec1) else 0.0

            if f1_1 > best_f1_anom:
                best_f1_anom = f1_1
                torch.save(model.state_dict(), out_ckpt)
                print(f"  [+] checkpoint saved: f1_anom={f1_1:.4f} (prec={prec1:.4f}, rec={rec1:.4f}) val_acc={val_acc:.4f}")

            print(f"Epoch {epoch+1}/{epochs} | Loss={train_loss:.4f} | "
                  f"Val Acc={val_acc:.4f} | F1_Anom={f1_1:.4f} (p={prec1:.4f}, r={rec1:.4f}) | "
                  f"CM=[[{cm00},{cm01}],[{cm10},{cm11}]] | Best_F1={best_f1_anom:.4f}")

        print(f"\n[*] Training done. Best F1_Anom={best_f1_anom:.4f}, saved: {out_ckpt}")
        model.load_state_dict(torch.load(out_ckpt, map_location="cpu"))
        model.eval()

    # ONNX export (NCHW: B, C, H, W)
    model = model.cpu()
    print("\n[*] Exporting ONNX for VectorBlox...")
    dummy_input = torch.randn(1, 1, 1, 256)

    try:
        # dynamo=False: legacy exporter keeps opset 11 (VBX / onnx2tf friendly)
        torch.onnx.export(
            model,
            dummy_input,
            out_onnx,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )
        print(f"[+] Success: {out_onnx} created.")
    except Exception as e:
        print(f"[-] Export Error: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train VBX sensor CNN and export ONNX")
    p.add_argument("--data_dir", type=str, default="./bearing_vbx",
                   help="Directory with X.npy / y.npy from prepare_bearing_vbx.py")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")
    p.add_argument("--out_onnx", type=str, default="vbx_sensor_256.onnx")
    p.add_argument("--out_ckpt", type=str, default="vbx_sensor_256.pth")
    p.add_argument("--no_train", action="store_true", help="Skip training; export ONNX from existing .pth")
    p.add_argument("--no_class_weight", action="store_true", help="Disable class weights")
    p.add_argument("--no_oversample", action="store_true", help="Disable train oversampling")
    args = p.parse_args()

    train_and_export(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        out_onnx=args.out_onnx,
        out_ckpt=args.out_ckpt,
        no_train=args.no_train,
        class_weight=(not args.no_class_weight),
        oversample=(not args.no_oversample),
    )
