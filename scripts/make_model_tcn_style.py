#!/usr/bin/env python3
"""
TCN-style CNN for VBX (RNN-replacement methodology sample)
- Instead of LSTM/GRU, capture past context via "temporal Conv1D + dilation"
- Per VBX OPS.md: only CONV_2D, RELU, ADD, FULLY_CONNECTED
- Input: (B, 1, 1, 256) → temporal Conv2d → GAP → FC
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.utils.data.sampler import WeightedRandomSampler


class VBXTCNStyleModel(nn.Module):
    """
    TCN-style CNN for VBX (RNN-replacement methodology)
    - Temporal processing via Conv2d (kernel=(1, k), with dilation)
    - Residual connection (ADD op)
    - VBX-supported ops only: CONV_2D, ADD, RELU, FULLY_CONNECTED
    """
    def __init__(self, input_channels=1, hidden_channels=32, num_blocks=3):
        super().__init__()
        self.input_channels = input_channels
        
        # Initial projection
        self.input_conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=(1, 3), 
                                     stride=(1, 1), padding=(0, 1))
        
        # TCN-style blocks: dilated conv + residual
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i  # 1, 2, 4, ...
            self.blocks.append(
                TCNBlock(hidden_channels, hidden_channels, kernel_size=3, dilation=dilation)
            )
        
        # Global pooling + classifier
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(hidden_channels, 2)
    
    def forward(self, x):
        # x: (B, 1, 1, 256)
        x = self.input_conv(x)  # (B, hidden, 1, 256)
        
        for block in self.blocks:
            x = block(x)  # residual connection inside
        
        x = self.gap(x)  # (B, hidden, 1, 1)
        x = torch.flatten(x, 1)  # (B, hidden)
        x = self.fc(x)  # (B, 2)
        return x


class TCNBlock(nn.Module):
    """
    TCN-style residual block
    - dilated conv + ReLU + residual add
    - VBX ops: CONV_2D, RELU, ADD
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        # Temporal conv with dilation (expand along time axis only)
        padding = (dilation * (kernel_size - 1)) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=(1, kernel_size),
            stride=(1, 1),
            padding=(0, padding),
            dilation=(1, dilation)
        )
        self.relu = nn.ReLU()
        
        # Residual projection (if needed)
        self.residual = nn.Identity() if in_channels == out_channels else \
                        nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        res = self.residual(x)
        return out + res  # ADD op (VBX supported)


def load_bearing_data(data_dir: str):
    data_dir = os.path.abspath(data_dir)
    X_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")
    if not os.path.isfile(X_path) or not os.path.isfile(y_path):
        raise FileNotFoundError(f"X.npy/y.npy not found: {data_dir}")
    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    return X, y


def train_and_export(
    data_dir: str = "./bearing_vbx",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    seed: int = 42,
    out_onnx: str = "vbx_tcn_256.onnx",
    out_ckpt: str = "vbx_tcn_256.pth",
    no_train: bool = False,
    class_weight: bool = True,
    oversample: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    model = VBXTCNStyleModel(input_channels=1, hidden_channels=32, num_blocks=3)

    if no_train:
        if os.path.isfile(out_ckpt):
            model.load_state_dict(torch.load(out_ckpt, map_location="cpu"))
            print(f"Loaded checkpoint: {out_ckpt}")
        model.eval()
    else:
        X, y = load_bearing_data(data_dir)
        print(f"Data loaded: X {X.shape}, y {y.shape} (normal={np.sum(y==0)}, anomaly={np.sum(y==1)})")

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        
        # Stratified split
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
            y_train = y[train_idx]
            w = np.where(y_train == 1, (y_train == 0).sum() / max((y_train == 1).sum(), 1), 1.0).astype(np.float64)
            sampler = WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=len(w), replacement=True)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
            print(f"oversample enabled: train_n={len(train_idx)} val_n={len(val_idx)}")
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            print(f"oversample disabled: train_n={len(train_idx)} val_n={len(val_idx)}")

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = model.to(device)
        
        if class_weight:
            y_train_np = y[train_idx]
            n0 = int((y_train_np == 0).sum())
            n1 = int((y_train_np == 1).sum())
            w0 = 1.0
            w1 = (n0 / max(n1, 1)) if n1 > 0 else 1.0
            weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
            print(f"class_weight enabled: n0={n0}, n1={n1}, weight=[{w0:.4f},{w1:.4f}]")
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            correct = 0
            total = 0
            cm00 = cm01 = cm10 = cm11 = 0
            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    pred = model(bx).argmax(dim=1)
                    correct += (pred == by).sum().item()
                    total += by.size(0)
                    for t, p_ in zip(by.tolist(), pred.tolist()):
                        if t == 0 and p_ == 0:
                            cm00 += 1
                        elif t == 0 and p_ == 1:
                            cm01 += 1
                        elif t == 1 and p_ == 0:
                            cm10 += 1
                        elif t == 1 and p_ == 1:
                            cm11 += 1
            val_acc = correct / total if total else 0.0
            prec1 = cm11 / (cm11 + cm01) if (cm11 + cm01) else 0.0
            rec1 = cm11 / (cm11 + cm10) if (cm11 + cm10) else 0.0
            f1_1 = (2 * prec1 * rec1 / (prec1 + rec1)) if (prec1 + rec1) else 0.0
            if f1_1 > best_f1_anom:
                best_f1_anom = f1_1
                torch.save(model.state_dict(), out_ckpt)
                print(f"  [checkpoint saved] f1_anom={f1_1:.4f} (prec={prec1:.4f}, rec={rec1:.4f})")
            print(
                f"Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  "
                f"val_acc={val_acc:.4f}  f1_anom={f1_1:.4f} (p={prec1:.4f}, r={rec1:.4f})  "
                f"cm=[[{cm00},{cm01}],[{cm10},{cm11}]]"
            )

        print(f"Training done. best f1_anom={best_f1_anom:.4f}, saved: {out_ckpt}")
        model.load_state_dict(torch.load(out_ckpt, map_location="cpu"))
        model.eval()

    # ONNX export
    dummy_input = torch.randn(1, 1, 1, 256)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            out_onnx,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )
        print(f"Success: {out_onnx} created.")
    except Exception as e:
        print(f"Export Error: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train VBX TCN-style CNN + export ONNX (RNN-replacement methodology)")
    p.add_argument("--data_dir", type=str, default="./bearing_vbx")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--out_onnx", type=str, default="vbx_tcn_256.onnx")
    p.add_argument("--out_ckpt", type=str, default="vbx_tcn_256.pth")
    p.add_argument("--no_train", action="store_true")
    p.add_argument("--no_class_weight", action="store_true")
    p.add_argument("--no_oversample", action="store_true")
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
