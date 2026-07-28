#!/usr/bin/env python3
"""
NASA Bearing Dataset → VectorBlox preprocessing
Segment (256), normal/anomaly labels, normalize, then write X.npy, y.npy, norm_params.json, calib_data.npy.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np


def _parse_timestamp(name: str):
    parts = name.replace("-", ".").split(".")
    if len(parts) >= 6:
        try:
            y, m, d, h, mi, s = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            return datetime(y, m, d, h, mi, s)
        except (ValueError, IndexError):
            pass
    return datetime.min


def collect_data_files(root: Path):
    root = Path(root)
    collected = []
    for folder, sub in [("1st_test", "1st_test"), ("2nd_test", "2nd_test"), ("3rd_test", "4th_test/txt")]:
        base = root / folder / sub
        if not base.exists():
            continue
        for f in base.iterdir():
            if f.is_file() and not f.name.startswith("."):
                collected.append((f, folder, _parse_timestamp(f.name)))
    order = {"1st_test": 0, "2nd_test": 1, "3rd_test": 2}
    collected.sort(key=lambda x: (order.get(x[1], 99), x[2]))
    return collected


def load_file(path: Path, channel: int = 0) -> np.ndarray:
    data = np.loadtxt(path, delimiter="\t", dtype=np.float64)
    if data.ndim == 1:
        return data
    return data[:, min(channel, data.shape[1] - 1)].ravel()


def load_all_series(file_list, channel: int = 0):
    return np.concatenate([load_file(p, channel) for p, _, _ in file_list], axis=0)


def segment_series(series: np.ndarray, segment_len: int = 256, stride: int = 256):
    n = len(series)
    segments = [series[s : s + segment_len] for s in range(0, n - segment_len + 1, stride)]
    if not segments:
        return np.empty((0, 1, 1, segment_len), dtype=np.float32)
    out = np.stack(segments, axis=0).astype(np.float32)
    out = out[:, np.newaxis, np.newaxis, :]
    if not out.flags["C_CONTIGUOUS"]:
        out = np.ascontiguousarray(out)
    return out


def build_labels(file_list, segment_len: int, stride: int, anomaly_ratio_3rd: float, channel: int = 0):
    seg_counts = []
    test_names = []
    for path, test_name, _ in file_list:
        data = load_file(path, channel)
        n_seg = max(0, (len(data) - segment_len) // stride + 1)
        seg_counts.append(n_seg)
        test_names.append(test_name)
    third_seg_count = sum(c for c, n in zip(seg_counts, test_names) if n == "3rd_test")
    n_normal_3rd = int(third_seg_count * (1 - anomaly_ratio_3rd))
    labels = []
    third_idx = 0
    for c, name in zip(seg_counts, test_names):
        if name in ("1st_test", "2nd_test"):
            labels.extend([0] * c)
        else:
            for _ in range(c):
                labels.append(0 if third_idx < n_normal_3rd else 1)
                third_idx += 1
    return np.array(labels, dtype=np.int64)


def build_labels_rms(file_list, segment_len: int, stride: int, rms_threshold: float, channel: int = 0):
    labels = []
    for path, test_name, _ in file_list:
        data = load_file(path, channel)
        for s in range(0, len(data) - segment_len + 1, stride):
            seg = data[s : s + segment_len]
            rms = np.sqrt(np.mean(seg**2))
            if test_name in ("1st_test", "2nd_test"):
                labels.append(0)
            else:
                labels.append(1 if rms > rms_threshold else 0)
    return np.array(labels, dtype=np.int64)


def estimate_rms_threshold(file_list, segment_len: int, stride: int, anomaly_ratio: float, channel: int = 0) -> float:
    if not (0.0 < anomaly_ratio < 1.0):
        raise ValueError("--rms_anomaly_ratio must be in (0,1)")
    percentile = 1.0 - anomaly_ratio
    rms_vals = []
    for path, test_name, _ in file_list:
        if test_name != "3rd_test":
            continue
        data = load_file(path, channel)
        for s in range(0, len(data) - segment_len + 1, stride):
            seg = data[s : s + segment_len]
            rms_vals.append(float(np.sqrt(np.mean(seg**2))))
    if not rms_vals:
        raise RuntimeError("No 3rd_test data under --root. Check path (e.g. .../versions/1).")
    return float(np.quantile(np.asarray(rms_vals, dtype=np.float64), percentile))


def normalize(X: np.ndarray, method: str, fit_params: dict = None):
    if fit_params:
        if fit_params["method"] == "minmax":
            X_n = (X - fit_params["min"]) / (fit_params["max"] - fit_params["min"] + 1e-8)
        else:
            X_n = (X - fit_params["mean"]) / (fit_params["std"] + 1e-8)
        X_n = np.clip(X_n.astype(np.float32), 0.0, 1.0) if fit_params["method"] == "minmax" else np.clip(X_n.astype(np.float32), -1.0, 1.0)
        return X_n, fit_params
    if method == "minmax":
        low, high = X.min(), X.max()
        params = {"method": "minmax", "min": float(low), "max": float(high)}
        X_n = np.clip(((X - low) / (high - low + 1e-8)).astype(np.float32), 0.0, 1.0)
    else:
        mean, std = X.mean(), X.std()
        params = {"method": "zscore", "mean": float(mean), "std": float(std)}
        X_n = np.clip(((X - mean) / (std + 1e-8)).astype(np.float32), -1.0, 1.0)
    return X_n, params


def main():
    p = argparse.ArgumentParser(description="NASA Bearing → VectorBlox preprocessing (X.npy, y.npy, calib_data.npy)")
    p.add_argument("--root", required=True, help="Dataset root (bearing-dataset/versions/1)")
    p.add_argument("--out_dir", default="./bearing_vbx", help="Output directory")
    p.add_argument("--segment_len", type=int, default=256)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--normalize", choices=["minmax", "zscore"], default="minmax")
    p.add_argument("--anomaly_ratio_3rd", type=float, default=0.3,
                   help="Anomaly ratio for the latter part of 3rd_test (default labeling)")
    p.add_argument("--rms_anomaly_ratio", type=float, default=None,
                   help="If set, use RMS-based labeling. e.g. 0.02 = top 2%% as anomaly")
    p.add_argument("--calib_samples", type=int, default=100, help="Number of samples for calib_data.npy (0=skip)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_list = collect_data_files(Path(args.root))
    if not file_list:
        raise FileNotFoundError(
            f"No data files found under --root: {args.root}\n"
            "Check that the path is .../bearing-dataset/versions/1."
        )

    series = load_all_series(file_list, args.channel)
    X = segment_series(series, args.segment_len, args.stride)

    if args.rms_anomaly_ratio is not None:
        thr = estimate_rms_threshold(file_list, args.segment_len, args.stride, args.rms_anomaly_ratio, args.channel)
        print(f"   RMS labeling (threshold={thr:.6g})")
        y = build_labels_rms(file_list, args.segment_len, args.stride, thr, args.channel)
    else:
        y = build_labels(file_list, args.segment_len, args.stride, args.anomaly_ratio_3rd, args.channel)
    y = y[: X.shape[0]]

    X_norm, norm_params = normalize(X, args.normalize)
    if args.normalize == "minmax":
        norm_params["vbx_uint8_guide"] = "X_uint8 = np.clip(X_norm * 255, 0, 255).astype(np.uint8)"
    else:
        norm_params["vbx_uint8_guide"] = "X_uint8 = np.clip((X_norm + 1) * 127.5, 0, 255).astype(np.uint8)"

    with open(out_dir / "norm_params.json", "w", encoding="utf-8") as f:
        json.dump(norm_params, f, indent=2, ensure_ascii=False)
    np.save(out_dir / "X.npy", X_norm)
    np.save(out_dir / "y.npy", y)

    if args.calib_samples > 0:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_norm), size=min(args.calib_samples, len(X_norm)), replace=False)
        np.save(out_dir / "calib_data.npy", X_norm[idx])

    print(f"Done: {out_dir}  X: {X_norm.shape}  y: {y.shape}  normal/anomaly: {np.sum(y==0)} / {np.sum(y==1)}")


if __name__ == "__main__":
    main()
