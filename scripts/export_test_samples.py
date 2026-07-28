"""
Sample export script for VBX 1D model testing
- Extract test samples from bearing_vbx/X.npy
- Save as uint8 binary file (used by run-1D-model.cpp)
"""

import os
import argparse
import numpy as np


def export_samples(
    data_dir: str = "./bearing_vbx",
    output_file: str = "bearing_samples.bin",
    num_samples: int = 100,
    seed: int = 42,
):
    """
    Export test samples to a binary file
    
    Args:
        data_dir: Directory containing X.npy and y.npy
        output_file: Output binary file path
        num_samples: Number of samples to extract
        seed: Random seed
    """
    data_dir = os.path.abspath(data_dir)
    X_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "y.npy")
    
    if not os.path.isfile(X_path) or not os.path.isfile(y_path):
        raise FileNotFoundError(
            f"X.npy or y.npy not found: {data_dir}\n"
            "Generate the data first with prepare_bearing_vbx.py."
        )
    
    # Load data
    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    
    print(f"Data loaded: X {X.shape}, y {y.shape}")
    print(f"  Normal(0): {np.sum(y == 0)}")
    print(f"  Anomaly(1): {np.sum(y == 1)}")
    
    # Random sample selection (keep class balance)
    rng = np.random.default_rng(seed)
    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)
    
    # Take roughly half from each class
    n0 = min(num_samples // 2, len(idx0))
    n1 = min(num_samples - n0, len(idx1))
    
    selected_idx0 = rng.choice(idx0, size=n0, replace=False)
    selected_idx1 = rng.choice(idx1, size=n1, replace=False)
    selected_idx = np.concatenate([selected_idx0, selected_idx1])
    rng.shuffle(selected_idx)
    
    X_selected = X[selected_idx]
    y_selected = y[selected_idx]
    
    print(f"\nSelected samples: {len(selected_idx)}")
    print(f"  Normal(0): {np.sum(y_selected == 0)}")
    print(f"  Anomaly(1): {np.sum(y_selected == 1)}")
    
    # X shape: (N, 1, 1, 256) → (N, 256)
    X_flat = X_selected.squeeze()  # (N, 256)
    
    # Convert to uint8 (scale to 0-255)
    # Assumes source data is already in 0-255 range
    # If data is normalized, inverse normalization may be needed
    X_uint8 = np.clip(X_flat * 255, 0, 255).astype(np.uint8)
    
    # Save as binary (contiguous 256-byte blocks)
    X_uint8.tofile(output_file)
    
    print(f"\n✓ Saved: {output_file}")
    print(f"  File size: {os.path.getsize(output_file)} bytes")
    print(f"  Expected size: {len(selected_idx) * 256} bytes")
    
    # Also save labels separately (for verification)
    label_file = output_file.replace(".bin", "_labels.txt")
    with open(label_file, "w") as f:
        for i, label in enumerate(y_selected):
            f.write(f"{i+1:04d}: {'Anomaly' if label == 1 else 'Normal'}\n")
    print(f"✓ Labels saved: {label_file}")
    
    # Sample stats
    print(f"\n=== Sample stats ===")
    print(f"Min: {X_uint8.min()}, Max: {X_uint8.max()}")
    print(f"Mean: {X_uint8.mean():.2f}, Std: {X_uint8.std():.2f}")
    
    # Preview first sample
    print(f"\n=== First sample (label: {y_selected[0]}) ===")
    print(f"First 10 values: {X_uint8[0, :10]}")
    print(f"Last 10 values: {X_uint8[0, -10:]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export samples for VBX testing")
    p.add_argument(
        "--data_dir",
        type=str,
        default="./bearing_vbx",
        help="Directory containing X.npy and y.npy",
    )
    p.add_argument(
        "--output",
        type=str,
        default="bearing_samples.bin",
        help="Output binary file path",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to extract",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = p.parse_args()
    
    export_samples(
        data_dir=args.data_dir,
        output_file=args.output,
        num_samples=args.num_samples,
        seed=args.seed,
    )
