# 1D-CNN-exam-VBX

Prototype 1D time-series anomaly detection for **Microchip PolarFire VectorBlox**.

## Why height-1 images?

VectorBlox is a **2D CNN accelerator**. It does not provide a native 1D/RNN path, so this project maps each length-256 window to an NCHW tensor `(1, 1, 256)` (H=1, W=time) and runs **Conv2D with kernels `(1, k)`** along the width. That is effectively 1D convolution expressed with 2D ops that VBX already supports.

## Contents

- `scripts/` — data prep, train, ONNX export, eval
- `artifacts/` — example `.pth`, `.onnx`, TFLite outputs
- `samples/` — board test `bearing_samples.bin` and calibration data
- `retrain_pipeline.sh` — retrain flow (paths are relative to this repo)

## Requirements

- Python 3 + NumPy + PyTorch
- [NASA Bearing dataset (Kaggle)](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset) for training from scratch (~6GB download)
- Optional: onnx2tf / VectorBlox SDK for further deployment

Ready-to-try checkpoints and ONNX/TFLite files are under `artifacts/` if you only want to inspect or convert models.

## Retrain

Download the Kaggle dataset, then point `--root` at the extracted tree (typically `.../bearing-dataset/versions/1`).

```bash
# from this repo root
bash retrain_pipeline.sh /path/to/bearing-dataset/versions/1

# or step by step
python scripts/prepare_bearing_vbx.py \
  --root /path/to/bearing-dataset/versions/1 \
  --out_dir ./bearing_vbx \
  --rms_anomaly_ratio 0.05

python scripts/make_model.py \
  --data_dir ./bearing_vbx \
  --out_onnx artifacts/vbx_sensor_256.onnx \
  --out_ckpt artifacts/vbx_sensor_256.pth
```

## RMS labeling notes

Labels on `3rd_test` use **RMS**: the top-`r` fraction of windows is marked anomaly (`--rms_anomaly_ratio r`).  
A small sweep (epochs=20, oversample + class weight, eval on the full prepared set) showed a clear **recall ↔ false-alarm** trade-off:

| RMS ratio `r` | Anomaly count | Anomaly recall | False positives (normal→anomaly) | Takeaway |
|---------------|---------------|----------------|----------------------------------|----------|
| 0.02 (2%) | ~10k | **low** (~0.03) | ~10k | Detects some anomalies, misses most |
| 0.01 (1%) | ~5k | **~1.0** | ~53k | Rarely misses anomalies, many false alarms |
| 0.005 (0.5%) | ~2.5k | **~1.0** | ~41k | Still near-perfect recall; FP still high |

**Note:** tighter RMS labels (more “extreme failure only”) raise anomaly recall, but FP grows a lot. For deployment, tune the **decision threshold** on logits, or harden labels (e.g. RMS + consecutive-window hysteresis), rather than chasing accuracy alone.

The default pipeline uses `r=0.05` as a starting point; re-sweep if your false-alarm budget differs.
