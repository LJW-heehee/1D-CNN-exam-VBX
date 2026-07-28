#!/bin/bash
# Retrain pipeline for 1D bearing anomaly CNN (VectorBlox prototype).
# Usage: bash retrain_pipeline.sh /path/to/bearing-dataset/versions/1
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS="${ROOT_DIR}/scripts"
OUT_VBX="${ROOT_DIR}/bearing_vbx"
DATASET_ROOT="${1:-}"

if [[ -z "${DATASET_ROOT}" ]]; then
  echo "Usage: $0 /path/to/bearing-dataset/versions/1"
  exit 1
fi

echo "======================================================================"
echo " STEP 1: Build dataset (RMS-based labels on 3rd_test)"
echo "======================================================================"
python "${SCRIPTS}/prepare_bearing_vbx.py" \
    --root "${DATASET_ROOT}" \
    --out_dir "${OUT_VBX}" \
    --rms_anomaly_ratio 0.05 \
    --normalize minmax \
    --calib_samples 200

echo ""
echo "======================================================================"
echo " STEP 2: Train + export ONNX"
echo "======================================================================"
cd "${ROOT_DIR}"
python "${SCRIPTS}/make_model.py" \
    --data_dir "${OUT_VBX}" \
    --epochs 80 \
    --no_oversample \
    --out_onnx "${ROOT_DIR}/artifacts/vbx_sensor_256.onnx" \
    --out_ckpt "${ROOT_DIR}/artifacts/vbx_sensor_256.pth"

echo ""
echo "======================================================================"
echo " STEP 3: Export board test samples"
echo "======================================================================"
python "${SCRIPTS}/export_test_samples.py" \
    --data_dir "${OUT_VBX}" \
    --output "${ROOT_DIR}/samples/bearing_samples.bin" \
    --num_samples 100

echo ""
echo "DONE."
echo "  checkpoint/onnx : ${ROOT_DIR}/artifacts/"
echo "  samples         : ${ROOT_DIR}/samples/bearing_samples.bin"
echo "======================================================================"
