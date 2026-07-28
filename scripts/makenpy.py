"""
Create calib_data.npy for calibration (onnx2tf INT8 quantization)
- Expected input for onnx2tf: NHWC relative to TF SavedModel → (N, 1, 256, 1)
- Generated with random data. For real data, use calib_from_bearing_vbx.py.
"""
import numpy as np

num_samples = 100
# After TFLite/onnx2tf conversion, input shape is (1, 1, 256, 1) NHWC
calib_data = np.random.randn(num_samples, 1, 256, 1).astype(np.float32)
np.save("calib_data.npy", calib_data)
print("calib_data.npy created:", calib_data.shape)
