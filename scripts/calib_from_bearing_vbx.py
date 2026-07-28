"""
Convert bearing_vbx/calib_data.npy (NCHW: N,1,1,256) → NHWC (N,1,256,1) and save.
Use this when onnx2tf -cind expects shape (1, 1, 256, 1) NHWC.
"""
import numpy as np

src = np.load("bearing_vbx/calib_data.npy")  # (N, 1, 1, 256)
# NCHW -> NHWC: (N,1,1,256) -> (N,1,256,1)
calib_nhwc = np.transpose(src, (0, 1, 3, 2))  # (N, 1, 256, 1)
np.save("calib_data.npy", calib_nhwc.astype(np.float32))
print("calib_data.npy (NHWC) created:", calib_nhwc.shape)
