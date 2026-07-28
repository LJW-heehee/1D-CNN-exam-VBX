import numpy as np

X = np.load("bearing_vbx/X.npy")  # (N,1,1,256), float32 (assumes minmax 0~1)
X_uint8 = np.clip(X * 255, 0, 255).astype(np.uint8)
X_uint8[0].reshape(-1).tofile("sample.bin")  # 256 bytes
