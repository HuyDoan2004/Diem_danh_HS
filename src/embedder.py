import onnxruntime as ort
import numpy as np
import cv2

class ArcFaceEmbedder:
    def __init__(self, onnx_path, gpu=True):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.inp = self.sess.get_inputs()[0].name

    def embed(self, face_bgr):
        if face_bgr is None or face_bgr.size == 0:
            return None
        img = cv2.resize(face_bgr, (112,112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = (img - 127.5)/128.0
        img = np.transpose(img, (2,0,1))[None]
        vec = self.sess.run(None, {self.inp: img})[0].squeeze().astype(np.float32)
        n = np.linalg.norm(vec) + 1e-8
        return vec / n
