import onnxruntime as ort
import numpy as np
import cv2

class RetinaFaceONNX:
    """
    Giả định model ONNX trả về:
      out[0]: boxes (N,4) theo thứ tự [x1,y1,x2,y2] chuẩn hoá khung input
      out[1]: scores (N,)
    Nếu bản tải khác format -> chỉnh phần hậu xử lý (decode) bên dưới.
    """
    def __init__(self, onnx_path, gpu=True, input_size=(640,640)):
        self.input_size = input_size
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.inp_name = self.sess.get_inputs()[0].name

    def _pre(self, img):
        h0, w0 = img.shape[:2]
        img_resized = cv2.resize(img, self.input_size)
        blob = img_resized[:, :, ::-1].astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = np.transpose(blob, (2,0,1))[None]  # 1x3xH xW
        return blob, (w0, h0)

    def detect(self, img, conf_thres=0.6):
        blob, (w0, h0) = self._pre(img)
        out = self.sess.run(None, {self.inp_name: blob})
        # ---- CHỈNH Ở ĐÂY nếu output model khác ----
        boxes = out[0][0]   # (N,4) theo input_size
        scores = out[1][0]  # (N,)
        keep = scores > conf_thres
        boxes, scores = boxes[keep], scores[keep]
        sx, sy = w0/self.input_size[0], h0/self.input_size[1]
        boxes[:, [0,2]] *= sx
        boxes[:, [1,3]] *= sy
        return [(float(x1), float(y1), float(x2), float(y2), float(sc))
                for (x1,y1,x2,y2), sc in zip(boxes, scores)]

class FallbackHaar:
    def __init__(self):
        self.clf = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, img, conf_thres=0.6):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.clf.detectMultiScale(gray, 1.2, 5)
        out = []
        for (x,y,w,h) in faces:
            out.append((x, y, x+w, y+h, 0.99))
        return out

def build_detector(onnx_path, gpu=True):
    try:
        return RetinaFaceONNX(onnx_path, gpu=gpu)
    except Exception as e:
        print("[detect] ONNX detector init failed, fallback Haar:", e)
        return FallbackHaar()
