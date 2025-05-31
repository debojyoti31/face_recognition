# face_engine/face_model.py

import insightface
import numpy as np
import cv2

class FaceModel:
    def __init__(self):
        self.model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=0)

    def get_face_embedding(self, img):
        faces = self.model.get(img)
        if not faces:
            return None, None
        face = faces[0]
        return face.embedding, face.bbox

    def draw_bbox(self, img, bbox, name, threshold, score=None):
        x1, y1, x2, y2 = map(int, bbox)
        label = f"{name}" if score > threshold else f"{name} ({score:.2f})"
        if score is not None and score < threshold:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
       
        return img

