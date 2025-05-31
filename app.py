# app.py

import streamlit as st
import numpy as np
import cv2
import os

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

from face_engine.face_model import FaceModel
from face_engine.matcher import FaceMatcher
from face_engine.db import FaceDB
from watcher.auto_enroll import AutoEnroller

# Auto-enroll new images at startup
AutoEnroller().enroll_once()

st.set_page_config(page_title="Face Recognition", layout="centered")

st.title("🧠 Face Recognition (WebRTC + Auto-Enroll)")

model = FaceModel()
matcher = FaceMatcher()
db = FaceDB()

# Sidebar controls
st.sidebar.write(f"🧬 Enrolled Faces: {matcher.index.ntotal}")
threshold = st.sidebar.slider("Recognition Threshold", 0.1, 1.5, 0.6, 0.05)

# ----------------------------
# WebRTC-based Webcam Section
# ----------------------------
st.subheader("📷 Live Recognition via Webcam")

class FaceRecognitionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        embedding, bbox = model.get_face_embedding(img)

        if embedding is not None:
            name, score = matcher.search(embedding)
            if score is not None and score < threshold:
                img = model.draw_bbox(img, bbox, name, score)
            else:
                img = model.draw_bbox(img, bbox, "Unknown", score)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

webrtc_streamer(key="facecam", video_transformer_factory=FaceRecognitionTransformer)

# ----------------------------
# View Enrolled Faces
# ----------------------------
st.subheader("🧑 Enrolled Faces")
names = db.list_faces()

if names:
    cols = st.columns(min(len(names), 4))
    for i, name in enumerate(names):
        with cols[i % len(cols)]:
            img_path = db.get_image_path(name)
            if img_path and os.path.exists(img_path):
                st.image(img_path, width=120, caption=name)
            else:
                st.write(name)
else:
    st.info("No faces enrolled yet.")
