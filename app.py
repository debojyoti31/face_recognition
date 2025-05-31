# app.py

import streamlit as st
import cv2
import numpy as np
import os

from face_engine.face_model import FaceModel
from face_engine.matcher import FaceMatcher
from face_engine.db import FaceDB
from watcher.auto_enroll import AutoEnroller

# Run auto-enrollment once at startup
AutoEnroller().enroll_once()

st.set_page_config(page_title="Face Recognition", layout="centered")

model = FaceModel()
matcher = FaceMatcher()
db = FaceDB()

st.title("Face Recognition (Real-Time)")

# Debug info in sidebar
st.sidebar.write(f"Enrolled faces: {matcher.index.ntotal}")
threshold = st.sidebar.slider("Recognition Threshold", 0.1, 1.0, 0.6, 0.05)

# ----------------------------
# Live Webcam Recognition
# ----------------------------
st.subheader("Live Webcam")
run_webcam = st.checkbox("Start Webcam 📷")

if run_webcam:
    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])
    debug_info = st.empty()

    while run_webcam:
        ret, frame = cap.read()
        if not ret:
            break

        embedding, bbox = model.get_face_embedding(frame)
        if embedding is not None:
            name, score = matcher.search(embedding)
            
            # Debug output
            debug_text = f"Best match: {name}, Score: {score:.3f}, Threshold: {threshold}"
            debug_info.text(debug_text)
            
            if score is not None and score < threshold:
                frame = model.draw_bbox(frame, bbox, name, score)
            else:
                frame = model.draw_bbox(frame, bbox, "Unknown", score)
        else:
            debug_info.text("No face detected")

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

# ----------------------------
# Enrolled Faces (View-Only)
# ----------------------------
st.subheader("Enrolled Faces")
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
