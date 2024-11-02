import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile

model_path = 'best.pt'  
try:
    model = YOLO(model_path)
    st.success(f"YOLOv8 model loaded successfully from {model_path}")
except Exception as e:
    st.error(f"Error loading YOLOv8 model: {e}")

st.title("Real-Time Object Detection with YOLOv8")
st.write("Using your camera to detect objects in real-time")

def run_camera(model):
    cap = cv2.VideoCapture(0)  
    
    if not cap.isOpened():
        st.error("Unable to open the camera")
        return

    camera_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            break
        
        results = model(frame)
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy.int().tolist()[0]  
            confidence = box.conf  
            label = box.cls_name  #
            
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

# Button to start the camera feed
if st.button("Start Camera"):
    run_camera(model)

# Note: To run this Streamlit app, save it to `app.py` and use:
# streamlit run app.py
