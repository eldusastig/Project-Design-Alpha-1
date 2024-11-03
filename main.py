import streamlit as st
import numpy as np
import cv2
import yaml
from ultralytics import YOLO

def load_class_labels(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

model_path = 'best.pt'  
data_yaml_path = 'data.yaml'  

st.title("Welding Image Detection")
st.write("Using your camera to detect welding-related objects")
model=YOLO(model_path)
class_labels=load_class_labels(data_yaml_path)
def run_camera(model):
    cap = cv2.VideoCapture(0)  
    
    if not cap.isOpened():
        st.error("Unable to open the camera")
        return

    camera_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        results = model.predict(source=frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  
                confidence = float(box.conf[0]) 
                class_index = int(box.cls[0])     
                if class_index < len(class_labels):
                    label = class_labels[class_index]
                else:
                    label = f"Unknown Class {class_index}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    cap.release()
run_camera(model)
st.markdown(
    """
    <a href="/logs" target="_blank">
        <button style="display: block; width: 100%; padding: 1em; margin-top: 1em; color: white; background-color: #007ACC; border: none; border-radius: 5px;">
            Logs
        </button>
    </a>
    """,
    unsafe_allow_html=True
)
