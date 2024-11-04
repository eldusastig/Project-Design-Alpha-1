import streamlit as st
import cv2
import yaml
from ultralytics import YOLO
import calendar

def load_class_labels(yaml_file):
    """Load class labels from the YAML file."""
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

def change_page(page_name):
    st.session_state.page = page_name
    st.session_state.camera_active = False

def list_connected_cameras(max_cameras=5):
    """List connected cameras by attempting to open indexes up to max_cameras."""
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

MODEL_PATH = 'model.pt'
DATA_YAML_PATH = 'data.yaml'

if 'page' not in st.session_state:
    st.session_state.page = 'Main'
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = True
if 'selected_camera' not in st.session_state:
    st.session_state.selected_camera = 0  

st.title("Welding Image Detection")

st.sidebar.header('Check Logs')
for month in calendar.month_name[1:]:  
    st.sidebar.button(month, on_click=change_page, args=(month,))

st.sidebar.header("Select Camera")
connected_cameras = list_connected_cameras()
if connected_cameras:
    selected_camera = st.sidebar.selectbox("Choose a camera:", connected_cameras, index=0)
    st.session_state.selected_camera = selected_camera
else:
    st.sidebar.write("No cameras detected.")

if st.session_state.page == 'Main':
    st.subheader("Main Page")
    st.write("Welcome to the main page for Welding Image Detection.")

    model = YOLO(MODEL_PATH)
    class_labels = load_class_labels(DATA_YAML_PATH)

    def run_camera(camera_id):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            st.error("Unable to open the camera")
            return

        camera_placeholder = st.empty()
        stop_button = st.button("Stop Camera")  
        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            results = model.predict(source=frame)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_index = int(box.cls[0])
                    label = (
                        class_labels[class_index]
                        if class_index < len(class_labels)
                        else f"Unknown Class {class_index}"
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label}: {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

         
        cap.release()

    if st.session_state.camera_active:
        run_camera(st.session_state.selected_camera)

elif st.session_state.page in calendar.month_name[1:]:
    st.subheader(f"{st.session_state.page} Logs")
    st.write(f"You are viewing logs for {st.session_state.page}.")

    if st.button("Back to Main Page"):
        st.session_state.page = 'Main'
        st.session_state.camera_active = True
