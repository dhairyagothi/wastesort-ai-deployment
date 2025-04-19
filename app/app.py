import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import logging
import matplotlib.pyplot as plt
from collections import Counter
from ultralytics import YOLO
import serial
import time
import logging
from colorama import Fore, init
import pandas as pd

st.markdown("""
    <style>
        .block-container {
            padding: 1rem;
        }
        button[kind="primary"], .stRadio > div {
            font-size: 16px !important;
        }
        .stFileUploader, .stSelectbox, .stButton, .stRadio {
            width: 100% !important;
        }
        img, video {
            max-width: 100%;
            height: auto;
        }
    </style>
""", unsafe_allow_html=True)

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

class_labels = ["Dry Waste", "Wet Waste"]
bin_mapping = {"Wet Waste": "\ud83d\udeae Green Bin", "Dry Waste": "\ud83d\udc9f\ufe0f Blue Bin"}
colors = {"Wet Waste": (0, 255, 0), "Dry Waste": (255, 0, 0)}

waste_info = {
    "Dry Waste": {
        "description": "Dry waste includes paper, plastic, metal, glass, and cardboard. These should be recycled properly.",
        "bin": "Blue Bin (Recyclable Waste)",
        "bin_image": "assets/blue_bin.png"
    },
    "Wet Waste": {
        "description": "Wet waste includes food scraps, vegetable peels, and garden waste. It should be composted or disposed in a wet waste bin.",
        "bin": "Green Bin (Organic Waste)",
        "bin_image": "assets/green_bin.png"
    }
}

@st.cache_resource
def load_vit_model():
    try:
        import gdown
        from vit_keras.layers import ClassToken, TransformerBlock
        file_id = "11xpp3FDyNfqGTvCrviSm9amM6cItoNNt"
        output_path = "final_vit_waste_classification_model.h5"
        if not tf.io.gfile.exists(output_path):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        with tf.keras.utils.custom_object_scope({'ClassToken': ClassToken, 'TransformerBlock': TransformerBlock}):
            model = tf.keras.models.load_model(output_path, compile=False)
        logging.info("\u2705 ViT Model loaded successfully from Google Drive!")
        return model
    except Exception as e:
        logging.error(f"\u274c ViT Model loading failed: {e}")
        return None

@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO("yolov8n.pt")
        logging.info("\u2705 YOLOv8 Model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"\u274c YOLOv8 Model loading failed: {e}")
        return None

vit_model = load_vit_model()
yolo_model = load_yolo_model()

def classify_waste(image):
    if vit_model is None:
        return "Unknown", 0.0
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = vit_model.predict(image, verbose=0)[0]
        wet_confidence = prediction[0] * 100
        dry_confidence = (1 - prediction[0]) * 100
        detected_class = "Wet Waste" if wet_confidence > dry_confidence else "Dry Waste"
        confidence = max(wet_confidence, dry_confidence)
        return detected_class, round(confidence, 2)
    except Exception as e:
        logging.error(f"\u274c Classification Error: {e}")
        return "Unknown", 0.0

def detect_and_classify_objects(frame):
    waste_types = []
    if yolo_model is None or not hasattr(yolo_model, '__call__'):
        return frame, [(classify_waste(frame))]
    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy if results and results[0].boxes.xyxy.shape[0] > 0 else []
    if len(boxes) == 0:
        return frame, [(classify_waste(frame))]
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.tolist())
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        detected_class, confidence = classify_waste(cropped)
        waste_types.append((detected_class, round(confidence, 2)))
        color = colors.get(detected_class, (255, 255, 255))
        label = f"{detected_class} ({confidence:.1f}%)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame, waste_types

st.title("\u267b\ufe0f WasteSort AI")
st.write("Upload an image, video, or use your **Live Camera** to classify waste into dry or wet types.")

option = st.radio("Choose an option:", ["\ud83d\udcc2 Upload Image", "\ud83d\udcf9 Upload Video", "\ud83d\udcf8 Live Camera"], index=0)

detected_waste_counter = Counter()
camera_index = 0
if option == "\ud83d\udcf8 Live Camera":
    cam_options = {
        "Default (0)": 0,
        "External (1)": 1,
        "Try Front (2)": 2,
        "Try Rear (3)": 3,
    }
    camera_choice = st.selectbox("Available Cameras", list(cam_options.keys()))
    camera_index = cam_options[camera_choice]

if option == "\ud83d\udcc2 Upload Image":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if st.button("\ud83d\udd0d Detect & Predict"):
            img_out, waste_types = detect_and_classify_objects(img)
            detected_waste_counter.update([wt[0] for wt in waste_types])
            st.image(img_out, channels="BGR", use_container_width=True)
            st.write("### Waste Detected:")
            for waste, conf in waste_types:
                st.write(f"\u2705 {waste}: {conf:.2f}% confidence")

elif option == "\ud83d\udcf9 Upload Video":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_out, waste_types = detect_and_classify_objects(frame)
            detected_waste_counter.update([wt[0] for wt in waste_types])
            stframe.image(frame_out, channels="BGR", use_container_width=True)
        cap.release()

elif option == "\ud83d\udcf8 Live Camera":
    col1, col2 = st.columns(2)
    if col1.button("\ud83d\udd0d Start Detecting"):
        st.session_state.live_detecting = True
        st.session_state.waste_counts = {"Dry Waste": 0, "Wet Waste": 0}
    if col2.button("\u23f9 Stop Detecting"):
        st.session_state.live_detecting = False
    stframe = st.empty()
    graph_placeholder = st.empty()
    list_placeholder = st.empty()
    summary_placeholder = st.empty()
    if st.session_state.get("live_detecting", False):
        cap = cv2.VideoCapture(camera_index)
        while st.session_state.live_detecting and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_out, waste_types = detect_and_classify_objects(frame)
            for waste, _ in waste_types:
                if waste in st.session_state.waste_counts:
                    st.session_state.waste_counts[waste] += 1
            stframe.image(frame_out, channels="BGR", use_container_width=True)
            with list_placeholder.container():
                st.markdown("### \ud83d\udccb Detected Waste List")
                for waste, conf in waste_types:
                    st.write(f"\u2705 {waste}: {conf:.2f}% confidence")
        cap.release()
    with summary_placeholder.container():
        st.markdown(f"### \u2705 Summary: {st.session_state.waste_counts.get('Wet Waste', 0)} Wet Waste | {st.session_state.waste_counts.get('Dry Waste', 0)} Dry Waste")
    with graph_placeholder.container():
        st.markdown("### \ud83d\udcca Live Waste Summary")
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = list(st.session_state.waste_counts.keys())
        values = list(st.session_state.waste_counts.values())
        colors_bar = ['blue' if label == 'Dry Waste' else 'green' for label in labels]
        ax.bar(labels, values, color=colors_bar)
        ax.set_ylabel("Count")
        ax.set_xlabel("Waste Type")
        ax.set_title("Waste Type Distribution")
        st.pyplot(fig)

if detected_waste_counter:
    st.subheader("\ud83d\udcca Waste Classification Summary")
    fig, ax = plt.subplots()
    labels = list(detected_waste_counter.keys())
    values = list(detected_waste_counter.values())
    ax.bar(labels, values, color=['green' if lbl == 'Wet Waste' else 'blue' for lbl in labels])
    ax.set_ylabel("Count")
    ax.set_xlabel("Waste Type")
    ax.set_title("Overall Waste Detection Summary")
    st.pyplot(fig)