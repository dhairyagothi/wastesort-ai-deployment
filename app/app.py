# Filename: app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import logging
import matplotlib.pyplot as plt
from collections import Counter
from ultralytics import YOLO
import gdown

# --- Streamlit UI Styling ---
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

# --- Init & Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# --- Waste Labels ---
class_labels = ["Dry Waste", "Wet Waste"]
bin_mapping = {"Wet Waste": "🚮 Green Bin", "Dry Waste": "🗑️ Blue Bin"}
colors = {"Wet Waste": (0, 255, 0), "Dry Waste": (255, 0, 0)}

# --- Load ViT Model ---
@st.cache_resource
def load_vit_model():
    try:
        file_id = "11xpp3FDyNfqGTvCrviSm9amM6cItoNNt"
        output_path = "final_vit_waste_classification_model.h5"
        if not tf.io.gfile.exists(output_path):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
        model = tf.keras.models.load_model(output_path, compile=False)
        logging.info("✅ ViT Model loaded!")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to load ViT: {e}")
        return None

# --- Load YOLOv8 ---
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO("yolov8n.pt")
        logging.info("✅ YOLOv8 loaded!")
        return model
    except Exception as e:
        logging.error(f"❌ Failed to load YOLOv8: {e}")
        return None

vit_model = load_vit_model()
yolo_model = load_yolo_model()

# --- Waste Classification ---
def classify_waste(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = vit_model.predict(image, verbose=0)[0]
        wet_conf = prediction[0] * 100
        dry_conf = (1 - prediction[0]) * 100
        cls = "Wet Waste" if wet_conf > dry_conf else "Dry Waste"
        return cls, round(max(wet_conf, dry_conf), 2)
    except Exception as e:
        logging.error(f"❌ ViT classification error: {e}")
        return "Unknown", 0.0

# --- Object Detection + Classification ---
def detect_and_classify_objects(frame):
    waste_types = []

    if yolo_model is None:
        waste_types.append(classify_waste(frame))
        return frame, waste_types

    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy if results and results[0].boxes else []

    if len(boxes) == 0:
        waste_types.append(classify_waste(frame))
        return frame, waste_types

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.tolist())
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            continue

        cls, conf = classify_waste(cropped)
        waste_types.append((cls, conf))

        # Draw boxes
        color = colors.get(cls, (255, 255, 255))
        label = f"{cls} ({conf:.1f}%)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, waste_types

# --- Streamlit UI ---
st.title("♻️ WasteSort AI")
st.write("Upload image/video or use **Live Camera** to detect & classify waste (Dry/Wet).")

option = st.radio("Choose input mode:", ["📂 Upload Image", "📹 Upload Video", "📸 Live USB Camera"])

# Session state
if "waste_counts" not in st.session_state:
    st.session_state.waste_counts = {"Wet Waste": 0, "Dry Waste": 0}

detected_waste_counter = Counter()

# --- Upload Image ---
if option == "📂 Upload Image":
    uploaded_image = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image and st.button("🔍 Detect & Predict"):
        try:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if img is None:
                st.error("❌ Failed to decode the image. Please upload a valid image file.")
            else:
                img_out, waste_types = detect_and_classify_objects(img)

                if img_out is not None and isinstance(img_out, np.ndarray):
                    st.image(img_out, channels="BGR", use_container_width=True)
                    st.write("### Waste Detected:")
                    for wt in waste_types:
                        if isinstance(wt, tuple):
                            waste, conf = wt
                            st.write(f"✅ {waste}: {conf:.2f}%")
                            st.session_state.waste_counts[waste] += 1
                        elif isinstance(wt, str):
                            st.write(f"✅ {wt}")
                            st.session_state.waste_counts[wt] += 1
                else:
                    st.error("❌ Processed image is invalid.")
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {e}")


# --- Upload Video ---
elif option == "📹 Upload Video":
    uploaded_video = st.file_uploader("Upload video...", type=["mp4", "avi", "mov"])
    if uploaded_video and st.button("▶️ Start Video Processing"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_out, waste_types = detect_and_classify_objects(frame)
            stframe.image(frame_out, channels="BGR", use_container_width=True)
            for wt in waste_types:
                if isinstance(wt, tuple):
                    waste, _ = wt
                    st.session_state.waste_counts[waste] += 1
        cap.release()

# --- Live Camera ---
elif option == "📸 Live USB Camera":
    cam_options = {
        "Default (0)": 0,
        "External (1)": 1,
        "Try Front (2)": 2,
        "Try Rear (3)": 3,
    }
    selected_cam = st.selectbox("Choose Camera", list(cam_options.keys()))
    camera_index = cam_options[selected_cam]

    col1, col2 = st.columns(2)
    if col1.button("📷 Start Live Detection"):
        st.session_state.live_detect = True
    if col2.button("⏹ Stop"):
        st.session_state.live_detect = False

    stframe = st.empty()
    graph_placeholder = st.empty()
    summary_placeholder = st.empty()

    if st.session_state.get("live_detect", False):
        cap = cv2.VideoCapture(camera_index)

        while st.session_state.live_detect and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_out, waste_types = detect_and_classify_objects(frame)
            stframe.image(frame_out, channels="BGR", use_container_width=True)
            for wt in waste_types:
                if isinstance(wt, tuple):
                    waste, _ = wt
                    st.session_state.waste_counts[waste] += 1
        cap.release()

# --- Summary ---
st.subheader("📊 Waste Summary")
total_wet = st.session_state.waste_counts["Wet Waste"]
total_dry = st.session_state.waste_counts["Dry Waste"]
st.write(f"✅ **Wet Waste:** {total_wet} | 🗑️ **Dry Waste:** {total_dry}")

fig, ax = plt.subplots()
labels = list(st.session_state.waste_counts.keys())
values = list(st.session_state.waste_counts.values())
bar_colors = ['green' if l == "Wet Waste" else 'blue' for l in labels]
ax.bar(labels, values, color=bar_colors)
ax.set_ylabel("Count")
ax.set_title("Waste Type Distribution")
st.pyplot(fig)
