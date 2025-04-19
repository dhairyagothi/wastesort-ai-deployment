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
        /* Make layout mobile-friendly */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        /* Make buttons and radio options larger */
        button[kind="primary"], .stRadio > div {
            font-size: 16px !important;
        }

        /* Ensure inputs fit screen width */
        .stFileUploader, .stSelectbox, .stButton, .stRadio {
            width: 100% !important;
        }

        /* Fix video/image overflow on mobile */
        img, video {
            max-width: 100%;
            height: auto;
        }

    </style>
""", unsafe_allow_html=True)

# Initialize colorama for colored text
init(autoreset=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Function to send prediction to Arduino
def send_to_arduino(prediction, port='COM5', baudrate=9600):
    try:
        with serial.Serial(port, baudrate, timeout=2) as arduino:
            time.sleep(2)  # Wait for Arduino to reset
            arduino.write(f"{prediction}\n".encode())  # Send prediction to Arduino

            # Log the detected waste type with color
            if prediction == "wet":
                logging.info(f"✅ Object detected: {Fore.GREEN}Wet Waste (GREEN) sent to Arduino.")
            elif prediction == "dry":
                logging.info(f"✅ Object detected: {Fore.BLUE}Dry Waste (BLUE) sent to Arduino.")
            else:
                logging.info(f"⚠️ Unknown object type: {prediction} sent to Arduino.")
                
    except serial.SerialException as e:
        logging.error(f"❌ Serial Error: {e}")


# Waste categories and bin mapping
class_labels = ["Dry Waste", "Wet Waste"]
bin_mapping = {"Wet Waste": "🚮 Green Bin", "Dry Waste": "🗑️ Blue Bin"}
colors = {"Wet Waste": (0, 255, 0), "Dry Waste": (255, 0, 0)}

# Waste disposal guide
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
@st.cache_resource
def load_vit_model():
    try:
        import gdown
        from vit_keras.layers import ClassToken, TransformerBlock

        # Define the download path and file name
        file_id = "11xpp3FDyNfqGTvCrviSm9amM6cItoNNt"
        output_path = "final_vit_waste_classification_model.h5"

        # Download if not already present
        if not tf.io.gfile.exists(output_path):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

        # Load the model with custom layers
        with tf.keras.utils.custom_object_scope({'ClassToken': ClassToken, 'TransformerBlock': TransformerBlock}):
            model = tf.keras.models.load_model(output_path, compile=False)
        logging.info("✅ ViT Model loaded successfully from Google Drive!")
        return model

    except Exception as e:
        logging.error(f"❌ ViT Model loading failed: {e}")
        return None


@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO("yolov8n.pt")
        logging.info("✅ YOLOv8 Model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"❌ YOLOv8 Model loading failed: {e}")
        return None

vit_model = load_vit_model()
yolo_model = load_yolo_model()

def classify_waste(image):
    if vit_model is None:
        return "Unknown", 0.0

    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Ensure RGB
        image = cv2.resize(image, (224, 224)) / 255.0   # Normalize
        image = np.expand_dims(image, axis=0)           # Add batch dim

        prediction = vit_model.predict(image, verbose=0)[0]
        wet_confidence = prediction[0] * 100
        dry_confidence = (1 - prediction[0]) * 100
        detected_class = "Wet Waste" if wet_confidence > dry_confidence else "Dry Waste"
        confidence = max(wet_confidence, dry_confidence)
        return detected_class, round(confidence, 2)

    except Exception as e:
        logging.error(f"❌ Classification Error: {e}")
        return "Unknown", 0.0

def detect_and_classify_objects(frame):
    waste_types = []

    if yolo_model is None:
        # Preprocess full frame for ViT
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (224, 224)) / 255.0
        image_input = np.expand_dims(image_resized, axis=0)
        prediction = vit_model.predict(image_input, verbose=0)[0]
        wet_confidence = prediction[0] * 100
        dry_confidence = (1 - prediction[0]) * 100
        detected_class = "Wet Waste" if wet_confidence > dry_confidence else "Dry Waste"
        confidence = max(wet_confidence, dry_confidence)
        waste_types.append((detected_class, round(confidence, 2)))

        # 🔁 Send to Arduino even in ViT-only mode
        send_to_arduino("wet" if detected_class == "Wet Waste" else "dry")

        return frame, waste_types

    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy if results and results[0].boxes.xyxy.shape[0] > 0 else []

    if len(boxes) == 0:
        # Preprocess full frame for ViT
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (224, 224)) / 255.0
        image_input = np.expand_dims(image_resized, axis=0)
        prediction = vit_model.predict(image_input, verbose=0)[0]
        wet_confidence = prediction[0] * 100
        dry_confidence = (1 - prediction[0]) * 100
        detected_class = "Wet Waste" if wet_confidence > dry_confidence else "Dry Waste"
        confidence = max(wet_confidence, dry_confidence)
        waste_types.append((detected_class, round(confidence, 2)))

        # 🔁 Send to Arduino in fallback too
        send_to_arduino("wet" if detected_class == "Wet Waste" else "dry")

        return frame, waste_types

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.tolist())
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            continue

        # Proper preprocessing for ViT
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_resized = cv2.resize(cropped_rgb, (224, 224)) / 255.0
        input_tensor = np.expand_dims(cropped_resized, axis=0)
        prediction = vit_model.predict(input_tensor, verbose=0)[0]
        wet_confidence = prediction[0] * 100
        dry_confidence = (1 - prediction[0]) * 100
        detected_class = "Wet Waste" if wet_confidence > dry_confidence else "Dry Waste"
        confidence = max(wet_confidence, dry_confidence)

        waste_types.append((detected_class, round(confidence, 2)))

        # 🔁 Send signal to Arduino
        send_to_arduino("wet" if detected_class == "Wet Waste" else "dry")

        # Draw box
        color = colors.get(detected_class, (255, 255, 255))
        label = f"{detected_class} ({confidence:.1f}%)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, waste_types



# UI
st.title("♻️ WasteSort AI")
st.write("Upload an image, video, or use your **Live Camera** to classify waste into dry or wet types.")

option = st.radio("Choose an option:", ["📂 Upload Image", "📹 Upload Video", "📸 Live Camera"], index=0)

# Session state to track detection status and waste counts
if "detection_active" not in st.session_state:
    st.session_state.detection_active = False
if "waste_counts" not in st.session_state:
    st.session_state.waste_counts = {"Wet Waste": 0, "Dry Waste": 0}

detected_waste_counter = Counter()

# For Live Camera Selection (if multiple cameras available)
camera_index = 0  # Default
if option == "📸 Live Camera":
    st.write("### Select Camera")
    cam_options = {
        "Default (0)": 0,
        "External (1)": 1,
        "Try Front (2)": 2,
        "Try Rear (3)": 3,
    }
    camera_choice = st.selectbox("Available Cameras", list(cam_options.keys()))
    camera_index = cam_options[camera_choice]

if option == "📂 Upload Image":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if st.button("🔍 Detect & Predict"):
            img_out, waste_types = detect_and_classify_objects(img)
            detected_waste_counter.update([wt[0] for wt in waste_types])
            st.image(img_out, channels="BGR", use_container_width=True)
            st.write("### Waste Detected:")
            for waste, conf in waste_types:
                st.write(f"✅ {waste}: {conf:.2f}% confidence")

elif option == "📹 Upload Video":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        stop_button = st.button("⏹ Stop Detecting", key="stop_video")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or st.session_state.get("stop_video", False):
                break
            frame_out, waste_types = detect_and_classify_objects(frame)
            detected_waste_counter.update([wt[0] for wt in waste_types])
            stframe.image(frame_out, channels="BGR", use_container_width=True)
        cap.release()
        st.session_state["stop_video"] = False

elif option == "📸 Live Camera":
    if "live_detecting" not in st.session_state:
        st.session_state.live_detecting = False
    if "last_detected" not in st.session_state:
        st.session_state.last_detected = []
    if "waste_counts" not in st.session_state:
        st.session_state.waste_counts = {"Dry Waste": 0, "Wet Waste": 0}

    col1, col2 = st.columns(2)
    if col1.button("🔍 Start Detecting"):
        st.session_state.live_detecting = True
        st.session_state.waste_counts = {"Dry Waste": 0, "Wet Waste": 0}  # Reset counts
    if col2.button("⏹ Stop Detecting"):
        st.session_state.live_detecting = False

    stframe = st.empty()
    graph_placeholder = st.empty()
    list_placeholder = st.empty()
    summary_placeholder = st.empty()

    if st.session_state.live_detecting:
        cap = cv2.VideoCapture(camera_index)

        while st.session_state.live_detecting and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_out, waste_types = detect_and_classify_objects(frame)

            if waste_types != st.session_state.last_detected:
                st.session_state.last_detected = waste_types

                # Update counts
                for waste, _ in waste_types:
                    if waste in st.session_state.waste_counts:
                        st.session_state.waste_counts[waste] += 1

                stframe.image(frame_out, channels="BGR", use_container_width=True)

                with list_placeholder.container():
                    st.markdown("### 📋 Detected Waste List")
                    for waste, conf in waste_types:
                        st.write(f"✅ {waste}: {conf:.2f}% confidence")

    # Show summary and graph even after stopping
    with summary_placeholder.container():
        total_wet = st.session_state.waste_counts.get("Wet Waste", 0)
        total_dry = st.session_state.waste_counts.get("Dry Waste", 0)
        st.markdown(f"### ✅ Summary: {total_wet} Wet Waste | {total_dry} Dry Waste")

    with graph_placeholder.container():
        st.markdown("### 📊 Live Waste Summary")
        fig, ax = plt.subplots(figsize=(5, 3))
        labels = list(st.session_state.waste_counts.keys())
        values = list(st.session_state.waste_counts.values())
        colors = ['blue' if label == 'Dry Waste' else 'green' for label in labels]
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Count")
        ax.set_xlabel("Waste Type")
        ax.set_title("Waste Type Distribution")
        st.pyplot(fig)

    if st.session_state.live_detecting:
        cap.release()




# Summary Chart
if detected_waste_counter:
    st.subheader("📊 Waste Classification Summary")
    fig, ax = plt.subplots(figsize=(5, 3))

    labels = list(detected_waste_counter.keys())
    values = list(detected_waste_counter.values())

    # Dynamically assign color based on waste type
    color_map = {"Dry Waste": "blue", "Wet Waste": "green"}
    colors = [color_map.get(label, "gray") for label in labels]  # default to gray if unknown

    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Count")
    ax.set_xlabel("Waste Type")
    ax.set_title("Waste Type Distribution")

    st.pyplot(fig)
    st.success("♻️ Proper disposal of waste helps keep the environment clean!")
    



# 📥 Generate report DataFrame


# Guide
st.subheader("🗑️ Waste Disposal Guide")
for waste_type, info in waste_info.items():
    st.write(f"### {waste_type}")
    st.write(info["description"])
    st.write(f"**Dispose in:** {info['bin']}")
    st.image(info["bin_image"], width=150)
