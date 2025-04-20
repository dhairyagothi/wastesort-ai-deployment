import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import logging
import matplotlib.pyplot as plt
from collections import Counter
from ultralytics import YOLO
st.set_page_config(page_title="♻️ WasteSort AI", layout="centered")
# Disable GPU for TensorFlow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Waste categories and mapping
class_labels = ["Dry Waste", "Wet Waste"]
bin_mapping = {"Wet Waste": "🚮 Green Bin", "Dry Waste": "🗑️ Blue Bin"}
colors = {"Wet Waste": (0, 255, 0), "Dry Waste": (255, 0, 0)}

# Waste disposal guide
waste_info = {
    "Dry Waste": {
        "description": "Dry waste includes paper, plastic, metal, glass, and cardboard.",
        "bin": "Blue Bin (Recyclable Waste)",
        "bin_image": "assets/blue_bin.png"
    },
    "Wet Waste": {
        "description": "Wet waste includes food scraps, vegetable peels, and garden waste.",
        "bin": "Green Bin (Organic Waste)",
        "bin_image": "assets/green_bin.png"
    }
}

@st.cache_resource
def load_vit_model():
    try:
        from vit_keras.layers import ClassToken, TransformerBlock
        with tf.keras.utils.custom_object_scope({'ClassToken': ClassToken, 'TransformerBlock': TransformerBlock}):
            model = tf.keras.models.load_model("final_vit_waste_classification_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"ViT model loading failed: {e}")
        return None

@st.cache_resource
def load_yolo_model():
    try:
        return YOLO("yolov8n.pt")
    except Exception as e:
        st.error(f"YOLO model loading failed: {e}")
        return None

vit_model = load_vit_model()
yolo_model = load_yolo_model()

def classify_waste(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = vit_model.predict(image, verbose=0)[0]
        wet_conf = prediction[0] * 100
        dry_conf = (1 - prediction[0]) * 100
        label = "Wet Waste" if wet_conf > dry_conf else "Dry Waste"
        return label, round(max(wet_conf, dry_conf), 2)
    except Exception as e:
        logging.error(f"Classification error: {e}")
        return "Unknown", 0.0

def detect_and_classify_objects(frame):
    waste_types = []
    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy if results and results[0].boxes.xyxy.shape[0] > 0 else []

    if len(boxes) == 0:
        label, conf = classify_waste(frame)
        waste_types.append((label, conf))
        return frame, waste_types

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.tolist())
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        label, conf = classify_waste(cropped)
        waste_types.append((label, conf))

        color = colors.get(label, (255, 255, 255))
        text = f"{label} ({conf:.1f}%)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, waste_types


st.title("♻️ WasteSort AI")
st.write("Classify waste into **dry** and **wet** using AI. Upload images/videos or use your webcam (locally only).")

option = st.radio("Choose an option:", ["📂 Upload Image", "📹 Upload Video", "📸 Live Camera"], index=0)

detected_waste_counter = Counter()

# Image Upload
if option == "📂 Upload Image":
    img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if img_file:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if st.button("🔍 Detect & Classify"):
            img_out, results = detect_and_classify_objects(img)
            for r in results:
                detected_waste_counter[r[0]] += 1
            st.image(img_out, channels="BGR", use_container_width=True)
            for label, conf in results:
                st.write(f"✅ {label}: {conf:.2f}%")

# Video Upload
elif option == "📹 Upload Video":
    vid_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        stop = st.button("⏹ Stop")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop:
                break
            out_frame, results = detect_and_classify_objects(frame)
            for r in results:
                detected_waste_counter[r[0]] += 1
            stframe.image(out_frame, channels="BGR", use_container_width=True)
        cap.release()

# Live Camera (only works locally)
elif option == "📸 Live Camera":
    st.warning("Live Camera works only when running Streamlit locally (not on Streamlit Cloud).")
    if st.button("🎥 Start Camera"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out_frame, results = detect_and_classify_objects(frame)
            for r in results:
                detected_waste_counter[r[0]] += 1
            stframe.image(out_frame, channels="BGR", use_container_width=True)
        cap.release()

# Summary Chart
if detected_waste_counter:
    st.subheader("📊 Waste Summary")
    fig, ax = plt.subplots()
    labels = list(detected_waste_counter.keys())
    values = list(detected_waste_counter.values())
    color_map = {"Dry Waste": "blue", "Wet Waste": "green"}
    colors = [color_map.get(label, "gray") for label in labels]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Count")
    ax.set_title("Waste Type Distribution")
    st.pyplot(fig)

# Waste Guide
st.subheader("🗑️ Disposal Guide")
for label, info in waste_info.items():
    st.markdown(f"### {label}")
    st.write(info["description"])
    st.markdown(f"**Dispose in:** {info['bin']}")
    st.image(info["bin_image"], width=150)
