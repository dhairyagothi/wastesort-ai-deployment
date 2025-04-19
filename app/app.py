import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import logging
import matplotlib.pyplot as plt
from collections import Counter
from ultralytics import YOLO

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
def load_vit_model():
    try:
        from vit_keras.layers import ClassToken, TransformerBlock
        with tf.keras.utils.custom_object_scope({'ClassToken': ClassToken, 'TransformerBlock': TransformerBlock}):
            model = tf.keras.models.load_model("final_vit_waste_classification_model.h5", compile=False)
        logging.info("✅ ViT Model loaded successfully!")
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

elif option == "📸 Live Camera":
    run_detection = st.button("Start Detection")
    stop_button = st.button("⏹ Stop Detection")
    if run_detection:
        st.session_state.detection_active = True

    if stop_button:
        st.session_state.detection_active = False
        st.write("Detection stopped.")

    if st.session_state.detection_active:
        capture = cv2.VideoCapture(camera_index)
        while True:
            ret, frame = capture.read()
            if not ret:
                st.write("Error with camera feed.")
                break
            frame_out, waste_types = detect_and_classify_objects(frame)
            detected_waste_counter.update([wt[0] for wt in waste_types])
            st.image(frame_out, channels="BGR", use_container_width=True)

# Display statistics
st.write("### Waste Statistics")
st.write(f"🟢 Wet Waste: {detected_waste_counter['Wet Waste']} items")
st.write(f"🔵 Dry Waste: {detected_waste_counter['Dry Waste']} items")
