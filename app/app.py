import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import logging
from collections import Counter
from ultralytics import YOLO

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Waste categories and bin mapping
class_labels = ["Dry Waste", "Wet Waste"]
bin_mapping = {"Wet Waste": "🚮 Green Bin", "Dry Waste": "🗑️ Blue Bin"}
colors = {"Wet Waste": (0, 255, 0), "Dry Waste": (255, 0, 0)}

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
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224)) / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = vit_model.predict(image, verbose=0)[0]
        wet_confidence = prediction[0] * 100
        dry_confidence = (1 - prediction[0]) * 100
        detected_class = "Wet Waste" if wet_confidence > dry_confidence else "Dry Waste"
        return detected_class, round(max(wet_confidence, dry_confidence), 2)
    except Exception as e:
        logging.error(f"❌ Classification Error: {e}")
        return "Unknown", 0.0

def detect_and_classify_objects(frame):
    waste_types = []
    if yolo_model is None or vit_model is None:
        return frame, waste_types

    results = yolo_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results and hasattr(results[0].boxes, 'xyxy') else []

    if len(boxes) == 0:
        detected_class, confidence = classify_waste(frame)
        waste_types.append((detected_class, confidence))
        return frame, waste_types

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.tolist())
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            continue

        detected_class, confidence = classify_waste(cropped)
        waste_types.append((detected_class, confidence))

        color = colors.get(detected_class, (255, 255, 255))
        label = f"{detected_class} ({confidence:.1f}%)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame, waste_types

def list_available_cameras(max_cams=5):
    available = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append(i)
        cap.release()
    return available

# Streamlit UI
st.set_page_config(page_title="WasteSort AI", layout="wide")
st.title("♻️ WasteSort AI")
st.write("Classify waste into Dry and Wet types using AI with images, video, or live camera.")

option = st.radio("Choose input type:", ["📂 Upload Image", "📹 Upload Video", "📸 Live Camera"], index=0)

detected_waste_counter = Counter()

if option == "📂 Upload Image":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image and st.button("🔍 Detect & Predict"):
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        output_img, waste_types = detect_and_classify_objects(img)
        detected_waste_counter.update([wt[0] for wt in waste_types])
        st.image(output_img, channels="BGR", use_container_width=True)
        st.write("### Waste Detected:")
        for waste, conf in waste_types:
            st.write(f"✅ {waste}: {conf:.2f}%")

elif option == "📹 Upload Video":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        stop = st.button("⏹ Stop Video")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop:
                break
            output_frame, waste_types = detect_and_classify_objects(frame)
            detected_waste_counter.update([wt[0] for wt in waste_types])
            stframe.image(output_frame, channels="BGR", use_container_width=True)

elif option == "📸 Live Camera":
    st.write("### Available Cameras")
    available_cams = list_available_cameras()
    if available_cams:
        cam_index = st.selectbox("Select Camera", available_cams, index=0)
        if st.button("Start Live Detection"):
            cap = cv2.VideoCapture(cam_index)
            stframe = st.empty()
            stop = st.button("⏹ Stop Live")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop:
                    break
                output_frame, waste_types = detect_and_classify_objects(frame)
                detected_waste_counter.update([wt[0] for wt in waste_types])
                stframe.image(output_frame, channels="BGR", use_container_width=True)
            cap.release()
    else:
        st.write("❌ No cameras found. Please connect a camera and refresh the page.")

# Show final stats
st.write("### 🧾 Waste Statistics")
st.write(f"🟢 Wet Waste: {detected_waste_counter['Wet Waste']}")
st.write(f"🔵 Dry Waste: {detected_waste_counter['Dry Waste']}")
