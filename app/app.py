import streamlit as st
import tensorflow as tf
from vit_keras.layers import ClassToken, TransformerBlock
from ultralytics import YOLO

st.title("✅ Model Load Test - WasteSort AI")

# ViT Model Load
@st.cache_resource
def load_vit_model():
    try:
        with tf.keras.utils.custom_object_scope({'ClassToken': ClassToken, 'TransformerBlock': TransformerBlock}):
            model = tf.keras.models.load_model("final_vit_waste_classification_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"❌ ViT model loading failed: {e}")
        return None

# YOLOv8 Model Load
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        st.error(f"❌ YOLO model loading failed: {e}")
        return None

# Load models and display result
vit_model = load_vit_model()
yolo_model = load_yolo_model()

if vit_model:
    st.success("✅ ViT model loaded successfully!")

if yolo_model:
    st.success("✅ YOLOv8 model loaded successfully!")
