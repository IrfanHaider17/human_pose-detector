import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import os
import torch
import cv2

# Set page config
st.set_page_config(
    page_title="Custom YOLOv8 Pose Detection",
    page_icon=":bust_in_silhouette:",
    layout="wide"
)

# Title and description
st.title("Custom YOLOv8 Pose Detection")
st.write("Upload an image to detect poses using your custom trained YOLOv8m model")

# Secure model loading function
@st.cache_resource
def load_model(model_path="model.pt"):
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
        
        # Add necessary globals to safe list
        from ultralytics.nn.tasks import PoseModel
        torch.serialization.add_safe_globals([PoseModel])
        
        # Load model with proper security settings
        model = YOLO(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model. Error: {str(e)}")
        return None

def process_image(image, model, conf_threshold):
    try:
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        results = model(image_np, conf=conf_threshold)
        annotated_frame = results[0].plot()
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), results
    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        return None, None

def main():
    with st.sidebar:
        st.header("Configuration")
        conf_threshold = st.slider(
            "Confidence Threshold", 0.1, 0.9, 0.5, 0.05,
            help="Minimum confidence level for detections"
        )
        model_path = st.text_input(
            "Model path", "model.pt",
            help="Path to your YOLOv8 model file"
        )
        st.warning("Only use models from trusted sources!")
    
    model = load_model(model_path)
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        if model and st.button("Detect Poses"):
            with st.spinner("Detecting poses..."):
                result_image, results = process_image(image, model, conf_threshold)
                
                if result_image:
                    with col2:
                        st.image(result_image, caption="Detected Poses", use_column_width=True)
                    
                    st.subheader("Detection Results")
                    if results and len(results) > 0:
                        st.write(f"Detected {len(results[0].boxes)} persons")
                        if hasattr(results[0], "keypoints"):
                            st.write(f"Total keypoints detected: {len(results[0].keypoints.xy[0])} per person")

if __name__ == "__main__":
    main()