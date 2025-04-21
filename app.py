import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Set page config
st.set_page_config(
    page_title="Custom YOLOv8 Pose Detection",
    page_icon=":bust_in_silhouette:",
    layout="wide"
)

# Title and description
st.title("Custom YOLOv8 Pose Detection")
st.write("Upload an image to detect poses using your custom trained YOLOv8m model")

# Function to load the custom YOLO model
@st.cache_resource
def load_model(model_path="model.pt"):
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
        model = YOLO(model_path)  # Load your custom trained model
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load model. Error: {e}")
        return None

# Function to perform pose detection
def detect_pose(model, image, conf_threshold=0.5):
    try:
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Convert RGB to BGR (OpenCV uses BGR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Perform detection with confidence threshold
        results = model(image_np, conf=conf_threshold)
        
        # Visualize the results
        annotated_frame = results[0].plot()
        
        # Convert BGR back to RGB for display
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        return annotated_frame, results
    except Exception as e:
        st.error(f"Error during pose detection: {e}")
        return None, None

# Main application
def main():
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        conf_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.5, 
            step=0.05,
            help="Adjust the minimum confidence level for detections"
        )
        
        # Model selection with default path
        model_path = st.text_input(
            "Model path (relative to app directory)",
            value="model.pt",
            help="Path to your custom trained YOLOv8 model file"
        )
    
    # Load model - moved inside main to access model_path
    model = load_model(model_path)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        
        # Only show detect button if model loaded successfully
        if model is not None:
            if st.button("Detect Poses"):
                with st.spinner("Detecting poses..."):
                    result_image, results = detect_pose(model, image, conf_threshold)
                    
                    if result_image is not None:
                        with col2:
                            st.image(
                                result_image, 
                                caption="Detected Poses", 
                                use_column_width=True
                            )
                        
                        # Display detection results
                        st.subheader("Detection Results")
                        
                        if results and len(results) > 0:
                            result = results[0]
                            num_detections = len(result.boxes)
                            st.write(f"Number of detections: {num_detections}")
                            
                            if hasattr(result, "keypoints") and result.keypoints is not None:
                                st.write("Keypoints detected:")
                                for i, kpts in enumerate(result.keypoints.xy):
                                    st.write(f"Person {i+1}: {len(kpts)} keypoints")
        else:
            st.warning("Please ensure the model file exists at the specified path.")

# Run the app
if __name__ == "__main__":
    main()