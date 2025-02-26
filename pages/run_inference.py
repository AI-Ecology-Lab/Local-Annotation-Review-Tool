import streamlit as st
import os
import pandas as pd
import tempfile
import shutil
import zipfile
from ultralytics import YOLO
import torch
from pathlib import Path
import time
import json

def show():
    st.header("Run YOLOv11 Inference")
    
    # Model selection
    st.subheader("Model Selection")
    model_options = {"Upload custom model": None}
    
    # Add default models if they exist in the weights folder
    weights_dir = Path("weights")
    if weights_dir.exists():
        for weight_file in weights_dir.glob("*.pt"):
            model_options[f"{weight_file.name}"] = str(weight_file)
    
    selected_model_option = st.selectbox(
        "Choose a model",
        options=list(model_options.keys())
    )
    
    # Handle custom model upload
    model_path = model_options[selected_model_option]
    if selected_model_option == "Upload custom model":
        uploaded_model = st.file_uploader("Upload model weights (.pt file)", type=['pt'])
        if uploaded_model:
            # Save the uploaded model to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_file.write(uploaded_model.getvalue())
                model_path = tmp_file.name
    
    # Data source selection
    st.subheader("Data Source")
    source_type = st.radio("Select source type", ["Upload Files", "Directory Path"])
    
    source_path = None
    
    if source_type == "Upload Files":
        uploaded_files = st.file_uploader("Upload images or videos", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'], accept_multiple_files=True)
        if uploaded_files:
            # Create a temporary directory to store uploaded files
            temp_dir = tempfile.mkdtemp()
            for uploaded_file in uploaded_files:
                temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            source_path = temp_dir
    else:
        source_path = st.text_input("Enter directory path containing images/videos")
    
    # Model parameters
    st.subheader("Model Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
        iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.7)
        img_size = st.number_input("Image Size", min_value=32, max_value=1920, value=1024)
        half_precision = st.checkbox("Half Precision (FP16)", value=False)
    
    with col2:
        device = st.selectbox(
            "Device",
            options=["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        )
        batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=1)
        max_detections = st.number_input("Maximum Detections", min_value=1, max_value=1000, value=300)
        agnostic_nms = st.checkbox("Class-agnostic NMS", value=False)
    
    # Output options
    st.subheader("Output Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        save_txt = st.checkbox("Save TXT", value=True)
        save_conf = st.checkbox("Save Confidence", value=True)
        save_crop = st.checkbox("Save Crops", value=True)
        
    with col2:
        save_csv = st.checkbox("Save CSV", value=True)
        visualize = st.checkbox("Visualize Features", value=False)
        show_labels = st.checkbox("Show Labels", value=True)
        show_conf = st.checkbox("Show Confidence", value=True)

    # Run inference button
    if st.button("Run Inference"):
        if not model_path:
            st.error("Please select or upload a model.")
            return
        
        if not source_path:
            st.error("Please provide a data source.")
            return
        
        # Create output directory
        output_dir = os.path.join("output", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(output_dir, exist_ok=True)
        
        # Status indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Loading model...")
            model = YOLO(model_path)
            
            status_text.text("Running inference...")
            
            # Run inference
            results = model(
                source=source_path,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=img_size,
                half=half_precision,
                device=device,
                batch=batch_size,
                max_det=max_detections,
                agnostic_nms=agnostic_nms,
                visualize=visualize,
                save=True,
                save_txt=save_txt,
                save_conf=save_conf,
                save_crop=save_crop,
                project=output_dir,
                name="inference",
                show_labels=show_labels,
                show_conf=show_conf
            )
            
            # Process results and create CSV
            if save_csv:
                status_text.text("Generating CSV output...")
                generate_csv_output(results, output_dir)
                
            # Create a "for_review" directory with organized data for annotation review
            status_text.text("Preparing data for review...")
            prepare_for_review(results, output_dir, model.names)
            
            progress_bar.progress(1.0)
            status_text.text("Inference completed successfully!")
            
            # Create a download link for results
            zip_results(output_dir)
            with open(f"{output_dir}.zip", "rb") as f:
                st.download_button(
                    label="Download Results",
                    data=f,
                    file_name=f"results_{time.strftime('%Y%m%d-%H%M%S')}.zip",
                    mime="application/zip"
                )
            
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")
        finally:
            # Clean up temporary files if they were created
            if source_type == "Upload Files" and source_path:
                shutil.rmtree(source_path, ignore_errors=True)
            if selected_model_option == "Upload custom model" and model_path:
                try:
                    os.unlink(model_path)
                except:
                    pass

def generate_csv_output(results, output_dir):
    """Generate CSV output in the specified format."""
    csv_data = []
    
    for result in results:
        filename = Path(result.path).name
        timestamp = time.strftime("%m/%d/%Y %H:%M")
        
        # Count detections per class
        class_counts = {}
        for cls in result.boxes.cls.cpu().numpy():
            cls_int = int(cls)
            cls_name = result.names[cls_int]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
        # Create row for this file
        row = {
            "File": filename,
            "Timestamp": timestamp
        }
        
        # Add class counts
        for cls_name in result.names.values():
            row[cls_name] = class_counts.get(cls_name, 0)
            
        csv_data.append(row)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, "predictions.csv")
    df.to_csv(csv_path, index=False)
    
    return csv_path

def prepare_for_review(results, output_dir, class_names):
    """Prepare data for the annotation review stage."""
    review_dir = os.path.join(output_dir, "for_review")
    os.makedirs(review_dir, exist_ok=True)
    
    # Save class names for later use
    with open(os.path.join(review_dir, "class_names.json"), 'w') as f:
        json.dump(class_names, f)
    
    # Create structure to track images and their detections
    review_data = []
    
    for i, result in enumerate(results):
        img_filename = Path(result.path).name
        img_path = result.path
        
        # Copy original image to review directory
        review_img_path = os.path.join(review_dir, img_filename)
        shutil.copy(img_path, review_img_path)
        
        detections = []
        
        # Process each detection
        for j, (bbox, cls, conf) in enumerate(zip(result.boxes.xyxy.cpu().numpy(), 
                                          result.boxes.cls.cpu().numpy(),
                                          result.boxes.conf.cpu().numpy())):
            cls_id = int(cls)
            cls_name = class_names[cls_id]
            
            detection = {
                "id": j,
                "bbox": bbox.tolist(),
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": float(conf),
                "reviewed": False
            }
            
            detections.append(detection)
        
        img_data = {
            "image_path": img_filename,
            "detections": detections,
            "reviewed": False
        }
        
        review_data.append(img_data)
    
    # Save the review data
    with open(os.path.join(review_dir, "review_data.json"), 'w') as f:
        json.dump(review_data, f, indent=2)

def zip_results(dir_path):
    """Create a zip file of the results directory."""
    shutil.make_archive(dir_path, 'zip', dir_path)
