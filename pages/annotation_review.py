import streamlit as st
import os
import json
import pandas as pd
import cv2
import numpy as np
from pathlib import Path
import zipfile
import shutil
import io

def show():
    st.header("Annotation Review")
    
    # Select review folder
    st.subheader("Select Data to Review")
    review_folder = st.text_input("Path to review folder (output/*/for_review)")
    
    if not review_folder or not os.path.exists(review_folder):
        st.warning("Please enter a valid review folder path")
        return
        
    # Load review data
    review_data_path = os.path.join(review_folder, "review_data.json")
    if not os.path.exists(review_data_path):
        st.error("Invalid review folder: missing review_data.json")
        return
        
    # Load class names
    class_names_path = os.path.join(review_folder, "class_names.json")
    if not os.path.exists(class_names_path):
        st.error("Invalid review folder: missing class_names.json")
        return
    
    with open(review_data_path, 'r') as f:
        review_data = json.load(f)
    
    with open(class_names_path, 'r') as f:
        class_names = json.load(f)
        
    # Determine review progress
    total_images = len(review_data)
    reviewed_images = sum(1 for img in review_data if img["reviewed"])
    
    st.progress(reviewed_images / total_images if total_images > 0 else 0)
    st.write(f"Reviewed: {reviewed_images}/{total_images} images")
    
    # Chunk selection
    st.subheader("Review Settings")
    
    # Filter options
    filter_status = st.radio("Filter by status", ["All", "Reviewed", "Not Reviewed"])
    
    # Apply filters
    filtered_data = review_data
    if filter_status == "Reviewed":
        filtered_data = [img for img in review_data if img["reviewed"]]
    elif filter_status == "Not Reviewed":
        filtered_data = [img for img in review_data if not img["reviewed"]]
    
    # Select number of items to review
    chunk_size = st.slider("Number of images to review", 
                          min_value=1, 
                          max_value=min(20, len(filtered_data)), 
                          value=min(5, len(filtered_data)))
    
    # Start review button
    if not filtered_data:
        st.warning("No images match the selected filters")
        return
        
    # Get first chunk_size unreviewed images
    chunk_to_review = filtered_data[:chunk_size]
    
    st.subheader("Image Review")
    
    # Store any changes to apply later
    changes_to_apply = []
    
    # Loop through images in the chunk
    for img_idx, img_data in enumerate(chunk_to_review):
        img_path = os.path.join(review_folder, img_data["image_path"])
        
        if not os.path.exists(img_path):
            st.error(f"Image file not found: {img_path}")
            continue
            
        st.write(f"### Image {img_idx + 1}/{len(chunk_to_review)}: {img_data['image_path']}")
        
        # Load and display the image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create columns for detections
        if not img_data["detections"]:
            st.write("No detections in this image")
            st.image(img, use_column_width=True)
        else:
            # Display original image with all bounding boxes
            img_with_boxes = img.copy()
            for det in img_data["detections"]:
                bbox = det["bbox"]
                class_name = det["class_name"]
                conf = det["confidence"]
                x1, y1, x2, y2 = [int(c) for c in bbox]
                
                # Draw bounding box
                color = (255, 0, 0) if det["reviewed"] else (0, 255, 0)
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img_with_boxes, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            st.image(img_with_boxes, caption="Original image with detections", use_column_width=True)
            
            # Create columns for detections (3 per row)
            num_detections = len(img_data["detections"])
            cols_per_row = 3
            for i in range(0, num_detections, cols_per_row):
                cols = st.columns(cols_per_row)
                
                for j in range(cols_per_row):
                    det_idx = i + j
                    if det_idx < num_detections:
                        det = img_data["detections"][det_idx]
                        with cols[j]:
                            # Extract the detection crop
                            bbox = det["bbox"]
                            x1, y1, x2, y2 = [int(c) for c in bbox]
                            crop = img[y1:y2, x1:x2]
                            
                            # Display the crop
                            st.image(crop, caption=f"Detection {det_idx + 1}")
                            
                            current_class = det["class_name"]
                            confidence = det["confidence"]
                            
                            st.write(f"Current: {current_class} ({confidence:.2f})")
                            
                            # Class selection dropdown
                            class_options = list(class_names.values())
                            new_class = st.selectbox(
                                f"Reclassify #{det_idx + 1}",
                                options=class_options,
                                index=class_options.index(current_class) if current_class in class_options else 0,
                                key=f"det_{img_idx}_{det_idx}"
                            )
                            
                            # If class changed, update changes to apply
                            if new_class != current_class:
                                changes_to_apply.append({
                                    "img_idx": review_data.index(img_data),
                                    "det_idx": det_idx,
                                    "new_class": new_class,
                                    "new_class_id": list(class_names.keys())[list(class_names.values()).index(new_class)]
                                })
        
        # Mark image as reviewed checkbox
        reviewed = st.checkbox(
            "Mark image as reviewed", 
            value=img_data["reviewed"],
            key=f"reviewed_{img_idx}"
        )
        
        if reviewed != img_data["reviewed"]:
            changes_to_apply.append({
                "img_idx": review_data.index(img_data),
                "mark_reviewed": reviewed
            })
            
        st.markdown("---")  # Divider between images
    
    # Save changes button
    if st.button("Save Changes"):
        # Apply all the changes
        for change in changes_to_apply:
            if "new_class" in change:
                # Update class for a detection
                img_idx = change["img_idx"]
                det_idx = change["det_idx"]
                
                review_data[img_idx]["detections"][det_idx]["class_name"] = change["new_class"]
                review_data[img_idx]["detections"][det_idx]["class_id"] = change["new_class_id"]
                review_data[img_idx]["detections"][det_idx]["reviewed"] = True
            
            if "mark_reviewed" in change:
                # Mark image as reviewed/unreviewed
                img_idx = change["img_idx"]
                review_data[img_idx]["reviewed"] = change["mark_reviewed"]
        
        # Save updated review data
        with open(review_data_path, 'w') as f:
            json.dump(review_data, f, indent=2)
            
        st.success("Changes saved successfully!")
        
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export CSV"):
            csv_data = export_to_csv(review_data, class_names)
            
            # Create download link
            csv_buffer = io.StringIO()
            csv_data.to_csv(csv_buffer, index=False)
            csv_bytes = csv_buffer.getvalue().encode('utf-8')
            
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name="annotations_reviewed.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export YOLO Format"):
            zip_path = export_to_yolo(review_data, review_folder, class_names)
            
            # Create download link
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="Download YOLO Dataset",
                    data=f,
                    file_name="yolo_dataset.zip",
                    mime="application/zip"
                )

def export_to_csv(review_data, class_names):
    """Export annotation data to CSV format."""
    csv_data = []
    
    for img_data in review_data:
        filename = img_data["image_path"]
        timestamp = pd.Timestamp.now().strftime("%m/%d/%Y %H:%M")
        
        # Count detections per class
        class_counts = {}
        for det in img_data["detections"]:
            class_name = det["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        # Create row for this file
        row = {
            "File": filename,
            "Timestamp": timestamp
        }
        
        # Add class counts
        for cls_name in class_names.values():
            row[cls_name] = class_counts.get(cls_name, 0)
            
        csv_data.append(row)
    
    # Convert to DataFrame
    return pd.DataFrame(csv_data)

def export_to_yolo(review_data, review_folder, class_names):
    """Export annotations in YOLO format."""
    # Create temporary directory for the YOLO dataset
    temp_dir = os.path.join(review_folder, "yolo_export")
    images_dir = os.path.join(temp_dir, "images")
    labels_dir = os.path.join(temp_dir, "labels")
    
    # Create directories
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Create class mapping (name -> id)
    class_map = {v: int(k) for k, v in class_names.items()}
    
    # Create data.yaml file
    yaml_content = {
        "path": ".",
        "train": "images",
        "val": "",
        "test": "",
        "nc": len(class_names),
        "names": list(class_names.values())
    }
    
    with open(os.path.join(temp_dir, "data.yaml"), 'w') as f:
        f.write("path: .\n")
        f.write("train: images\n")
        f.write("val: \n")
        f.write("test: \n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {list(class_names.values())}\n")
    
    # Process each image
    for img_data in review_data:
        img_path = os.path.join(review_folder, img_data["image_path"])
        if not os.path.exists(img_path):
            continue
        
        # Copy image to the YOLO images directory
        img_filename = os.path.basename(img_path)
        dst_img_path = os.path.join(images_dir, img_filename)
        shutil.copy(img_path, dst_img_path)
        
        # Load the image to get dimensions
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]
        
        # Create the label file
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            # Write each detection in YOLO format: <class> <x_center> <y_center> <width> <height>
            for det in img_data["detections"]:
                cls_id = class_map.get(det["class_name"], 0)
                
                # Get bounding box in absolute coordinates
                x1, y1, x2, y2 = det["bbox"]
                
                # Convert to YOLO format (normalized)
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                
                # Write to file
                f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")
    
    # Create zip file
    zip_path = temp_dir + ".zip"
    shutil.make_archive(temp_dir, 'zip', temp_dir)
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    
    return zip_path + ".zip"
