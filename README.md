# Local Annotation Review Tool

A Streamlit application for running YOLOv11 inference and reviewing/correcting annotations.

## Features

### Run Inference Page
- Load YOLOv11 models from weights folder or upload custom models
- Configure inference parameters (confidence threshold, IoU, image size, etc.)
- Process images or videos from a directory or upload files
- Export results in multiple formats (TXT, CSV)
- Save cropped detections for easy review

### Annotation Review Page
- Review annotations in manageable chunks
- Visualize all detections in an image
- Review individual cropped detections
- Reclassify detections with a simple dropdown interface
- Track review progress
- Export finalized annotations in CSV and YOLO formats

## Installation

1. Clone this repository
2. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```
streamlit run app.py
```

2. Open your browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

3. Use the navigation sidebar to switch between pages:
   - **Run Inference**: Execute YOLOv11 model on selected data
   - **Annotation Review**: Review and correct model predictions

### Running Inference

1. Select or upload a YOLOv11 model
2. Choose your data source (upload files or specify a directory)
3. Configure model parameters
4. Set output options
5. Click "Run Inference" to start processing
6. Download the results when complete

### Reviewing Annotations

1. Enter the path to the review folder (created during inference)
2. Set how many images you want to review in the current session
3. Review and correct class predictions as needed
4. Mark images as reviewed
5. Save your changes
6. Export the final annotations in your preferred format (CSV or YOLO)

## Directory Structure

- `app.py`: Main application entry point
- `pages/`: Contains the individual application pages
  - `run_inference.py`: Code for the inference page
  - `annotation_review.py`: Code for the annotation review page
- `weights/`: Directory for storing model weights
- `output/`: Directory where results are saved
