# YOLO Object Detection with Webcam

This project uses YOLOv8 to perform real-time object detection on a webcam feed. It can detect various objects including accidents and display relevant information on the video stream.

## Features

- **Real-time object detection** using YOLOv8.
- **Detection of multiple classes** such as 'Accident', 'pothole', 'helmet', and more.
- **Customizable class colors** for better visualization.
- **Styled messages** to emphasize detected accidents.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- ultralytics library (`YOLO`)

You can install the required Python packages using:

```bash
pip install opencv-python ultralytics
```

## How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```

2. Navigate to the project directory:

   ```bash
   cd your-repo-name
   ```

3. Run the script:

   ```bash
   python detect_accidents.py
   ```

4. Open the webcam window. Press 'q' to quit the application.

### Details

- **Model Path**: The script loads a pre-trained YOLO model from a specified path.
- **Class Names**: It uses a predefined list of class names to label detected objects.
- **Detection Threshold**: The confidence threshold for displaying bounding boxes is set to 0.5.
- **Accident Detection**: A special message is displayed when an accident is detected.

### Customization

You can customize the colors for each class in the `class_colors` dictionary in the code. Modify the `model_path` to point to your trained YOLO model file if needed.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgements

- YOLOv8 for object detection.
- OpenCV for real-time video processing.
