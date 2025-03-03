# IFSEC Combine Models

## Overview
This project integrates multiple AI models into a single script for real-time detection of various safety and security events. It is designed for deployment at the IFSEC Exhibition in Delhi, demonstrating real-time video analytics capabilities for multiple scenarios, including:

- **Drowning Detection**
- **Sitting & Standing Detection**
- **Pothole Detection**
- **Accident Detection**
- **Unknown Object Detection**
- **PPE Kit Compliance Detection**

## Features
- **Multi-functional detection system** covering various safety and security concerns
- **Real-time processing** using OpenCV and YOLO-based deep learning models
- **Single integrated script** for seamless execution
- **Multi-camera support** for extended monitoring
- **Custom alert mechanisms** for different detected scenarios

## File Structure
```
|-- LICENSE                                # Project license file
|-- README.md                              # Project documentation
|-- app2.py                                # Initial version of the application
|-- app5.py                                # Updated version with improvements
|-- app6.py                                # Latest version of the unified detection script
|-- best.pt                                # Trained YOLO model for multiple detections
|-- bounding_box_text.py                   # Script for overlaying text on bounding boxes
|-- video.py                               # Video processing script
|-- webcam.py                              # Webcam integration script
|-- webcam_1.py                            # Alternative webcam processing script
```

## Requirements
Ensure the following dependencies are installed:
- Python 3.8+
- OpenCV
- Ultralytics YOLO
- PyTorch
- NumPy
- Requests (for alerting system)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Running the Unified Detection System
```bash
python app6.py
```

### Running Video-Based Inference
```bash
python video.py
```

### Running Webcam-Based Inference
```bash
python webcam.py
```

## Model Information
The system utilizes a unified YOLO model trained on multiple datasets for detecting:
- Drowning incidents
- Sitting/Standing posture
- Potholes
- Accidents
- Unknown objects
- PPE kit compliance (Helmet, Vest, Boots detection)

## Future Enhancements
- Integration with a centralized dashboard for real-time monitoring
- Cloud-based storage for detected events
- Expansion to detect additional safety scenarios

## Contributors
- Kavusikaa Prabhu
- AI-ML R&D Team

## License
This project is licensed under the MIT License.

