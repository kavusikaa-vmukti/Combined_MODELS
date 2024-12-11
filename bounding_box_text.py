import cv2
import math
from ultralytics import YOLO

# Load the trained model
model_path = "/home/vmukti/IFSEC-FINAL/Final_Folder/train69/weights/best.pt"  # Path to your trained YOLO model
model = YOLO(model_path)

# Define class names
class_names = ['Fall', 'Sitting', 'Standing', 'drowning', 'pothole', 'Accident',
               'Unknown_Car', 'negative', 'helmet', 'no-helmet', 'no-vest',
               'person', 'vest']

# Define the specific class logic list
special_classes = ["helmet", "no-helmet", "no-vest", "person", "vest"]

# Define colors for each class
class_colors = {
    'Fall': (255, 0, 0),
    'Sitting': (0, 255, 255),
    'Standing': (255, 165, 0),
    'drowning': (0, 128, 255),
    'pothole': (128, 0, 128),
    'Accident': (0, 0, 255),  # Red for Accident
    'Unknown_Car': (128, 128, 0),
    'negative': (64, 64, 64),
    'helmet': (0, 255, 0),
    'no-helmet': (255, 255, 0),
    'no-vest': (255, 255, 255),
    'person': (0, 255, 255),
    'vest': (0, 128, 0)
}

# Utility function to draw a label with a background
def draw_label_with_background(img, label, x, y, color, font_scale=0.6, thickness=1):
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_width, text_height = text_size
    text_offset_x, text_offset_y = 5, 5
    box_coords = ((x, y), (x + text_width + text_offset_x, y - text_height - text_offset_y))
    cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

# Function to display a styled accident message
def display_accident_message(frame, text, position=None, font=cv2.FONT_HERSHEY_COMPLEX, font_scale=1, color=(0, 0, 255), thickness=3):
    """
    Display a styled text with a translucent rectangle background for emphasis.
    """
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size

    # Default position to the top-left corner if not provided
    if position is None:
        position = (10, text_height + 20)  # Padding of 10 pixels from the top and left

    x, y = position
    padding = 10

    # Draw translucent background rectangle
    cv2.rectangle(frame, (x, y - text_height - padding), (x + text_width + padding * 2, y), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.3, frame, 0.7, 0, frame)  # Add some transparency to the background

    # Draw text
    cv2.putText(frame, text, (x + padding, y - padding), font, font_scale, color, thickness, cv2.LINE_AA)

# Input and output video paths
input_video_path = "/home/vmukti/Dataset/Combine_dataset/captured_video.mp4"  # Replace with the path to your input video
output_video_path = "/home/vmukti/Dataset/Combine_dataset/captured_video_with_detections_7.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# Get video properties for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video file name and codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Video frame processing
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection on the current frame
    results = model(frame)

    # Lists to hold detected bounding boxes for specific classes
    accident_detected = False  # Flag for accident detection

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            # Apply general logic for all classes
            if conf > 0.5:
                color = class_colors.get(class_name, (255, 255, 255))  # Default to white if class not in colors
                if class_name != "Accident":
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {conf:.2f}"
                    draw_label_with_background(frame, label, x1, y1 - 10, color)

                # Detect accidents and mark the flag
                elif class_name == "Accident":
                    accident_detected = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Display accident message if detected
    if accident_detected:
        display_accident_message(frame, "Accident Detected")

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame with bounding boxes (optional)
    cv2.imshow("Processed Video", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()

