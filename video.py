import cv2
import math
from ultralytics import YOLO

# Load the trained model
model_path = "best.pt"  # Path to your trained YOLO model
model = YOLO(model_path)

# Define class names
class_names = ['Fall', 'Sitting', 'Standing', 'drowning', 'pothole', 'Accident', 
               'Unknown_Car', 'negative', 'helmet', 'no-helmet', 'no-vest', 
               'person', 'vest']

# Define the specific class logic list
special_classes = ["helmet", "no-helmet", "no-vest", "person", "vest"]

# Utility function to check bounding box overlap
def is_overlap(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    return not (x2 < x1_p or x2_p < x1 or y2 < y1_p or y2_p < y2)

# Utility function to draw a label with a background
def draw_label_with_background(img, label, x, y, color, font_scale=0.6, thickness=1):
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_width, text_height = text_size
    text_offset_x, text_offset_y = 5, 5
    box_coords = ((x, y), (x + text_width + text_offset_x, y - text_height - text_offset_y))
    cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

# Input video file
input_path = "/home/vmukti/Downloads/VID-20241211-WA0017.mp4"
cap = cv2.VideoCapture(input_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get video properties for output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
output_path = "accident.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
output_video = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

# Video frame processing
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO detection on the current frame
    results = model(frame)

    # Lists to hold detected bounding boxes for specific classes
    detected_helmet = []
    detected_vest = []
    persons = []

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            # Apply general logic for all classes
            if conf > 0.5:
                if class_name not in special_classes:
                    # General bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 0), 2)

                # Apply specific logic for special classes
                else:
                    if class_name == "helmet":
                        detected_helmet.append((x1, y1, x2, y2))
                    elif class_name == "vest":
                        detected_vest.append((x1, y1, x2, y2))
                    elif class_name == "person":
                        persons.append((x1, y1, x2, y2))

    # Specific logic for persons, helmet, and vest
    for person_bbox in persons:
        x1, y1, x2, y2 = person_bbox
        is_helmet = any(is_overlap(person_bbox, (hx1, hy1, hx2, hy2)) for (hx1, hy1, hx2, hy2) in detected_helmet)
        is_vest = any(is_overlap(person_bbox, (vx1, vy1, vx2, vy2)) for (vx1, vy1, vx2, vy2) in detected_vest)

        label_y_offset = y1 - 15
        if is_helmet and is_vest:
            color = (0, 255, 0)
            label = "PPE kit detected"
        elif is_helmet and not is_vest:
            color = (0, 255, 255)
            label = "Helmet, No Jacket"
        elif is_vest and not is_helmet:
            color = (0, 255, 255)
            label = "Jacket, No Helmet"
        else:
            color = (0, 0, 255)
            label = "No PPE kit"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label_with_background(frame, label, x1, label_y_offset, color)

    # Write the frame to the output video
    output_video.write(frame)

# Release the resources
cap.release()
output_video.release()
cv2.destroyAllWindows()

