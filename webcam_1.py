import cv2
import math
from ultralytics import YOLO

# Load the trained model
model_path = "/home/vmukti/Dataset/Combine_dataset/train69/weights/best.pt"  # Path to your trained YOLO model
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

# Utility function to check bounding box overlap
def is_overlap(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    return not (x2 < x1_p or x2_p < x1 or y2 < y1_p or y2_p < y1)

# Utility function to draw a label with a background
def draw_label_with_background(img, label, x, y, color, font_scale=0.6, thickness=1):
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_width, text_height = text_size
    text_offset_x, text_offset_y = 5, 5
    box_coords = ((x, y), (x + text_width + text_offset_x, y - text_height - text_offset_y))
    cv2.rectangle(img, box_coords[0], box_coords[1], color, cv2.FILLED)
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

# Open webcam (0 for default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get video properties for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video file name and codec
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

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
                color = class_colors.get(class_name, (255, 255, 255))  # Default to white if class not in colors
                if class_name not in special_classes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {conf:.2f}"
                    draw_label_with_background(frame, label, x1, y1 - 10, color)

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

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame with bounding boxes
    cv2.imshow("Real-Time Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()

