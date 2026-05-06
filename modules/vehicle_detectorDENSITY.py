import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open traffic video
video_path = r"K:\TMSYoLo\videos\Video 1.mp4"
cap = cv2.VideoCapture(video_path)

# Check video opened or not
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# Vehicle classes from COCO dataset
vehicle_classes = ["car", "motorcycle", "bus", "truck"]

# FPS calculation
prev_time = 0

while True:

    # Read frame
    ret, frame = cap.read()

    # Stop if video ends
    if not ret:
        print("Video completed.")
        break

    # Resize frame for better performance
    frame = cv2.resize(frame, (640, 384))

    # Vehicle counter
    vehicle_count = 0

    # Default density
    density = "LOW"

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Loop through detections
    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Confidence score
            confidence = float(box.conf[0])

            # Ignore weak detections
            if confidence < 0.4:
                continue

            # Class ID
            class_id = int(box.cls[0])

            # Class name
            class_name = model.names[class_id]

            # Detect only vehicles
            if class_name in vehicle_classes:

                # Increase count
                vehicle_count += 1

                # Density classification
                if vehicle_count <= 5:
                    density = "LOW"

                elif vehicle_count <= 15:
                    density = "MEDIUM"

                else:
                    density = "HIGH"

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Box color based on density
                if density == "LOW":
                    color = (0, 255, 0)

                elif density == "MEDIUM":
                    color = (0, 255, 255)

                else:
                    color = (0, 0, 255)

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Label text
                label = f"{class_name} {confidence:.2f}"

                # Draw label background
                cv2.rectangle(
                    frame,
                    (x1, y1 - 25),
                    (x1 + 140, y1),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x1 + 5, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Black dashboard background
    cv2.rectangle(frame, (0, 0), (350, 130), (0, 0, 0), -1)

    # Vehicle count
    cv2.putText(
        frame,
        f"Vehicle Count : {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    # Density display
    cv2.putText(
        frame,
        f"Traffic Density : {density}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    # FPS display
    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

    # Show output window
    cv2.imshow("Smart Traffic Management System", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Program stopped by user.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()