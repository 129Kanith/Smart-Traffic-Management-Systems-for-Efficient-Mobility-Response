import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open traffic video
video_path = r"K:\TMSYoLo\videos\Video 1.mp4"
cap = cv2.VideoCapture(video_path)

# Vehicle classes from COCO dataset
vehicle_classes = ["car", "motorcycle", "bus", "truck"]

while True:
    # Read frame
    ret, frame = cap.read()

    # Stop if video ends
    if not ret:
        break

    # Run YOLO detection
    results = model(frame)

    # Get detections
    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Get class ID
            class_id = int(box.cls[0])

            # Get class name
            class_name = model.names[class_id]

            # Detect only vehicles
            if class_name in vehicle_classes:

                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Put vehicle label
                cv2.putText(
                    frame,
                    class_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Show output
    cv2.imshow("Traffic Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()