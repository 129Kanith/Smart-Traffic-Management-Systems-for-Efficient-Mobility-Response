import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open video
video_path = r"K:\TMSYoLo\videos\Video 1.mp4"
cap = cv2.VideoCapture(video_path)

# Vehicle classes
vehicle_classes = ["car", "motorcycle", "bus", "truck"]

while True:

    # Read frame
    ret, frame = cap.read()

    # Stop video if ended
    if not ret:
        break

    # Vehicle counter
    vehicle_count = 0

    # Run YOLO detection
    results = model(frame)

    # Loop through detections
    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Class ID
            class_id = int(box.cls[0])

            # Class name
            class_name = model.names[class_id]

            # Detect only vehicles
            if class_name in vehicle_classes:

                # Increase count
                vehicle_count += 1

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

                # Draw label
                cv2.putText(
                    frame,
                    class_name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # Show vehicle count
    cv2.putText(
        frame,
        f"Vehicle Count: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        3
    )

    # Display frame
    cv2.imshow("Smart Traffic Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()