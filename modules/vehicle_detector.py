import cv2
from ultralytics import YOLO
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open traffic video
video_path = r"K:\TMSYoLo\videos\Video 1.mp4"
video_path = r"K:\TMSYoLo\videos\Video 2.mp4"
video_path = r"K:\TMSYoLo\videos\Video 3.mp4"
video_path = r"K:\TMSYoLo\videos\Video 4.mp4"
cap = cv2.VideoCapture(video_path)

# Check video opened or not
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# Vehicle classes from COCO dataset
vehicle_classes = ["car", "motorcycle", "bus", "truck"]

# Emergency vehicle classes
emergency_classes = ["ambulance"]

# FPS calculation
prev_time = 0

while True:

    # Read frame
    ret, frame = cap.read()

    # Stop if video ends
    if not ret:
        print("Video completed.")
        break

    # Resize frame
    frame = cv2.resize(frame, (640, 384))

    # Vehicle counter
    vehicle_count = 0

    # Default values
    density = "LOW"
    green_signal_time = 10
    signal_status = "GREEN"
    emergency_detected = False

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

            # Get class ID
            class_id = int(box.cls[0])

            # Get class name
            class_name = model.names[class_id]

            # -----------------------------
            # VEHICLE DETECTION
            # -----------------------------
            if class_name in vehicle_classes:

                # Increase vehicle count
                vehicle_count += 1

                # Density classification
                if vehicle_count <= 5:
                    density = "LOW"
                    green_signal_time = 10

                elif vehicle_count <= 15:
                    density = "MEDIUM"
                    green_signal_time = 25

                else:
                    density = "HIGH"
                    green_signal_time = 45

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Color based on density
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

                # Minimal label
                cv2.putText(
                    frame,
                    class_name,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1
                )

            # -----------------------------
            # EMERGENCY VEHICLE DETECTION
            # -----------------------------
            if class_name in emergency_classes:

                emergency_detected = True

                # Emergency coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Emergency box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3
                )

                # Emergency label
                cv2.putText(
                    frame,
                    "EMERGENCY",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2
                )

    # -----------------------------
    # EMERGENCY OVERRIDE
    # -----------------------------
    if emergency_detected:
        signal_status = "EMERGENCY GREEN"
        green_signal_time = 60

    # -----------------------------
    # FPS CALCULATION
    # -----------------------------
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # -----------------------------
    # TRANSPARENT DASHBOARD
    # -----------------------------
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (320, 145),
        (0, 0, 0),
        -1
    )

    alpha = 0.6

    frame = cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0
    )

    # -----------------------------
    # DASHBOARD TEXT
    # -----------------------------

    # Vehicle count
    cv2.putText(
        frame,
        f"Vehicles : {vehicle_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    # Density
    cv2.putText(
        frame,
        f"Density : {density}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # Signal timing
    cv2.putText(
        frame,
        f"Signal : {green_signal_time} sec",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    # Signal status
    cv2.putText(
        frame,
        f"Status : {signal_status}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 165, 255),
        2
    )

    # FPS
    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )

    # -----------------------------
    # EMERGENCY ALERT
    # -----------------------------
    if emergency_detected:

        cv2.rectangle(
            frame,
            (470, 10),
            (630, 45),
            (0, 0, 255),
            -1
        )

        cv2.putText(
            frame,
            "EMERGENCY MODE",
            (480, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    # -----------------------------
    # SHOW OUTPUT
    # -----------------------------
    cv2.imshow(
        "Smart Traffic Management System",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Program stopped by user.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()