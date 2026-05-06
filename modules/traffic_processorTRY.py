import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Vehicle classes
vehicle_classes = ["car", "motorcycle", "bus", "truck"]


def process_video(video_path, frame_placeholder):

    # Open video
    cap = cv2.VideoCapture(video_path)

    # FPS calculation
    prev_time = 0

    while True:

        # Read frame
        ret, frame = cap.read()

        # Stop video
        if not ret:
            break

        # Resize frame
        frame = cv2.resize(frame, (900, 500))

        # Default values
        vehicle_count = 0
        density = "LOW"
        green_signal_time = 10

        # Run YOLO
        results = model(frame, verbose=False)

        # Detection loop
        for result in results:

            boxes = result.boxes

            for box in boxes:

                confidence = float(box.conf[0])

                if confidence < 0.4:
                    continue

                class_id = int(box.cls[0])

                class_name = model.names[class_id]

                if class_name in vehicle_classes:

                    vehicle_count += 1

                    # Density logic
                    if vehicle_count <= 5:
                        density = "LOW"
                        green_signal_time = 10

                    elif vehicle_count <= 15:
                        density = "MEDIUM"
                        green_signal_time = 25

                    else:
                        density = "HIGH"
                        green_signal_time = 45

                    # Coordinates
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    # Colors
                    if density == "LOW":
                        color = (0, 255, 0)

                    elif density == "MEDIUM":
                        color = (0, 255, 255)

                    else:
                        color = (0, 0, 255)

                    # Draw box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

        # FPS calculation
        current_time = time.time()

        fps = 1 / (current_time - prev_time)

        prev_time = current_time

        # Dashboard overlay
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

        # Dashboard text
        cv2.putText(
            frame,
            f"Vehicles : {vehicle_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Density : {density}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Signal : {green_signal_time} sec",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"FPS : {int(fps)}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        # Convert BGR → RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Display in Streamlit
        frame_placeholder.image(
            frame,
            channels="RGB",
            use_container_width=True
        )

    cap.release()