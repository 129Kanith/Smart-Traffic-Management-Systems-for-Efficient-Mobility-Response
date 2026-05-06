import cv2
import time
import os
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from ultralytics import YOLO

# =====================================================
# LOAD YOLO MODEL
# =====================================================
# yolov8m.pt -> Better accuracy
# yolov8n.pt -> Faster but less accurate
model = YOLO("yolov8m.pt")

# =====================================================
# VEHICLE CLASSES
# =====================================================
vehicle_classes = [
    "car",
    "motorcycle",
    "bus",
    "truck"
]

# =====================================================
# VALID CLASSES
# =====================================================
valid_classes = vehicle_classes

# =====================================================
# MAIN VIDEO PROCESSING FUNCTION
# =====================================================
def process_video(
    video_path,
    frame_placeholder,
    analytics_placeholder,
    graphs_placeholder,
    enable_heatmap=False
):

    # =====================================================
    # OPEN VIDEO
    # =====================================================
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # FPS
    prev_time = 0

    # Analytics
    vehicle_history = []
    signal_history = []
    frame_history = []

    # Heatmap
    heatmap_accum = np.zeros(
        (720, 1280),
        dtype=np.float32
    )

    # Session report
    session_data = []

    # Optimization
    graph_update_interval = 10
    frame_skip = 2
    frame_count = 0

    # Output folder
    os.makedirs("outputs", exist_ok=True)

    # =====================================================
    # MAIN LOOP
    # =====================================================
    while True:

        # Read frame
        ret, frame = cap.read()

        if not ret:
            break

        # Frame skip
        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        # Resize
        frame = cv2.resize(
            frame,
            (1280, 720)
        )

        # =====================================================
        # DEFAULT VALUES
        # =====================================================
        vehicle_count = 0

        emergency_detected = False
        ambulance_detected = False
        police_detected = False

        lane_counts = {
            "Lane 1": 0,
            "Lane 2": 0,
            "Lane 3": 0
        }

        # =====================================================
        # DRAW LANE LINES
        # =====================================================
        cv2.line(
            frame,
            (426, 0),
            (426, 720),
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (852, 0),
            (852, 720),
            (255, 255, 255),
            2
        )

        # =====================================================
        # YOLO TRACKING
        # =====================================================
        results = model.track(
            frame,
            persist=True,
            verbose=False,
            conf=0.35,
            imgsz=960,
            tracker="bytetrack.yaml"
        )

        # =====================================================
        # DETECTION LOOP
        # =====================================================
        for result in results:

            boxes = result.boxes

            for box in boxes:

                # Confidence
                confidence = float(box.conf[0])

                if confidence < 0.35:
                    continue

                # Class
                class_id = int(box.cls[0])

                class_name = model.names[class_id]

                # Ignore unwanted classes
                if class_name not in valid_classes:
                    continue

                # Vehicle count
                vehicle_count += 1

                # Tracking ID
                track_id = -1

                if box.id is not None:
                    track_id = int(box.id.item())

                # Coordinates
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # =====================================================
                # LANE COUNTING
                # =====================================================
                if center_x < 426:
                    lane_counts["Lane 1"] += 1

                elif center_x < 852:
                    lane_counts["Lane 2"] += 1

                else:
                    lane_counts["Lane 3"] += 1

                # =====================================================
                # DEFAULT COLOR
                # =====================================================
                color = (0, 255, 0)

                # =====================================================
                # SMART AMBULANCE / POLICE DETECTION
                # =====================================================
                if class_name in ["car", "truck", "bus"]:

                    roi = frame[y1:y2, x1:x2]

                    if roi.size > 0:

                        hsv = cv2.cvtColor(
                            roi,
                            cv2.COLOR_BGR2HSV
                        )

                        # =================================================
                        # RED LIGHT DETECTION
                        # =================================================
                        lower_red1 = np.array(
                            [0, 120, 120]
                        )

                        upper_red1 = np.array(
                            [10, 255, 255]
                        )

                        lower_red2 = np.array(
                            [170, 120, 120]
                        )

                        upper_red2 = np.array(
                            [180, 255, 255]
                        )

                        red_mask1 = cv2.inRange(
                            hsv,
                            lower_red1,
                            upper_red1
                        )

                        red_mask2 = cv2.inRange(
                            hsv,
                            lower_red2,
                            upper_red2
                        )

                        red_mask = red_mask1 + red_mask2

                        # =================================================
                        # BLUE LIGHT DETECTION
                        # =================================================
                        lower_blue = np.array(
                            [100, 120, 120]
                        )

                        upper_blue = np.array(
                            [140, 255, 255]
                        )

                        blue_mask = cv2.inRange(
                            hsv,
                            lower_blue,
                            upper_blue
                        )

                        # =================================================
                        # PIXEL COUNT
                        # =================================================
                        red_pixels = cv2.countNonZero(
                            red_mask
                        )

                        blue_pixels = cv2.countNonZero(
                            blue_mask
                        )

                        # =================================================
                        # AMBULANCE DETECTION
                        # =================================================
                        if red_pixels > 600:

                            ambulance_detected = True
                            emergency_detected = True

                            color = (0, 0, 255)

                            cv2.putText(
                                frame,
                                "AMBULANCE",
                                (x1, y1 - 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                3
                            )

                        # =================================================
                        # POLICE DETECTION
                        # =================================================
                        elif blue_pixels > 600:

                            police_detected = True
                            emergency_detected = True

                            color = (255, 0, 0)

                            cv2.putText(
                                frame,
                                "POLICE VEHICLE",
                                (x1, y1 - 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (255, 0, 0),
                                3
                            )

                # =====================================================
                # NORMAL VEHICLE COLORS
                # =====================================================
                if not emergency_detected:

                    if vehicle_count <= 10:
                        color = (0, 255, 0)

                    elif vehicle_count <= 25:
                        color = (0, 255, 255)

                    else:
                        color = (0, 0, 255)

                # =====================================================
                # HEATMAP
                # =====================================================
                if enable_heatmap:

                    cv2.circle(
                        heatmap_accum,
                        (center_x, center_y),
                        20,
                        1,
                        -1
                    )

                # =====================================================
                # DRAW BOUNDING BOX
                # =====================================================
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # =====================================================
                # LABEL
                # =====================================================
                label = f"{class_name} ID:{track_id}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        # =====================================================
        # DENSITY LOGIC
        # =====================================================
        if ambulance_detected:

            density = "AMBULANCE PRIORITY"

            green_signal_time = 120

        elif police_detected:

            density = "POLICE PRIORITY"

            green_signal_time = 100

        elif vehicle_count <= 10:

            density = "LOW"

            green_signal_time = 15

        elif vehicle_count <= 25:

            density = "MEDIUM"

            green_signal_time = 35

        else:

            density = "HIGH"

            green_signal_time = 60

        # =====================================================
        # LANE DENSITY
        # =====================================================
        lane_densities = {}

        for lane, count in lane_counts.items():

            if count <= 3:
                lane_densities[lane] = "LOW"

            elif count <= 8:
                lane_densities[lane] = "MEDIUM"

            else:
                lane_densities[lane] = "HIGH"

        # =====================================================
        # FPS
        # =====================================================
        current_time = time.time()

        fps = 1 / (current_time - prev_time)

        prev_time = current_time

        # =====================================================
        # STORE ANALYTICS
        # =====================================================
        vehicle_history.append(vehicle_count)

        signal_history.append(green_signal_time)

        frame_history.append(frame_count)

        session_data.append({
            "Frame": frame_count,
            "Vehicles": vehicle_count,
            "Density": density,
            "Signal_Time": green_signal_time
        })

        # Limit history
        if len(vehicle_history) > 30:

            vehicle_history.pop(0)
            signal_history.pop(0)
            frame_history.pop(0)

        # =====================================================
        # SAVE REPORT
        # =====================================================
        if len(session_data) % 30 == 0:

            pd.DataFrame(session_data).to_csv(
                "outputs/traffic_report.csv",
                index=False
            )

        # =====================================================
        # HEATMAP OVERLAY
        # =====================================================
        if enable_heatmap:

            heatmap_blur = cv2.GaussianBlur(
                heatmap_accum,
                (15, 15),
                0
            )

            heatmap_norm = cv2.normalize(
                heatmap_blur,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )

            heatmap_color = cv2.applyColorMap(
                heatmap_norm,
                cv2.COLORMAP_JET
            )

            frame = cv2.addWeighted(
                frame,
                0.7,
                heatmap_color,
                0.3,
                0
            )

            heatmap_accum *= 0.95

        # =====================================================
        # DASHBOARD OVERLAY
        # =====================================================
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (10, 10),
            (360, 260),
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

        # =====================================================
        # AMBULANCE ALERT OVERLAY
        # =====================================================
        if ambulance_detected:

            cv2.rectangle(
                frame,
                (280, 20),
                (1020, 100),
                (0, 0, 255),
                -1
            )

            cv2.putText(
                frame,
                "AMBULANCE PRIORITY ACTIVATED",
                (320, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                3
            )

        # =====================================================
        # POLICE ALERT OVERLAY
        # =====================================================
        elif police_detected:

            cv2.rectangle(
                frame,
                (300, 20),
                (1000, 100),
                (255, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                "POLICE PRIORITY ACTIVATED",
                (350, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                3
            )

        # =====================================================
        # DASHBOARD TEXT
        # =====================================================
        text_color = (255, 255, 255)

        cv2.putText(
            frame,
            f"Vehicles : {vehicle_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2
        )

        cv2.putText(
            frame,
            f"Density : {density}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2
        )

        cv2.putText(
            frame,
            f"Signal : {green_signal_time} sec",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2
        )

        cv2.putText(
            frame,
            f"FPS : {int(fps)}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            text_color,
            2
        )

        # =====================================================
        # LANE DETAILS
        # =====================================================
        y_pos = 180

        for lane in lane_counts:

            lane_text = (
                f"{lane}: "
                f"{lane_counts[lane]} "
                f"({lane_densities[lane]})"
            )

            cv2.putText(
                frame,
                lane_text,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2
            )

            y_pos += 30

        # =====================================================
        # CONVERT BGR TO RGB
        # =====================================================
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # =====================================================
        # SHOW FRAME
        # =====================================================
        frame_placeholder.image(
            frame_rgb,
            channels="RGB",
            width="stretch"
        )

        # =====================================================
        # ANALYTICS PANEL
        # =====================================================
        with analytics_placeholder.container():

            col1, col2 = st.columns(2)

            col1.metric(
                "🚗 Vehicles",
                vehicle_count
            )

            col2.metric(
                "🚦 Density",
                density
            )

            col1.metric(
                "⏱ Signal",
                f"{green_signal_time} sec"
            )

            col2.metric(
                "⚡ FPS",
                int(fps)
            )

            st.markdown("### 🚦 Lane Analysis")

            lane1, lane2, lane3 = st.columns(3)

            lane1.metric(
                "Lane 1",
                lane_counts["Lane 1"]
            )

            lane2.metric(
                "Lane 2",
                lane_counts["Lane 2"]
            )

            lane3.metric(
                "Lane 3",
                lane_counts["Lane 3"]
            )

        # =====================================================
        # GRAPH PANEL
        # =====================================================
        if len(frame_history) % graph_update_interval == 0:

            with graphs_placeholder.container():

                g1, g2 = st.columns(2)

                # Vehicle Graph
                vehicle_df = pd.DataFrame({
                    "Frame": frame_history,
                    "Vehicles": vehicle_history
                })

                vehicle_fig = px.line(
                    vehicle_df,
                    x="Frame",
                    y="Vehicles",
                    title="📈 Vehicle Count Trend"
                )

                g1.plotly_chart(
                    vehicle_fig,
                    width="stretch"
                )

                # Signal Graph
                signal_df = pd.DataFrame({
                    "Frame": frame_history,
                    "Signal": signal_history
                })

                signal_fig = px.line(
                    signal_df,
                    x="Frame",
                    y="Signal",
                    title="🚦 Signal Timing Analysis"
                )

                g2.plotly_chart(
                    signal_fig,
                    width="stretch"
                )

    # =====================================================
    # RELEASE VIDEO
    # =====================================================
    cap.release()