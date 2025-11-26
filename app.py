import tempfile
import time
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO


# ================== ORIGINAL BACKEND (LIGHTLY ADAPTED) ==================

class TrafficSurveillanceSystem:
    def __init__(self):
        self.vehicle_classes = ["car", "truck", "bus", "motorbike", "bicycle"]
        self.model = None
        self.tracked_vehicles = {}
        self.next_vehicle_id = 0
        self.speed_data = defaultdict(lambda: {
            'coordinates': deque(maxlen=30),
            'frames': deque(maxlen=30),
            'speeds': []
        })
        self.meter_per_pixel = 0.05
        self.min_detection_frames = 5
        self.video_stats = {}
        self.frame_stats = []

    def load_model(self, model_name="yolov8x.pt"):
        if self.model is None:
            self.model = YOLO(model_name)
        return self.model

    def set_calibration(self, meter_per_pixel):
        self.meter_per_pixel = meter_per_pixel

    def get_centroid(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, y2)

    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def calculate_speed(self, vehicle_id, current_position, current_frame, fps):
        speed_info = self.speed_data[vehicle_id]
        speed_info['coordinates'].append(current_position)
        speed_info['frames'].append(current_frame)

        if len(speed_info['coordinates']) < max(2, int(fps / 2)):
            return None

        start_pos = speed_info['coordinates'][0]
        end_pos = speed_info['coordinates'][-1]

        pixel_distance = np.sqrt(
            (end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2
        )
        distance_meters = pixel_distance * self.meter_per_pixel

        frame_diff = speed_info['frames'][-1] - speed_info['frames'][0]
        time_seconds = frame_diff / fps

        if time_seconds > 0:
            speed_kmh = (distance_meters / time_seconds) * 3.6
            if 0 < speed_kmh < 200:
                speed_info['speeds'].append(speed_kmh)
                return speed_kmh
        return None

    def track_vehicles(self, detections, frame_number):
        current_detections = []

        for detection in detections:
            box, label, confidence = detection
            centroid = self.get_centroid(box)
            best_match_id = None
            best_match_score = 0

            for vehicle_id, tracked_data in list(self.tracked_vehicles.items()):
                if frame_number - tracked_data['last_frame'] <= 5:
                    iou = self.calculate_iou(box, tracked_data['last_box'])
                    centroid_dist = self.calculate_distance(
                        centroid, tracked_data['last_centroid']
                    )

                    if tracked_data['label'] == label:
                        if iou > 0.3 or centroid_dist < 150:
                            score = iou * 0.7 + (1 - min(centroid_dist / 150, 1)) * 0.3
                            if score > best_match_score:
                                best_match_score = score
                                best_match_id = vehicle_id

            if best_match_id is not None and best_match_score > 0.3:
                self.tracked_vehicles[best_match_id].update({
                    'last_box': box,
                    'last_centroid': centroid,
                    'last_frame': frame_number,
                    'detection_count': self.tracked_vehicles[best_match_id]['detection_count'] + 1
                })
                self.tracked_vehicles[best_match_id]['confidence'].append(confidence)
                current_detections.append((best_match_id, box, label, confidence))
            else:
                vehicle_id = self.next_vehicle_id
                self.next_vehicle_id += 1
                self.tracked_vehicles[vehicle_id] = {
                    'label': label,
                    'last_box': box,
                    'last_centroid': centroid,
                    'last_frame': frame_number,
                    'first_frame': frame_number,
                    'confidence': [confidence],
                    'detection_count': 1
                }
                current_detections.append((vehicle_id, box, label, confidence))

        stale_ids = [
            vid for vid, data in self.tracked_vehicles.items()
            if frame_number - data['last_frame'] > 30
        ]
        for vid in stale_ids:
            del self.tracked_vehicles[vid]

        return current_detections

    def get_valid_vehicles(self):
        return {
            vid: data for vid, data in self.tracked_vehicles.items()
            if data['detection_count'] >= self.min_detection_frames
        }


def get_class_color(class_name):
    colors = {
        'car': (0, 255, 0),
        'truck': (0, 0, 255),
        'bus': (255, 0, 0),
        'motorbike': (255, 255, 0),
        'bicycle': (255, 0, 255)
    }
    return colors.get(class_name, (255, 255, 255))


def preprocess_video(video_path, system):
    cap = cv2.VideoCapture(video_path)
    video_stats = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS) or 30.0,
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    video_stats['duration'] = (
        video_stats['total_frames'] / video_stats['fps']
        if video_stats['fps'] > 0 else 0
    )
    cap.release()
    system.video_stats = video_stats
    return video_stats


def process_video_with_analysis(video_path, system, fps, conf_threshold):
    cap = cv2.VideoCapture(video_path)
    width = system.video_stats['width']
    height = system.video_stats['height']
    total_frames = system.video_stats['total_frames']

    # temp file for processed video
    temp_out = tempfile.NamedTemporaryFile(
        suffix=".mp4", delete=False
    )
    output_path = temp_out.name
    temp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = system.model(frame, conf=conf_threshold, iou=0.45, verbose=False)[0]

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = system.model.names[cls_id]
                confidence = float(box.conf[0])

                if label in system.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2, y2], label, confidence))

        tracked_detections = system.track_vehicles(detections, frame_count)
        valid_vehicles = system.get_valid_vehicles()

        frame_counts = defaultdict(int)
        frame_speeds = []

        for vehicle_id, box, label, confidence in tracked_detections:
            x1, y1, x2, y2 = box
            frame_counts[label] += 1

            centroid = system.get_centroid(box)
            speed = system.calculate_speed(vehicle_id, centroid, frame_count, fps)

            if speed is not None:
                frame_speeds.append(speed)

            color = get_class_color(label) if vehicle_id in valid_vehicles else (128, 128, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            display_label = f'{label} {confidence:.2f}'
            if speed is not None:
                display_label += f' | {speed:.1f} km/h'
            cv2.putText(
                frame, display_label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # overlay similar to Colab version
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame, 'Traffic Analysis with Speed Detection', (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            frame, f'Frame: {frame_count}/{total_frames}', (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        cv2.putText(
            frame, f'Vehicles in Frame: {sum(frame_counts.values())}', (20, 95),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2
        )

        if frame_speeds:
            cv2.putText(
                frame, f'Avg Speed: {np.mean(frame_speeds):.1f} km/h', (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2
            )

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    return output_path


# ================== STREAMLIT FRONTEND ==================

st.set_page_config(page_title="Traffic Surveillance with YOLOv8", layout="wide")

st.title("Traffic Surveillance System – YOLOv8")

left_col, right_col = st.columns([3, 1])

with right_col:
    st.subheader("About this project")
    st.markdown(
        """
This app performs automatic traffic analysis from CCTV videos using the YOLOv8 object detection model.
It detects and tracks vehicles frame‑by‑frame, estimates their speed, and overlays metrics like vehicle count and average speed directly on the processed video.
Upload any road‑side video clip to quickly visualize traffic behaviour for research, monitoring, or signal‑timing studies.
        """
    )

with left_col:
    uploaded_file = st.file_uploader(
        "Upload a traffic video", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:
        with st.spinner("Saving uploaded video..."):
            # save upload to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_file.read())
                temp_input_path = tmp.name

        st.success("Video uploaded. Starting processing...")

        # sidebar controls
        st.sidebar.header("Settings")
        model_name = st.sidebar.selectbox(
            "YOLOv8 model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
            index=4
        )
        conf_th = st.sidebar.slider(
            "Confidence threshold", 0.1, 0.9, 0.4, 0.05
        )
        meter_per_pixel = st.sidebar.number_input(
            "Meters per pixel (calibration)", min_value=0.001, max_value=1.0,
            value=0.05, step=0.005, format="%.3f"
        )

        # main processing
        system = TrafficSurveillanceSystem()
        system.load_model(model_name)
        system.set_calibration(meter_per_pixel)

        stats = preprocess_video(temp_input_path, system)
        fps = stats["fps"] if stats["fps"] > 0 else 30.0

        start_time = time.time()
        with st.spinner("Processing video with YOLOv8..."):
            processed_path = process_video_with_analysis(
                temp_input_path, system, fps, conf_th
            )
        end_time = time.time()

        st.success(
            f"Processing complete in {end_time - start_time:.1f} seconds. "
            f"Duration: {stats['duration']:.1f} s, FPS used: {fps:.1f}"
        )

        # show only processed video
        with open(processed_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
    else:
        st.info("Please upload a traffic video to begin analysis.")
