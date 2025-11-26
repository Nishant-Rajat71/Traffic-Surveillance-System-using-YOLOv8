import streamlit as st
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import tempfile
import os

# ---------------------- PAGE CONFIG ---------------------- #
st.set_page_config(
    page_title="Traffic Surveillance with Speed Detection",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    padding: 0rem 1rem;
}
.stButton>button {
    width: 100%;
    background-color: #FF4B4B;
    color: white;
}
.stButton>button:hover {
    background-color: #FF6B6B;
    color: white;
}
h1 {
    color: #FF4B4B;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- CORE CLASS ---------------------- #
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
        self.meter_per_pixel = 0.05       # you can tune in sidebar
        self.min_detection_frames = 5      # min frames before ID is trusted

    @st.cache_resource
    def load_model(_self, model_name="yolov8x.pt"):
        return YOLO(model_name)

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

    def calculate_distance(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def calculate_speed(self, vehicle_id, current_position, current_frame, fps):
        s = self.speed_data[vehicle_id]
        s['coordinates'].append(current_position)
        s['frames'].append(current_frame)
        if len(s['coordinates']) < max(2, int(fps/2)):
            return None
        start_pos = s['coordinates'][0]
        end_pos   = s['coordinates'][-1]
        pixel_dist = np.sqrt((end_pos[0]-start_pos[0])**2 +
                             (end_pos[1]-start_pos[1])**2)
        dist_m = pixel_dist * self.meter_per_pixel
        frame_diff = s['frames'][-1] - s['frames'][0]
        t = frame_diff / fps
        if t <= 0:
            return None
        speed_kmh = (dist_m / t) * 3.6
        if 0 < speed_kmh < 200:
            s['speeds'].append(speed_kmh)
            return speed_kmh
        return None

    def track_vehicles(self, detections, frame_number):
        current_detections = []
        for box, label, conf in detections:
            centroid = self.get_centroid(box)
            best_id, best_score = None, 0
            for vid, data in list(self.tracked_vehicles.items()):
                if frame_number - data['last_frame'] > 5:
                    continue
                iou = self.calculate_iou(box, data['last_box'])
                cdist = self.calculate_distance(centroid, data['last_centroid'])
                if data['label'] == label and (iou > 0.3 or cdist < 150):
                    score = iou*0.7 + (1 - min(cdist/150, 1))*0.3
                    if score > best_score:
                        best_score, best_id = score, vid
            if best_id is not None and best_score > 0.3:
                self.tracked_vehicles[best_id]['last_box'] = box
                self.tracked_vehicles[best_id]['last_centroid'] = centroid
                self.tracked_vehicles[best_id]['last_frame'] = frame_number
                self.tracked_vehicles[best_id]['detection_count'] += 1
                self.tracked_vehicles[best_id]['confidence'].append(conf)
                current_detections.append((best_id, box, label, conf))
            else:
                vid = self.next_vehicle_id
                self.next_vehicle_id += 1
                self.tracked_vehicles[vid] = {
                    'label': label,
                    'last_box': box,
                    'last_centroid': centroid,
                    'last_frame': frame_number,
                    'first_frame': frame_number,
                    'confidence': [conf],
                    'detection_count': 1
                }
                current_detections.append((vid, box, label, conf))

        # cleanup
        stale = [vid for vid, d in self.tracked_vehicles.items()
                 if frame_number - d['last_frame'] > 30]
        for vid in stale:
            del self.tracked_vehicles[vid]
        return current_detections

    def get_valid_vehicles(self):
        return {vid: d for vid, d in self.tracked_vehicles.items()
                if d['detection_count'] >= self.min_detection_frames}


def get_class_color(c):
    colors = {
        'car': (0,255,0),
        'truck': (0,0,255),
        'bus': (255,0,0),
        'motorbike': (255,255,0),
        'bicycle': (255,0,255)
    }
    return colors.get(c, (255,255,255))

# ---------------------- PROCESSING ---------------------- #
def process_video(video_path, system, fps, conf_threshold, progress_bar, status_text):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = 0
    all_speeds = []
    frame_stats = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = system.model(frame, conf=conf_threshold, iou=0.45, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label  = system.model.names[cls_id]
                conf   = float(box.conf[0])
                if label in system.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2, y2], label, conf))

        tracked = system.track_vehicles(detections, frame_count)
        valid   = system.get_valid_vehicles()

        frame_counts = defaultdict(int)
        frame_speeds = []

        for vid, box, label, conf in tracked:
            x1, y1, x2, y2 = box
            frame_counts[label] += 1
            centroid = system.get_centroid(box)
            speed = system.calculate_speed(vid, centroid, frame_count, fps)
            if speed is not None:
                frame_speeds.append(speed)
                all_speeds.append(speed)

            color = get_class_color(label) if vid in valid else (128,128,128)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            txt = f"{label} {conf:.2f}"
            if speed is not None:
                txt += f" | {speed:.1f} km/h"
            cv2.putText(frame, txt, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # overlay (same logic as Colab, but without total unique/overall avg)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10,10), (450,130), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, 'Traffic Analysis with Speed Detection', (20,35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (20,65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, f'Vehicles in Frame: {sum(frame_counts.values())}', (20,95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        if frame_speeds:
            cv2.putText(frame, f'Avg Speed: {np.mean(frame_speeds):.1f} km/h', (20,125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,100), 2)

        stats = dict(frame_counts)
        stats['frame_number']      = frame_count
        stats['vehicles_in_frame'] = sum(frame_counts.values())
        if frame_speeds:
            stats['avg_speed'] = float(np.mean(frame_speeds))
        frame_stats.append(stats)

        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            progress_bar.progress(frame_count/total_frames)
            status_text.text(
                f"Processing {frame_count}/{total_frames} frames"
            )

    cap.release()
    out.release()
    return out_path, frame_stats, all_speeds

# ---------------------- STREAMLIT UI ---------------------- #
def main():
    st.title("🚦 Traffic Surveillance & Speed Measurement System")
    st.markdown("Real-time vehicle detection, counting, and speed estimation (YOLOv8).")

    with st.sidebar:
        st.header("⚙️ Configuration")
        model_choice = st.selectbox(
            "YOLOv8 Model",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
            index=4  # default x
        )
        conf_th = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
        meter_per_pixel = st.number_input(
            "Calibration (meter/pixel)", 0.01, 1.0, 0.05, 0.01,
            help="Real-world meters represented by one pixel."
        )
        fps = st.number_input(
            "Output Video FPS", 1, 120, 30,
            help="Set equal to source FPS for correct speed."
        )

    uploaded = st.file_uploader(
        "Upload traffic video (mp4, avi, mov, mkv)", 
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded is not None:
        tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tmp_in.write(uploaded.read())
        video_path = tmp_in.name

        col1, col2 = st.columns([2,1])
        with col1:
            st.subheader("Original Video")
            st.video(video_path)
        with col2:
            cap = cv2.VideoCapture(video_path)
            v_fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            dur = frames / v_fps if v_fps > 0 else 0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            st.subheader("Video Info")
            st.write(f"Resolution: **{w}x{h}**")
            st.write(f"FPS: **{v_fps:.2f}**")
            st.write(f"Frames: **{frames}**")
            st.write(f"Duration: **{dur:.2f} s**")

        if st.button("🚀 Start Processing", type="primary"):
            with st.spinner("Loading YOLOv8 model..."):
                system = TrafficSurveillanceSystem()
                system.model = system.load_model(model_choice)
                system.set_calibration(meter_per_pixel)

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            with st.spinner("Processing video..."):
                out_path, frame_stats, speeds = process_video(
                    video_path, system, fps, conf_th,
                    progress_bar, status_text
                )

            st.success("Processing complete!")

            st.subheader("Processed Video")
            st.video(out_path)

            st.download_button(
                "⬇️ Download Processed Video",
                data=open(out_path, "rb").read(),
                file_name=f"processed_{uploaded.name}",
                mime="video/mp4"
            )

            # quick summary (similar to Colab prints)
            st.subheader("Quick Summary")
            if frame_stats:
                df = pd.DataFrame(frame_stats)
                st.write(f"Total frames: **{len(df)}**")
                st.write(f"Average vehicles per frame: **{df['vehicles_in_frame'].mean():.2f}**")
            if speeds:
                st.write(f"Average speed: **{np.mean(speeds):.2f} km/h**")
                st.write(f"Max speed: **{np.max(speeds):.2f} km/h**")

            try:
                os.unlink(video_path)
            except:
                pass

if __name__ == "__main__":
    main()
