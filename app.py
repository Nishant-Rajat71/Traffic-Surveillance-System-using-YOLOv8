import streamlit as st
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from ultralytics import YOLO
import tempfile
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Traffic Surveillance System",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

class TrafficSurveillanceSystem:
    def __init__(self):
        self.vehicle_classes = ["car", "truck", "bus", "motorbike", "bicycle"]
        self.model = None
        self.tracked_vehicles = {}
        self.next_vehicle_id = 0
        self.speed_data = defaultdict(lambda: {
            'coordinates': deque(maxlen=30),
            'timestamps': deque(maxlen=30),
            'speeds': [],
            'frames': deque(maxlen=30)
        })
        self.meter_per_pixel = 0.05
        
    @st.cache_resource
    def load_model(_self, model_name="yolov8n.pt"):
        """Load YOLOv8 model (using nano for faster inference in web app)"""
        return YOLO(model_name)
    
    def set_calibration(self, meter_per_pixel):
        """Set calibration for speed measurement"""
        self.meter_per_pixel = meter_per_pixel
    
    def get_centroid(self, box):
        """Get the centroid (bottom-center) of bounding box"""
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = y2
        return (cx, cy)
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def calculate_speed(self, vehicle_id, current_position, current_frame, fps):
        """Calculate vehicle speed based on position history"""
        speed_info = self.speed_data[vehicle_id]
        
        speed_info['coordinates'].append(current_position)
        speed_info['frames'].append(current_frame)
        
        if len(speed_info['coordinates']) < max(2, int(fps / 2)):
            return None
        
        start_pos = speed_info['coordinates'][0]
        end_pos = speed_info['coordinates'][-1]
        
        pixel_distance = np.sqrt(
            (end_pos[0] - start_pos[0])**2 + 
            (end_pos[1] - start_pos[1])**2
        )
        
        distance_meters = pixel_distance * self.meter_per_pixel
        
        frame_diff = speed_info['frames'][-1] - speed_info['frames'][0]
        time_seconds = frame_diff / fps
        
        if time_seconds > 0:
            speed_ms = distance_meters / time_seconds
            speed_kmh = speed_ms * 3.6
            
            if 0 < speed_kmh < 200:
                speed_info['speeds'].append(speed_kmh)
                return speed_kmh
        
        return None
    
    def track_vehicles(self, detections, frame_number):
        """Track vehicles across frames"""
        current_detections = []
        
        for detection in detections:
            box, label, confidence = detection
            matched = False
            
            for vehicle_id, tracked_data in list(self.tracked_vehicles.items()):
                last_box = tracked_data['last_box']
                last_frame = tracked_data['last_frame']
                
                if frame_number - last_frame <= 3:
                    iou = self.calculate_iou(box, last_box)
                    if iou > 0.4 and tracked_data['label'] == label:
                        self.tracked_vehicles[vehicle_id]['last_box'] = box
                        self.tracked_vehicles[vehicle_id]['last_frame'] = frame_number
                        self.tracked_vehicles[vehicle_id]['confidence'].append(confidence)
                        matched = True
                        current_detections.append((vehicle_id, box, label, confidence))
                        break
            
            if not matched:
                vehicle_id = self.next_vehicle_id
                self.next_vehicle_id += 1
                self.tracked_vehicles[vehicle_id] = {
                    'label': label,
                    'last_box': box,
                    'last_frame': frame_number,
                    'first_frame': frame_number,
                    'confidence': [confidence]
                }
                current_detections.append((vehicle_id, box, label, confidence))
        
        stale_ids = [vid for vid, data in self.tracked_vehicles.items() 
                    if frame_number - data['last_frame'] > 30]
        for vid in stale_ids:
            del self.tracked_vehicles[vid]
        
        return current_detections

def get_class_color(class_name):
    """Get consistent color for each vehicle class"""
    colors = {
        'car': (0, 255, 0),
        'truck': (0, 0, 255),
        'bus': (255, 0, 0),
        'motorbike': (255, 255, 0),
        'bicycle': (255, 0, 255)
    }
    return colors.get(class_name, (255, 255, 255))

def process_video(video_path, system, fps, progress_bar, status_text):
    """Process video and return results"""
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = defaultdict(int)
    all_speeds = []
    frame_stats = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = system.model(
            frame,
            conf=0.25,
            iou=0.45,
            verbose=False
        )[0]
        
        # Collect detections
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = system.model.names[cls_id]
                confidence = float(box.conf[0])
                
                if label in system.vehicle_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2, y2], label, confidence))
        
        # Track vehicles
        tracked_detections = system.track_vehicles(detections, frame_count)
        
        frame_counts = defaultdict(int)
        frame_speeds = []
        
        for vehicle_id, box, label, confidence in tracked_detections:
            x1, y1, x2, y2 = box
            
            frame_counts[label] += 1
            total_detections[label] += 1
            
            # Calculate speed
            centroid = system.get_centroid(box)
            speed = system.calculate_speed(vehicle_id, centroid, frame_count, fps)
            
            # Draw bounding box
            color = get_class_color(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            display_label = f'{label} {confidence:.2f}'
            if speed is not None:
                display_label += f' | {speed:.1f} km/h'
                frame_speeds.append(speed)
                all_speeds.append(speed)
            
            cv2.putText(frame, display_label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f'Frame: {frame_count}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Total Vehicles: {sum(frame_counts.values())}', (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if frame_speeds:
            cv2.putText(frame, f'Avg Speed: {np.mean(frame_speeds):.1f} km/h', (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        # Save stats
        stats = dict(frame_counts)
        stats['frame_number'] = frame_count
        stats['total_vehicles'] = sum(frame_counts.values())
        if frame_speeds:
            stats['avg_speed'] = np.mean(frame_speeds)
        frame_stats.append(stats)
        
        out.write(frame)
        frame_count += 1
        
        # Update progress
        if frame_count % 10 == 0:
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing: {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    return output_path, total_detections, all_speeds, frame_stats

# Streamlit UI
def main():
    st.title("🚦 Traffic Surveillance & Speed Measurement System")
    st.markdown("### AI-Powered Vehicle Detection, Classification, and Speed Estimation")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        model_choice = st.selectbox(
            "Select YOLO Model",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
            help="Larger models are more accurate but slower"
        )
        
        meter_per_pixel = st.number_input(
            "Calibration (meter/pixel)",
            min_value=0.01,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="Adjust based on camera setup for accurate speed"
        )
        
        fps = st.number_input(
            "Video FPS",
            min_value=1,
            max_value=60,
            value=30,
            help="Frames per second of input video"
        )
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("""
        This system uses YOLOv8 to:
        - Detect vehicles in real-time
        - Classify: car, truck, bus, motorcycle, bicycle
        - Measure speed using centroid tracking
        - Generate analytics and statistics
        """)
    
    # Main content
    uploaded_file = st.file_uploader(
        "📹 Upload Traffic Video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a traffic surveillance video for analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Display video info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(video_path)
        
        with col2:
            st.markdown("### 📹 Video Info")
            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / video_fps if video_fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            st.write(f"**Resolution:** {width}x{height}")
            st.write(f"**FPS:** {video_fps:.2f}")
            st.write(f"**Duration:** {duration:.2f}s")
            st.write(f"**Total Frames:** {total_frames}")
        
        # Process button
        if st.button("🚀 Start Analysis", type="primary"):
            with st.spinner("Loading model..."):
                system = TrafficSurveillanceSystem()
                system.model = system.load_model(model_choice)
                system.set_calibration(meter_per_pixel)
            
            st.success("✅ Model loaded successfully!")
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process video
            with st.spinner("Processing video..."):
                output_path, detections, speeds, frame_stats = process_video(
                    video_path, system, fps, progress_bar, status_text
                )
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            # Display results
            st.markdown("---")
            st.header("📊 Analysis Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Vehicles", sum(detections.values()))
            
            with col2:
                if speeds:
                    st.metric("Avg Speed", f"{np.mean(speeds):.1f} km/h")
                else:
                    st.metric("Avg Speed", "N/A")
            
            with col3:
                if speeds:
                    st.metric("Max Speed", f"{np.max(speeds):.1f} km/h")
                else:
                    st.metric("Max Speed", "N/A")
            
            with col4:
                st.metric("Unique Classes", len(detections))
            
            # Vehicle breakdown
            st.markdown("### 🚗 Vehicle Distribution")
            if detections:
                df_detections = pd.DataFrame([
                    {"Vehicle Type": k.capitalize(), "Count": v}
                    for k, v in detections.items()
                ])
                st.dataframe(df_detections, use_container_width=True)
            
            # Speed statistics
            if speeds:
                st.markdown("### 🏎️ Speed Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Mean:** {np.mean(speeds):.2f} km/h")
                    st.write(f"**Median:** {np.median(speeds):.2f} km/h")
                    st.write(f"**Std Dev:** {np.std(speeds):.2f} km/h")
                
                with col2:
                    slow = sum(1 for s in speeds if s < 40)
                    medium = sum(1 for s in speeds if 40 <= s < 80)
                    fast = sum(1 for s in speeds if s >= 80)
                    
                    st.write(f"**Slow (<40 km/h):** {slow}")
                    st.write(f"**Medium (40-80 km/h):** {medium}")
                    st.write(f"**Fast (≥80 km/h):** {fast}")
            
            # Download results
            st.markdown("### 📥 Download Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                with open(output_path, 'rb') as f:
                    st.download_button(
                        "⬇️ Download Processed Video",
                        f,
                        file_name=f"processed_{uploaded_file.name}",
                        mime="video/mp4"
                    )
            
            with col2:
                if frame_stats:
                    df_stats = pd.DataFrame(frame_stats)
                    csv = df_stats.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download Statistics CSV",
                        csv,
                        file_name="traffic_statistics.csv",
                        mime="text/csv"
                    )
            
            # Display processed video
            st.markdown("### 🎬 Processed Video")
            st.video(output_path)
            
            # Cleanup
            try:
                os.unlink(video_path)
                os.unlink(output_path)
            except:
                pass

if __name__ == "__main__":
    main()
