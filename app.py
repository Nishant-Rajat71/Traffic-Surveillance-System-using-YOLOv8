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
        # Track minimum frames before counting as valid vehicle
        self.min_detection_frames = 5
        
    @st.cache_resource
    def load_model(_self, model_name="yolov8n.pt"):
        """Load YOLOv8 model"""
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
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
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
        """Enhanced tracking with stricter matching"""
        current_detections = []
        
        for detection in detections:
            box, label, confidence = detection
            centroid = self.get_centroid(box)
            matched = False
            best_match_id = None
            best_match_score = 0
            
            # Try to match with existing tracks
            for vehicle_id, tracked_data in list(self.tracked_vehicles.items()):
                last_box = tracked_data['last_box']
                last_centroid = tracked_data['last_centroid']
                last_frame = tracked_data['last_frame']
                
                # Only match if not too many frames have passed
                if frame_number - last_frame <= 5:  # Increased tolerance
                    # Calculate IoU
                    iou = self.calculate_iou(box, last_box)
                    
                    # Calculate centroid distance
                    centroid_dist = self.calculate_distance(centroid, last_centroid)
                    max_movement = 150  # Maximum pixels a vehicle can move between frames
                    
                    # Match if same class, good IoU OR close centroid distance
                    if tracked_data['label'] == label:
                        if iou > 0.3 or centroid_dist < max_movement:
                            # Combined score (IoU weighted more)
                            score = iou * 0.7 + (1 - min(centroid_dist / max_movement, 1)) * 0.3
                            
                            if score > best_match_score:
                                best_match_score = score
                                best_match_id = vehicle_id
            
            # If good match found, update existing track
            if best_match_id is not None and best_match_score > 0.3:
                self.tracked_vehicles[best_match_id]['last_box'] = box
                self.tracked_vehicles[best_match_id]['last_centroid'] = centroid
                self.tracked_vehicles[best_match_id]['last_frame'] = frame_number
                self.tracked_vehicles[best_match_id]['confidence'].append(confidence)
                self.tracked_vehicles[best_match_id]['detection_count'] += 1
                current_detections.append((best_match_id, box, label, confidence))
                matched = True
            
            # Create new track if no match
            if not matched:
                vehicle_id = self.next_vehicle_id
                self.next_vehicle_id += 1
                self.tracked_vehicles[vehicle_id] = {
                    'label': label,
                    'last_box': box,
                    'last_centroid': centroid,
                    'last_frame': frame_number,
                    'first_frame': frame_number,
                    'confidence': [confidence],
                    'detection_count': 1  # Track how many times detected
                }
                current_detections.append((vehicle_id, box, label, confidence))
        
        # Clean up stale tracks
        stale_ids = [vid for vid, data in self.tracked_vehicles.items() 
                    if frame_number - data['last_frame'] > 30]
        for vid in stale_ids:
            del self.tracked_vehicles[vid]
        
        return current_detections
    
    def get_valid_vehicles(self):
        """Get only vehicles that were detected enough times (reduces false positives)"""
        valid_vehicles = {}
        for vid, data in self.tracked_vehicles.items():
            # Only count if detected at least min_detection_frames times
            if data['detection_count'] >= self.min_detection_frames:
                valid_vehicles[vid] = data
        return valid_vehicles

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

def process_video(video_path, system, fps, conf_threshold, progress_bar, status_text):
    """Process video with improved tracking"""
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    all_speeds = []
    frame_stats = []
    vehicle_speed_records = {}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection with configurable confidence
        results = system.model(
            frame,
            conf=conf_threshold,  # User-configurable
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
        
        # Get valid vehicles (detected enough times)
        valid_vehicles = system.get_valid_vehicles()
        unique_vehicles = set(valid_vehicles.keys())
        vehicle_class_counts = defaultdict(set)
        for vid, data in valid_vehicles.items():
            vehicle_class_counts[data['label']].add(vid)
        
        # Frame-level counts
        frame_counts = defaultdict(int)
        frame_speeds = []
        
        for vehicle_id, box, label, confidence in tracked_detections:
            x1, y1, x2, y2 = box
            
            # Frame-level count
            frame_counts[label] += 1
            
            # Calculate speed
            centroid = system.get_centroid(box)
            speed = system.calculate_speed(vehicle_id, centroid, frame_count, fps)
            
            # Track speed per vehicle
            if speed is not None:
                frame_speeds.append(speed)
                all_speeds.append(speed)
                if vehicle_id not in vehicle_speed_records:
                    vehicle_speed_records[vehicle_id] = {
                        'label': label,
                        'speeds': [],
                        'first_frame': frame_count
                    }
                vehicle_speed_records[vehicle_id]['speeds'].append(speed)
            
            # Draw bounding box (color based on validity)
            if vehicle_id in valid_vehicles:
                color = get_class_color(label)
            else:
                color = (128, 128, 128)  # Gray for unconfirmed vehicles
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            display_label = f'{label} {confidence:.2f}'
            if speed is not None:
                display_label += f' | {speed:.1f} km/h'
            
            # Show detection count for debugging
            detection_count = system.tracked_vehicles[vehicle_id]['detection_count']
            if detection_count < system.min_detection_frames:
                display_label += f' ({detection_count}/{system.min_detection_frames})'
            
            cv2.putText(frame, display_label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        # Add overlay with statistics
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Detections in Frame: {sum(frame_counts.values())}', (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f'Valid Unique Vehicles: {len(unique_vehicles)}', (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        cv2.putText(frame, f'(Min {system.min_detection_frames} detections required)', (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        if frame_speeds:
            cv2.putText(frame, f'Avg Speed (frame): {np.mean(frame_speeds):.1f} km/h', (20, 155),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        
        if all_speeds:
            cv2.putText(frame, f'Overall Avg Speed: {np.mean(all_speeds):.1f} km/h', (20, 185),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 150), 2)
        
        # Save frame statistics
        stats = dict(frame_counts)
        stats['frame_number'] = frame_count
        stats['detections_in_frame'] = sum(frame_counts.values())
        stats['valid_unique_vehicles'] = len(unique_vehicles)
        if frame_speeds:
            stats['avg_speed_frame'] = np.mean(frame_speeds)
        if all_speeds:
            stats['avg_speed_overall'] = np.mean(all_speeds)
        frame_stats.append(stats)
        
        out.write(frame)
        frame_count += 1
        
        # Update progress
        if frame_count % 10 == 0:
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing: {frame_count}/{total_frames} frames | Valid Vehicles: {len(unique_vehicles)}")
    
    cap.release()
    out.release()
    
    # Final counts (only valid vehicles)
    total_detections = {k: len(v) for k, v in vehicle_class_counts.items()}
    
    return output_path, total_detections, all_speeds, frame_stats, vehicle_speed_records

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
            help="Larger models are more accurate but slower. Start with 'n' (nano) for fast results."
        )
        
        st.markdown("**Model Comparison:**")
        st.markdown("""
        - **n (nano)**: Fastest, 37% accuracy
        - **s (small)**: Fast, 45% accuracy
        - **m (medium)**: Balanced, 50% accuracy
        - **l (large)**: Slower, 53% accuracy
        - **x (extra-large)**: Slowest, 54% accuracy
        """)
        
        st.markdown("---")
        
        conf_threshold = st.slider(
            "🎯 Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.4,
            step=0.05,
            help="Higher = fewer false positives but might miss some vehicles. Lower = detect more but with false positives."
        )
        
        meter_per_pixel = st.number_input(
            "📏 Calibration (meter/pixel)",
            min_value=0.01,
            max_value=1.0,
            value=0.05,
            step=0.01,
            help="Measure a known distance in video (e.g., lane = 3.5m), count pixels, divide: 3.5/70 = 0.05"
        )
        
        fps = st.number_input(
            "🎬 Video FPS",
            min_value=1,
            max_value=120,
            value=30,
            help="Match your video's actual frame rate"
        )
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("""
        **This system:**
        - ✅ Detects vehicles with adjustable confidence
        - ✅ Requires 5+ detections to count as valid vehicle
        - ✅ Classifies: car, truck, bus, motorcycle, bicycle
        - ✅ Measures speed using centroid tracking
        - ✅ Reduces false positives with enhanced tracking
        
        **Tip:** If count is too high, increase confidence threshold to 0.5-0.6
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
            
            if abs(video_fps - fps) > 2:
                st.warning(f"⚠️ Video FPS is {video_fps:.1f} but you set {fps}. Update for accurate speed!")
        
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
                output_path, detections, speeds, frame_stats, vehicle_records = process_video(
                    video_path, system, fps, conf_threshold, progress_bar, status_text
                )
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            # Display results
            st.markdown("---")
            st.header("📊 Analysis Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_vehicles = sum(detections.values())
                st.metric("🚗 Valid Unique Vehicles", total_vehicles)
            
            with col2:
                if speeds:
                    st.metric("⚡ Avg Speed", f"{np.mean(speeds):.1f} km/h")
                else:
                    st.metric("⚡ Avg Speed", "N/A")
            
            with col3:
                if speeds:
                    st.metric("🚀 Max Speed", f"{np.max(speeds):.1f} km/h")
                else:
                    st.metric("🚀 Max Speed", "N/A")
            
            with col4:
                st.metric("📋 Vehicle Types", len(detections))
            
            st.info(f"ℹ️ **Note:** Only vehicles detected in {system.min_detection_frames}+ frames are counted to reduce false positives.")
            
            # Vehicle breakdown
            st.markdown("### 🚗 Vehicle Distribution")
            if detections:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    df_detections = pd.DataFrame([
                        {"Vehicle Type": k.capitalize(), "Count": v}
                        for k, v in sorted(detections.items(), key=lambda x: x[1], reverse=True)
                    ])
                    st.dataframe(df_detections, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("**Breakdown:**")
                    total = sum(detections.values())
                    for vehicle, count in sorted(detections.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / total) * 100 if total > 0 else 0
                        st.write(f"**{vehicle.capitalize()}:** {count} ({percentage:.1f}%)")
            else:
                st.warning("No valid vehicles detected. Try lowering the confidence threshold.")
            
            # Speed statistics
            if speeds:
                st.markdown("### 🏎️ Speed Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Mean:** {np.mean(speeds):.2f} km/h")
                    st.write(f"**Median:** {np.median(speeds):.2f} km/h")
                    st.write(f"**Std Dev:** {np.std(speeds):.2f} km/h")
                    st.write(f"**Min:** {np.min(speeds):.2f} km/h")
                    st.write(f"**Max:** {np.max(speeds):.2f} km/h")
                
                with col2:
                    slow = sum(1 for s in speeds if s < 40)
                    medium = sum(1 for s in speeds if 40 <= s < 80)
                    fast = sum(1 for s in speeds if s >= 80)
                    total_speed_measurements = len(speeds)
                    
                    st.write(f"**Total Measurements:** {total_speed_measurements}")
                    st.write(f"**Slow (<40 km/h):** {slow} ({slow/total_speed_measurements*100:.1f}%)")
                    st.write(f"**Medium (40-80 km/h):** {medium} ({medium/total_speed_measurements*100:.1f}%)")
                    st.write(f"**Fast (≥80 km/h):** {fast} ({fast/total_speed_measurements*100:.1f}%)")
            
            # Download results
            st.markdown("### 📥 Download Results")
            
            col1, col2, col3 = st.columns(3)
            
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
                        "⬇️ Download Frame Statistics",
                        csv,
                        file_name="frame_statistics.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if vehicle_records:
                    vehicle_data = []
                    for vid, data in vehicle_records.items():
                        if data['speeds']:
                            vehicle_data.append({
                                'vehicle_id': vid,
                                'type': data['label'],
                                'first_frame': data['first_frame'],
                                'avg_speed_kmh': np.mean(data['speeds']),
                                'max_speed_kmh': np.max(data['speeds']),
                                'measurements': len(data['speeds'])
                            })
                    
                    if vehicle_data:
                        df_vehicles = pd.DataFrame(vehicle_data)
                        csv_vehicles = df_vehicles.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Vehicle Speeds",
                            csv_vehicles,
                            file_name="individual_vehicle_speeds.csv",
                            mime="text/csv"
                        )
            
            # Display processed video
            st.markdown("### 🎬 Processed Video")
            st.info("💡 Gray boxes = unconfirmed vehicles (< 5 detections). Colored boxes = valid tracked vehicles.")
            st.video(output_path)
            
            # Cleanup
            try:
                os.unlink(video_path)
            except:
                pass

if __name__ == "__main__":
    main()
