import streamlit as st
import cv2
import tempfile
import time
import collections
import numpy as np
from ultralytics import YOLO
import supervision as sv

st.set_page_config(
    page_title="Real-Time Multi-Object Tracking",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 Real-Time Multi-Object Tracking System")
st.caption("Powered by YOLOv8, ByteTrack, and Supervision Engine")

st.sidebar.header("🕹️ Model Controls")
model_size = st.sidebar.selectbox("YOLOv8 Model Variant", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)
conf_thresh = st.sidebar.slider("Detection Confidence Threshold", 0.10, 1.00, 0.25, 0.05)
iou_thresh = st.sidebar.slider("ByteTrack IoU Threshold", 0.10, 1.00, 0.70, 0.05)

COCO_CLASSES = {"Person": 0, "Bicycle": 1, "Car": 2, "Motorcycle": 3, "Bus": 5, "Truck": 7}
selected_class_names = st.sidebar.multiselect("Target Classes", list(COCO_CLASSES.keys()), default=["Person", "Car", "Bus", "Truck"])
target_class_ids = [COCO_CLASSES[name] for name in selected_class_names]

@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(model_size)

uploaded_file = st.file_uploader("Upload Video (.mp4, .avi)", type=["mp4", "avi", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st_frame = st.empty()
    with col2:
        st.subheader("Analytics")
        fps_metric = st.metric("Speed", "0.0 FPS")
        targets_metric = st.metric("Active Targets", "0")
        class_placeholder = st.empty()

    if st.button("🚀 Start Tracking Execution", use_container_width=True):
        tracker = sv.ByteTrack(track_activation_threshold=conf_thresh, minimum_matching_threshold=iou_thresh)
        cap = cv2.VideoCapture(tfile.name)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        output_temp_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        out = cv2.VideoWriter(output_temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        trajectory_memory = collections.defaultdict(lambda: collections.deque(maxlen=30))
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        
        prev_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = model(frame, conf=conf_thresh, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[np.isin(detections.class_id, target_class_ids)]
            detections = tracker.update_with_detections(detections=detections)
            
            curr_time = time.time()
            instant_fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            
            class_counts = collections.defaultdict(int)
            
            if detections.tracker_id is not None:
                labels = []
                for xyxy, class_id, tracker_id, conf in zip(detections.xyxy, detections.class_id, detections.tracker_id, detections.confidence):
                    cx, cy = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
                    trajectory_memory[tracker_id].append((cx, cy))
                    
                    c_name = model.model.names[class_id]
                    class_counts[c_name] += 1
                    labels.append(f"#{tracker_id} {c_name} {conf:.2f}")
                    
                    pts = np.array(trajectory_memory[tracker_id], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
                    
                frame = box_annotator.annotate(scene=frame, detections=detections)
                frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
            
            out.write(frame)
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            fps_metric.metric("Speed", f"{instant_fps:.1f} FPS")
            targets_metric.metric("Active Targets", f"{len(detections)}")
            
            breakdown_md = "**Class Tally:**\n" + "\n".join([f"- **{k}**: {v}" for k, v in class_counts.items()])
            class_placeholder.markdown(breakdown_md)

        cap.release()
        out.release()
        st.success("✅ Tracking Finished!")
        
        with open(output_temp_path, "rb") as file:
            st.download_button("📥 Download Tracked Video", file, file_name="tracked_output.mp4", mime="video/mp4", use_container_width=True)
