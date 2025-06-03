import cv2
import uuid
import time
import numpy as np
import gradio as gr
from ultralytics import YOLO

# Cache for loaded models
model_cache = {}

def load_model(model_name):
    if model_name not in model_cache:
        model_cache[model_name] = YOLO(f"{model_name}.pt")
    return model_cache[model_name]

def detect_objects_on_video(video, model_name, conf_threshold, iou_threshold, frame_step, progress=gr.Progress()):
    video_path = video if isinstance(video, str) else video.name
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    model = load_model(model_name)

    out_path = f"output_{uuid.uuid4().hex}.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_id = 0
    detections_summary = {}
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_step == 0:
            results = model(frame, conf=conf_threshold, iou=iou_threshold)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls]} {conf:.2f}"

                detections_summary[model.names[cls]] = detections_summary.get(model.names[cls], 0) + 1

                color = tuple(np.random.randint(0, 255, size=3).tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        frame_id += 1

        # Yield progress update
        progress(progress=frame_id / total_frames)

    cap.release()
    out.release()
    end_time = time.time()

    duration = round(end_time - start_time, 2)
    summary = [[k, v] for k, v in detections_summary.items()]
    summary.append(["Total Time (s)", duration])

    return out_path, summary

# Gradio Interface
demo = gr.Interface(
    fn=detect_objects_on_video,
    inputs=[
        gr.Video(label="Upload Video"),
        gr.Radio(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"], value="yolov8n", label="YOLOv8 Model"),
        gr.Slider(0.1, 1.0, value=0.3, label="Confidence Threshold"),
        gr.Slider(0.1, 1.0, value=0.5, label="IoU Threshold"),
        gr.Slider(1, 10, value=1, step=1, label="Process Every Nth Frame"),
    ],
    outputs=[
        gr.Video(label="Detected Video"),
        gr.Dataframe(headers=["Object Class", "Count"], label="Detection Summary")
    ],
    title="🎯 YOLOv8 Video Object Detection App",
    description="Upload a video, choose a YOLOv8 model, set thresholds, and get object detections with bounding boxes and stats. Powered by Ultralytics & Gradio.",
examples=[
        ["examples/example.mp4", "yolov8n", 0.3, 0.5, 1],
    ]
)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
