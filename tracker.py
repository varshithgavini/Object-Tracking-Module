import csv
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


project_root = Path(__file__).resolve().parent
model_dir = project_root / "models"
video_dir = project_root / "videos"
output_dir = project_root / "output"

model_dir.mkdir(parents=True, exist_ok=True)
video_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "yolov8n.pt"
tracking_output_dir = output_dir / "tracking_output"
tracking_output_dir.mkdir(parents=True, exist_ok=True)

if not model_path.is_file():
    print(f"Model not found: {model_path}")
    print("Place yolov8n.pt inside the models folder and run again.")
    raise SystemExit(1)

model = YOLO(str(model_path))
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

supported_formats = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
video_files = sorted(
    [
        file for file in video_dir.iterdir()
        if file.is_file() and file.suffix.lower() in supported_formats
    ]
)

if not video_files:
    print(f"No video files found in: {video_dir}")
    print("Add your CCTV video to the videos folder.")
    raise SystemExit(1)

video_path = video_files[0]
print(f"Processing video: {video_path.name}")

csv_path = tracking_output_dir / "tracking_data.csv"
output_video_path = tracking_output_dir / "tracked_people.mp4"

start_time = time.time()
frame_count = 0
tracked_detection_count = 0

result_stream = model.track(
    source=str(video_path),
    project=str(output_dir),
    name="tracking_output",
    exist_ok=True,
    tracker="bytetrack.yaml",
    classes=[0],
    conf=0.25,
    iou=0.5,
    imgsz=640,
    vid_stride=1,
    persist=True,
    device=device,
    stream=True,
    save=False,
    show=False,
    verbose=False,
)

with open(csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Frame", "Track_ID", "X1", "Y1", "X2", "Y2", "Confidence"])

    first_result = next(iter(result_stream), None)
    if first_result is None:
        print("No frames were processed.")
        raise SystemExit(1)

    frame_shape = first_result.orig_img.shape
    height, width = frame_shape[:2]
    fps = 25
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print("Could not open video writer for output MP4.")
        raise SystemExit(1)

    for result in [first_result] + list(result_stream):
        frame_count += 1
        frame = result.orig_img.copy()

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for box, track_id, confidence in zip(boxes, track_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID {int(track_id)}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                csv_writer.writerow([
                    frame_count,
                    int(track_id),
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(x2), 2),
                    round(float(y2), 2),
                    round(float(confidence), 4),
                ])

                tracked_detection_count += 1

        person_count = 0
        if result.boxes is not None and result.boxes.id is not None:
            person_count = len(result.boxes.id)

        cv2.putText(
            frame,
            f"Persons: {person_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        video_writer.write(frame)

    video_writer.release()

end_time = time.time()
processing_time = end_time - start_time

print("\nTracking completed successfully!")
print(f"Video processed: {video_path.name}")
print(f"Frames processed: {frame_count}")
print(f"Tracked detections saved: {tracked_detection_count}")
print(f"Processing time: {processing_time:.2f} seconds")
print(f"Tracked video folder: {tracking_output_dir}")
print(f"Tracked video file: {output_video_path}")
print(f"Tracking CSV file: {csv_path}")