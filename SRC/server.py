import base64
import os
import time
import logging
import pickle
from io import BytesIO
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from transformers import DetrImageProcessor, DetrForObjectDetection
from pathlib import Path
import torch
import random
from PIL import Image

matplotlib.use("agg")

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = "SRC/UPLOADS/"
app.config["ALLOWED_EXTENSIONS"] = {"mp4", "avi", "mov", "mkv", "flv"}

logging.basicConfig(level=logging.INFO)

detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

label_color_map = {}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in list(
        app.config["ALLOWED_EXTENSIONS"]
    )


def detect_objects(frame, frame_idx, fps):
    pil_image = Image.fromarray(frame)
    inputs = detr_processor(images=pil_image, return_tensors="pt")
    outputs = detr_model(**inputs)

    target_sizes = torch.tensor([pil_image.size[::-1]])
    detection_threshold = 0.99
    results = detr_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=detection_threshold
    )[0]

    formatted_results = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        label_name = detr_model.config.id2label[label.item()]
        score_value = score.item()
        box_value = box.tolist()
        frame_time = frame_idx / fps
        if score_value > detection_threshold:
            formatted_results.append(
                {
                    "label": label_name,
                    "score": round(score_value, 3),
                    "box": [round(i, 2) for i in box_value],
                    "time": frame_time,
                }
            )
            if label_name not in label_color_map:
                label_color_map[label_name] = "#{:06x}".format(
                    random.randint(0, 0xFFFFFF)
                )

    return formatted_results


def analyze_video(video_path, skip_frames=1, resize_factor=1):
    global label_color_map
    label_color_map = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    original_video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    original_video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    video_basename = Path(video_path).stem
    timeline_path = Path(app.config["UPLOAD_FOLDER"]) / f"{video_basename}_timeline.pkl"
    objects_path = Path(app.config["UPLOAD_FOLDER"]) / f"{video_basename}_objects.pkl"
    label_color_map_path = (
        Path(app.config["UPLOAD_FOLDER"]) / f"{video_basename}_label_color_map.pkl"
    )

    if (
        timeline_path.exists()
        and objects_path.exists()
        and label_color_map_path.exists()
    ):
        with open(timeline_path, "rb") as f:
            movement_events, video_duration = pickle.load(f)
        with open(objects_path, "rb") as f:
            object_detections = pickle.load(f)
        with open(label_color_map_path, "rb") as f:
            label_color_map = pickle.load(f)

        return (
            movement_events,
            video_duration,
            object_detections,
            label_color_map,
            original_video_width,
            original_video_height,
        )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_duration = frame_count / fps * 1000

    movement_events = []
    object_detections = []
    fgbg = cv2.createBackgroundSubtractorMOG2()

    intensities = []
    frame_idx = 0
    last_intensity = 0

    detection_interval = 1 * fps

    while True:
        ret, frame = cap.read()
        frame_idx += 1

        if not ret:
            break
        if frame_idx % detection_interval == 0:
            detected_objects = detect_objects(frame, frame_idx, fps)
            for obj in detected_objects:
                label = obj["label"]
                if label not in label_color_map:
                    label_color_map[label] = "#{:06x}".format(
                        random.randint(0, 0xFFFFFF)
                    )
                object_detections.append(obj)

        frame = cv2.resize(
            frame,
            (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor)),
        )
        fgmask = fgbg.apply(frame)
        movement_intensity = cv2.norm(fgmask, cv2.NORM_L1) / (fgmask.size * 255)

        if frame_idx % skip_frames == 0:
            interpolated_values = np.linspace(
                last_intensity, movement_intensity, skip_frames
            )
            intensities.extend(interpolated_values)
            last_intensity = movement_intensity

    threshold = np.percentile(intensities, 55)

    for idx, intensity in enumerate(intensities):
        if intensity > threshold:
            movement_events.append(
                {
                    "timestamp": int((idx / len(intensities)) * video_duration),
                    "intensity": intensity,
                }
            )
    cap.release()

    save_object_detections_png(object_detections, video_duration, video_basename)
    with open(timeline_path, "wb") as f:
        pickle.dump((movement_events, video_duration), f)
    with open(objects_path, "wb") as f:
        pickle.dump(object_detections, f)
    with open(label_color_map_path, "wb") as f:
        pickle.dump(label_color_map, f)

    return (
        movement_events,
        video_duration,
        object_detections,
        label_color_map,
        original_video_width,
        original_video_height,
    )


def generate_waveform(
    movement_events,
    video_duration,
    height_range=480,
    figsize=(16, 4),
    show_axis=False,
):
    if not movement_events:
        raise ValueError("No movement events found in the video.")

    time_interval = video_duration / (figsize[0] * 100)
    num_bins = int(np.ceil(video_duration / time_interval))

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    waveform_bins = np.zeros(num_bins, dtype=float)

    for event in movement_events:
        bin_idx = int(event["timestamp"] / time_interval)
        waveform_bins[bin_idx] += event["intensity"]

    clip_value = np.percentile(waveform_bins, 98)
    waveform_bins = np.clip(waveform_bins, 0, clip_value)
    max_val = np.max(waveform_bins)
    waveform_bins = waveform_bins * height_range / max_val
    median_val = np.median(waveform_bins)
    waveform_bins -= median_val
    opposite_waveform_bins = -waveform_bins

    ax.fill_between(
        np.arange(num_bins),
        waveform_bins,
        where=waveform_bins >= 0,
        interpolate=True,
        color="darkgoldenrod",
        alpha=1,
    )
    ax.fill_between(
        np.arange(num_bins),
        opposite_waveform_bins,
        where=opposite_waveform_bins <= 0,
        interpolate=True,
        color="darkgoldenrod",
        alpha=1,
    )
    ax.plot(waveform_bins, color="darkgoldenrod")
    ax.plot(opposite_waveform_bins, color="darkgoldenrod")

    ax.set_xlim([0, num_bins - 1])
    ax.set_ylim([-height_range, height_range])

    x_ticks = np.linspace(0, num_bins - 1, 7, dtype=int)
    x_tick_labels = [
        f"{int(x//60)}:{int(x%60):02d}"
        for x in x_ticks * (video_duration / (num_bins - 1)) / 1000
    ]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)

    if show_axis:
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.tick_params(left=False, bottom=True)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks([])
    else:
        ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig


def save_object_detections_png(object_detections, video_duration, filename):
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    video_duration_sec = video_duration / 1000

    object_detection_times = {}
    for detection in object_detections:
        label = detection["label"]
        detection_time = detection["time"]
        if label not in object_detection_times:
            object_detection_times[label] = []
        object_detection_times[label].append(detection_time)

    sorted_labels = sorted(object_detection_times.keys())
    for i, label in enumerate(sorted_labels):
        for start_time in object_detection_times[label]:
            end_time = start_time + 5
            position_start = (start_time / video_duration_sec) * 100
            position_end = (end_time / video_duration_sec) * 100
            ax.plot(
                [position_start, position_end],
                [i, i],
                color=label_color_map.get(label, "#FFFFFF"),
                linewidth=20,
            )

    ax.set_xlim([0, 100])
    ax.set_ylim([-1, len(sorted_labels)])
    ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    object_detection_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"{filename}_objects.png"
    )
    fig.savefig(object_detection_path, format="png")
    plt.close(fig)


def generate_timeline_png(movement_events, video_duration):
    fig = generate_waveform(movement_events, video_duration, figsize=(16, 4))

    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    plot_bytes = buffer.read()
    buffer.close()

    plt.close(fig)

    return plot_bytes


@app.route("/api/detect-events", methods=["POST"])
def detect_events():
    try:
        start_time = time.time()
        result_files = {}

        files = request.files.getlist("video[]")
        if not files:
            return make_response(jsonify({"error": "No video uploaded"}), 400)

        for video in files:
            if isinstance(video, FileStorage) and allowed_file(video.filename):
                filename = secure_filename(video.filename)
                video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

                try:
                    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
                        os.makedirs(app.config["UPLOAD_FOLDER"])
                    video.save(video_path)
                except Exception as e:
                    app.logger.exception("An error occurred during video file saving")
                    return make_response(jsonify({"error": str(e)}), 500)

                try:
                    print(f"Starting analysis for {filename}...")
                    (
                        movement_events,
                        video_duration,
                        object_detections,
                        label_color_map,
                        original_video_width,
                        original_video_height,
                    ) = analyze_video(video_path)

                    plot_bytes = generate_timeline_png(movement_events, video_duration)
                    timeline_filename = f"{os.path.splitext(filename)[0]}_timeline.png"
                    timeline_path = os.path.join(
                        app.config["UPLOAD_FOLDER"], timeline_filename
                    )
                    with open(timeline_path, "wb") as f:
                        f.write(plot_bytes)

                    object_detections_filename = (
                        f"{os.path.splitext(filename)[0]}_objects.png"
                    )
                    object_detections_path = os.path.join(
                        app.config["UPLOAD_FOLDER"], object_detections_filename
                    )
                    with open(object_detections_path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode()

                    object_detections_data = []
                    for detection in object_detections:
                        object_detections_data.append(
                            {
                                "label": detection["label"],
                                "box": detection["box"],
                                "time": detection["time"],
                            }
                        )

                    result_files[filename] = {
                        "timeline": base64.b64encode(plot_bytes).decode(),
                        "object_detections": encoded_string,
                        "object_detections_data": object_detections_data,
                        "video_duration": video_duration,
                        "label_color_map": label_color_map,
                        "original_video_width": original_video_width,
                        "original_video_height": original_video_height,
                    }
                except Exception as e:
                    app.logger.exception("An error occurred during video analysis")
                    return make_response(jsonify({"error": str(e)}), 500)

                print(
                    f"Processing time for {filename}: {time.time() - start_time:.2f} seconds"
                )
            else:
                return make_response(jsonify({"error": "Invalid video format"}), 400)

        return jsonify(result_files)

    except Exception as e:
        app.logger.error(f"Unexpected error: {e}")
        return make_response(jsonify({"error": "An unexpected error occurred"}), 500)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
