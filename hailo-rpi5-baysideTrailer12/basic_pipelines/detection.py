from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.detection.detection_pipeline import GStreamerDetectionApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example

        # 下面这几个属性是我们自己加的，用于保存结果
        self.resimage_dir = None
        self.reslabel_dir = None
        self.output_stem = "frame"
        self.saved_once = False  # 只保存第一帧

    def new_function(self):  # New function example
        return "The meaning of life is: "

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data: user_app_callback_class):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # 帧计数
    user_data.increment()
    frame_idx = user_data.get_count()
    string_to_print = f"Frame count: {frame_idx}\n"

    if frame_idx <= 2:
        print(f"[DEBUG] Warm-up frame {frame_idx}, skip detections")
        return Gst.PadProbeReturn.OK

    # 读取 caps，尝试从 buffer 拿出整帧图像（RGB）
    format, width, height = get_caps_from_pad(pad)
    frame = None
    img_w = img_h = None
    if format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)
        # frame.shape 是 H, W, C
        img_h, img_w = frame.shape[:2]

    # 拿检测结果
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    print(f"[DEBUG] Frame {frame_idx} -> detections: {len(detections)}")

    # 准备 YOLO-like 文本行
    yolo_lines = []

    detection_count = 0
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        print(f"Detection: label -> {label}, confidence -> {confidence}")

        # Hailo 的 bbox 一般是 0~1 归一化坐标
        x_min_n = bbox.xmin()
        y_min_n = bbox.ymin()
        x_max_n = bbox.xmax()
        y_max_n = bbox.ymax()

        w_n = max(0.0, x_max_n - x_min_n)
        h_n = max(0.0, y_max_n - y_min_n)
        cx_n = x_min_n + w_n / 2.0
        cy_n = y_min_n + h_n / 2.0

        # 记录一行 YOLO-like： class cx cy w h score
        yolo_lines.append(
            f"{label} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f} {confidence:.4f}"
        )

        # person 计数 + ID（你原来的逻辑）
        if label == "person":
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            string_to_print += (
                f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n"
            )
            detection_count += 1

        # 在 frame 上画框
        if frame is not None and img_w is not None and img_h is not None:
            x1 = int(x_min_n * img_w)
            y1 = int(y_min_n * img_h)
            x2 = int(x_max_n * img_w)
            y2 = int(y_max_n * img_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    # 如果拿到了图像，就加上文字、存到 user_data 里（给 GStreamer 显示）
    frame_bgr = None
    if frame is not None:
        cv2.putText(
            frame,
            f"Detections(person): {detection_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{user_data.new_function()} {user_data.new_variable}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame_bgr)

    # ---------- 保存图片 & label（只保存一次） ----------
    if (
        (not user_data.saved_once)
        and (user_data.resimage_dir is not None)
        and (user_data.reslabel_dir is not None)
    ):
        img_out_path = user_data.resimage_dir / f"{user_data.output_stem}.jpg"
        label_out_path = user_data.reslabel_dir / f"{user_data.output_stem}.txt"

        # 1) 保存图片（如果有 frame）
        if frame_bgr is not None:
            try:
                cv2.imwrite(str(img_out_path), frame_bgr)
                print(f"[INFO] Saved detection image to: {img_out_path}")
            except Exception as e:
                print(f"[WARN] Failed to save image: {e}")
        else:
            print("[WARN] No frame available, skip saving image")

        # 2) 保存 label（即使没有 frame 也可以保存）
        try:
            with open(label_out_path, "w", encoding="utf-8") as f:
                for line in yolo_lines:
                    f.write(line + "\n")
            print(f"[INFO] Saved detection labels to: {label_out_path}")
        except Exception as e:
            print(f"[WARN] Failed to save label file: {e}")

        user_data.saved_once = True

    print(string_to_print)

    # directly exit, we do not need to loop on one picture
    if frame_idx >= 4:
        print("[INFO] Reached frame 4, exiting...")
        os._exit(0)

    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    # 仓库根目录（和 .env 在一起）
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    # 准备输出目录
    resimage_dir = project_root / "resimage"
    reslabel_dir = project_root / "reslabel"
    resimage_dir.mkdir(exist_ok=True)
    reslabel_dir.mkdir(exist_ok=True)

    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    user_data.use_frame = True
    # 先给一下输出目录
    user_data.resimage_dir = resimage_dir
    user_data.reslabel_dir = reslabel_dir

    app = GStreamerDetectionApp(app_callback, user_data)

    # 根据输入源推一个输出文件名（图片的话就是原文件名的 stem）
    src = app.video_source
    if isinstance(src, str):
        # 对于 "rawimage/snapshot_1765xxxx.jpg" -> "snapshot_1765xxxx"
        user_data.output_stem = Path(src).stem
    else:
        user_data.output_stem = "frame"

    # 1) 打印命令行参数里的 hef-path（你没传就是 None）
    print(f"[INFO] options_menu.hef_path = {app.options_menu.hef_path}")

    # 2) 打印 GStreamerDetectionApp 实际使用的 hef_path
    print(f"[INFO] resolved hef_path     = {app.hef_path}")

    # 打印 labels 的 JSON 路径
    print(f"[INFO] labels_json           = {app.labels_json}")

    # 3) 如果是目录，把目录里所有 .hef 列出来
    hef_path = Path(app.hef_path)
    if hef_path.is_dir():
        print("[INFO] hef_path 是目录，包含的 HEF 模型：")
        for p in sorted(hef_path.glob("*.hef")):
            print(f"       - {p}")
    elif hef_path.is_file():
        print(f"[INFO] hef_path 是文件，将使用该 HEF： {hef_path}")
    else:
        print("[WARN] hef_path 既不是文件也不是目录，请检查配置")

    app.run()
