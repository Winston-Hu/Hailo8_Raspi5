"""
use imageCapture.py and readraw_detection.py
if there is a car in snapshoot(we have cut it) in raw image
rotate the milesight to that preset position(use http packege)
"""

import json
import logging
import os
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2

# =========================
# 路径配置
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.json"

RAWIMAGE_DIR = PROJECT_ROOT / "rawimage"
TMPIMAGE_DIR = PROJECT_ROOT / "rawimage_tmp"
LOG_DIR = PROJECT_ROOT / "logs"

RAWIMAGE_DIR.mkdir(parents=True, exist_ok=True)
TMPIMAGE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 日志配置：5MB * 2 个轮转
# =========================

logger = logging.getLogger("app_moving_camera")
logger.setLevel(logging.INFO)

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

file_handler = RotatingFileHandler(
    LOG_DIR / "app_moving_camera.log",
    maxBytes=5 * 1024 * 1024,
    backupCount=2,
    encoding="utf-8",
)
file_handler.setFormatter(log_formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.propagate = False


# =========================
# 配置加载
# =========================

def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def get_rtsp_url(cam: dict) -> str:
    """
    "rtsp://admin:31415926Pi@192.168.1.101:554/Streaming/Channels/101"
    """
    return (
        f"rtsp://{cam['username']}:{cam['password']}"
        f"@{cam['ip']}:{cam['port']}/Streaming/Channels/{cam['channel']}"
    )


# =========================
# Camera_imageCapture：抓帧并保存到 rawimage/
# =========================

def capture_one_frame(rtsp_url: str) -> Path | None:
    """
    从 Camera_imageCapture 抓取一张图片。
    先写入 rawimage_tmp/ 临时文件，校验后原子移动到 rawimage/，
    避免 0KB 或半写文件被 watcher 使用。
    成功返回保存的 Path，失败返回 None。
    """
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    logger.info("[Capture] 尝试连接 RTSP: %s", rtsp_url)

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logger.error("[Capture] 无法打开 RTSP 流，请检查 IP / 账号 / 密码 / 网络")
        return None

    tmp_path: Path | None = None

    try:
        # 丢弃前几帧，防止灰帧
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            logger.error(
                "[Capture] 读取帧失败：ret=%s, frame=%s",
                ret,
                None if frame is None else frame.shape,
            )
            return None

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        stem = f"snapshot_{ts}"
        tmp_path = TMPIMAGE_DIR / f"{stem}.jpg"
        final_path = RAWIMAGE_DIR / f"{stem}.jpg"

        ok = cv2.imwrite(str(tmp_path), frame)
        if not ok:
            logger.error("[Capture] cv2.imwrite 写入失败: %s", tmp_path)
            tmp_path.unlink(missing_ok=True)
            return None

        if tmp_path.stat().st_size == 0:
            logger.error("[Capture] 写出了 0 字节文件（已删除）: %s", tmp_path)
            tmp_path.unlink(missing_ok=True)
            return None

        # 原子移动到 rawimage/
        tmp_path.replace(final_path)

        h, w = frame.shape[:2]
        logger.info(
            "[Capture] 成功保存快照: %s (分辨率=%dx%d, 大小=%d bytes)",
            final_path.name, w, h, final_path.stat().st_size,
        )
        return final_path

    except Exception as e:
        logger.exception("[Capture] 抓拍过程中发生异常: %s", e)
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        return None

    finally:
        cap.release()


# =========================
# 裁剪快照
# =========================

def crop_and_save(original_path: Path, snapshots_cfg: dict) -> bool:
    """
    按 config.json 中 snapshots 的四个坐标取 bounding box，
    裁剪 original_path 并保存为 rawimage/sub_<原文件名>。
    """
    pts = [
        snapshots_cfg["left_up"],
        snapshots_cfg["left_bottom"],
        snapshots_cfg["right_up"],
        snapshots_cfg["right_bottom"],
    ]
    x_min = min(p[0] for p in pts)
    x_max = max(p[0] for p in pts)
    y_min = min(p[1] for p in pts)
    y_max = max(p[1] for p in pts)

    frame = cv2.imread(str(original_path))
    if frame is None:
        logger.error("[Crop] 无法读取图片: %s", original_path)
        return False

    cropped = frame[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        logger.error("[Crop] 裁剪区域为空，请检查坐标配置")
        return False

    sub_path = RAWIMAGE_DIR / f"sub_{original_path.name}"
    ok = cv2.imwrite(str(sub_path), cropped)
    if not ok:
        logger.error("[Crop] 保存裁剪图失败: %s", sub_path)
        return False

    h, w = cropped.shape[:2]
    logger.info("[Crop] 裁剪保存: %s (分辨率=%dx%d)", sub_path.name, w, h)
    return True


# =========================
# 旧图清理：只保留最新的 keep_count 张
# =========================

def cleanup_old_images(directory: Path, prefix: str, keep_count: int = 1):
    """
    删除 directory 中以 prefix 开头的 .jpg，只保留最新的 keep_count 张。
    """
    files = sorted(
        directory.glob(f"{prefix}*.jpg"),
        key=lambda p: p.stat().st_mtime,
    )
    to_delete = files[:-keep_count] if len(files) > keep_count else []
    for f in to_delete:
        f.unlink(missing_ok=True)
        logger.info("[Cleanup] 删除旧图: %s", f.name)


# =========================
# Camera_Moving：TODO 云台控制
# =========================

def handle_moving_camera(rtsp_url: str):
    # TODO: 对 Camera_Moving 的处理逻辑（云台控制等）
    pass


# =========================
# 主循环
# =========================

def main():
    config = load_config()
    last_reload = time.time()
    last_capture = 0.0
    capture_interval = 10  # 抓帧间隔（秒），与原 imageCapture.py 保持一致

    logger.info("[Main] 启动，rawimage 目录: %s", RAWIMAGE_DIR)

    while True:
        now = time.time()

        # 每 30 秒重新读取配置
        if now - last_reload >= config["config_reload_interval"]:
            config = load_config()
            last_reload = now
            logger.info("[Main] 配置已重新加载")

        cam_capture = next((c for c in config["cameras"] if c["name"] == "Camera_imageCapture" and c["enabled"]), None)
        cam_moving = next((c for c in config["cameras"] if c["name"] == "Camera_Moving" and c["enabled"]), None)

        # Camera_imageCapture：每 capture_interval 秒抓一帧
        if cam_capture and now - last_capture >= capture_interval:
            url = get_rtsp_url(cam_capture)
            saved_path = capture_one_frame(url)
            if saved_path:
                crop_and_save(saved_path, config["snapshots"])
                cleanup_old_images(RAWIMAGE_DIR, "snapshot_")
                cleanup_old_images(RAWIMAGE_DIR, "sub_")
            else:
                logger.warning("[Main] 本次抓拍失败，下次重试")
            last_capture = now

        # Camera_Moving：云台控制
        if cam_moving:
            url = get_rtsp_url(cam_moving)
            handle_moving_camera(url)

        time.sleep(1)


if __name__ == "__main__":
    main()
