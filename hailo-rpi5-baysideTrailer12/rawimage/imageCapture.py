#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler

import cv2

# =========================
# 基本配置
# =========================

# RTSP 地址（101 = 主码流）
RTSP_URL = "rtsp://admin:1234qwer@172.16.210.50:554/Streaming/Channels/102"

# 抓拍间隔（秒）
CAPTURE_INTERVAL = 10

# 目录结构：
#   project_root/
#       rawimage/imageCapture.py
#       rawimage_tmp/         # 临时目录（新建）
#       logs/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAWIMAGE_DIR = Path(__file__).resolve().parent
TMPIMAGE_DIR = PROJECT_ROOT / "rawimage_tmp"
LOG_DIR = PROJECT_ROOT / "logs"

RAWIMAGE_DIR.mkdir(parents=True, exist_ok=True)
TMPIMAGE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

IMG_LOG_FILE = LOG_DIR / "imgCapture_log.log"

# =========================
# 日志配置：5MB * 2 个轮转
# =========================

logger = logging.getLogger("imgCapture")
logger.setLevel(logging.INFO)

log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

file_handler = RotatingFileHandler(
    IMG_LOG_FILE,
    maxBytes=5 * 1024 * 1024,  # 5 MB
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
# 抓拍核心逻辑
# =========================

def capture_one_frame() -> bool:
    """
    抓取一张图片，先写入 rawimage_tmp 目录中的临时 jpg，
    校验无误后再移动到 rawimage，避免 0KB 或半写文件被 watcher 使用。
    返回 True 表示本次抓拍成功，False 表示失败。
    """
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    logger.info("尝试连接 RTSP: %s", RTSP_URL)

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        logger.error("无法打开 RTSP 视频流，请检查 IP / 账号 / 密码 / 网络")
        return False

    tmp_path: Path | None = None

    try:
        # 丢弃前几帧，防止灰帧
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            logger.error(
                "读取帧失败：ret=%s, frame=%s",
                ret,
                None if frame is None else frame.shape,
            )
            return False

        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        stem = f"snapshot_{ts}"

        # 临时文件放在 rawimage_tmp，扩展名必须是 .jpg
        tmp_path = TMPIMAGE_DIR / f"{stem}.jpg"
        final_path = RAWIMAGE_DIR / f"{stem}.jpg"

        # 写临时 jpg
        ok = cv2.imwrite(str(tmp_path), frame)
        if not ok:
            logger.error("cv2.imwrite 写入失败: %s", tmp_path)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return False

        size = tmp_path.stat().st_size
        if size == 0:
            logger.error("写出了 0 字节文件（已删除）: %s", tmp_path)
            tmp_path.unlink(missing_ok=True)
            return False

        # 原子移动到 rawimage 目录
        tmp_path.replace(final_path)

        h, w = frame.shape[:2]
        logger.info(
            "成功保存快照: %s (分辨率=%dx%d, 大小=%d bytes)",
            final_path.name,
            w,
            h,
            size,
        )
        return True

    except Exception as e:
        logger.exception("抓拍或保存过程中发生异常: %s", e)
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
        return False

    finally:
        cap.release()


def main():
    logger.info(
        "启动 imgCapture 循环：间隔=%s 秒，rawimage 目录=%s",
        CAPTURE_INTERVAL,
        RAWIMAGE_DIR,
    )

    try:
        while True:
            start = time.time()
            ok = capture_one_frame()
            elapsed = time.time() - start

            sleep_time = max(0.0, CAPTURE_INTERVAL - elapsed)
            if not ok:
                logger.warning("本次抓拍失败，将在 %.1f 秒后重试", sleep_time)

            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，imgCapture 停止。")


if __name__ == "__main__":
    main()
