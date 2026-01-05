#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Watcher 脚本：定期扫描 rawimage 目录，发现最新的 jpg 丢给 basic_pipelines/detection.py 处理。
"""

import argparse
import time
from pathlib import Path
import subprocess
import sys
import logging
from logging.handlers import RotatingFileHandler


# =========================
# 路径配置
# =========================

PROJECT_ROOT = Path("/home/pi/hailo-rpi5-examples").resolve()
RAWIMAGE_DIR = PROJECT_ROOT / "rawimage"
DETECTION_SCRIPT = PROJECT_ROOT / "basic_pipelines" / "detection.py"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

READRAW_LOG_FILE = LOG_DIR / "readraw_detection_log.log"

# =========================
# 日志配置：5MB * 2 个轮转
# =========================

logger = logging.getLogger("readraw_detection")
logger.setLevel(logging.INFO)

log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)

file_handler = RotatingFileHandler(
    READRAW_LOG_FILE,
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
# 原有逻辑
# =========================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=1,
        help="传给 detection.py 的 --frame-rate（默认 1）",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="传给 detection.py 的 --score-threshold（默认 0.5）",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="轮询 rawimage 的间隔秒数（默认 5）",
    )
    return parser.parse_args()


def find_latest_jpg(raw_dir: Path) -> Path | None:
    jpgs = list(raw_dir.glob("*.jpg"))
    if not jpgs:
        return None
    # 按修改时间排序，取最新的
    return max(jpgs, key=lambda p: p.stat().st_mtime)


def run_detection_on_image(
    image_path: Path,
    frame_rate: int,
    score_threshold: float,
) -> bool:
    """调用 detection.py，对指定图片跑一次检测。成功返回 True，失败返回 False。"""
    cmd = [
        sys.executable,
        str(DETECTION_SCRIPT),
        "--input",
        str(image_path),
        "--frame-rate",
        str(frame_rate),
        "--score-threshold",
        str(score_threshold),
    ]
    logger.info("[WATCHER] Running detection: %s", " ".join(cmd))
    try:
        # 不用 check=True，自己检查 returncode，避免 CalledProcessError 把细节吞掉
        result = subprocess.run(cmd)
        if result.returncode == 0:
            logger.info("[WATCHER] detection.py finished successfully.")
            return True
        else:
            logger.error(
                "[WATCHER][ERROR] detection.py exit code = %s",
                result.returncode,
            )
            return False
    except Exception as e:
        logger.exception("[WATCHER][ERROR] detection.py raised exception: %s", e)
        return False


def main():
    args = parse_args()

    logger.info("[WATCHER] Project root : %s", PROJECT_ROOT)
    logger.info("[WATCHER] Raw image dir: %s", RAWIMAGE_DIR)
    logger.info("[WATCHER] Detection.py : %s", DETECTION_SCRIPT)
    logger.info(
        "[WATCHER] frame_rate=%s, score_threshold=%s",
        args.frame_rate,
        args.score_threshold,
    )

    RAWIMAGE_DIR.mkdir(parents=True, exist_ok=True)

    last_processed_path: Path | None = None
    last_processed_mtime: float = 0.0

    while True:
        latest = find_latest_jpg(RAWIMAGE_DIR)

        if latest is None:
            logger.info("[WATCHER] No jpg found, sleep...")
            time.sleep(args.interval)
            continue

        stat = latest.stat()
        latest_mtime = stat.st_mtime
        latest_size = stat.st_size

        logger.info(
            "[WATCHER] Latest jpg: %s, size=%d, mtime=%s, last_processed=%s, last_mtime=%s",
            latest.name,
            latest_size,
            latest_mtime,
            getattr(last_processed_path, "name", None),
            last_processed_mtime,
        )

        # 防御性：如果是 0 字节文件，直接跳过等待下一轮
        if latest_size == 0:
            logger.warning(
                "[WATCHER] Latest jpg %s is 0 bytes, skip this file.",
                latest.name,
            )
            time.sleep(args.interval)
            continue

        # 只要「文件不同」或者「mtime 更大」，就认为是新图，跑一次
        if (
            last_processed_path is None
            or latest != last_processed_path
            or latest_mtime > last_processed_mtime
        ):
            success = run_detection_on_image(
                latest,
                args.frame_rate,
                args.score_threshold,
            )
            if success:
                last_processed_path = latest
                last_processed_mtime = latest_mtime
            else:
                logger.warning(
                    "[WATCHER] detection failed, will retry this image later."
                )
        else:
            logger.info("[WATCHER] No new image to process.")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
