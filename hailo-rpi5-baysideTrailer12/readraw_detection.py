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


PROJECT_ROOT = Path("/home/pi/hailo-rpi5-examples").resolve()
RAWIMAGE_DIR = PROJECT_ROOT / "rawimage"
DETECTION_SCRIPT = PROJECT_ROOT / "basic_pipelines" / "detection.py"


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


def run_detection_on_image(image_path: Path, frame_rate: int, score_threshold: float) -> bool:
    """调用 detection.py，对指定图片跑一次检测。成功返回 True，失败返回 False。"""
    cmd = [
        sys.executable,
        str(DETECTION_SCRIPT),
        "--input", str(image_path),
        "--frame-rate", str(frame_rate),
        "--score-threshold", str(score_threshold),
    ]
    print(f"[WATCHER] Running detection: {' '.join(cmd)}", flush=True)
    try:
        # 不用 check=True，自己检查 returncode，避免 CalledProcessError 把细节吞掉
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print("[WATCHER] detection.py finished successfully.", flush=True)
            return True
        else:
            print(f"[WATCHER][ERROR] detection.py exit code = {result.returncode}", flush=True)
            return False
    except Exception as e:
        print(f"[WATCHER][ERROR] detection.py raised exception: {e}", flush=True)
        return False



def main():
    args = parse_args()

    print(f"[WATCHER] Project root : {PROJECT_ROOT}", flush=True)
    print(f"[WATCHER] Raw image dir: {RAWIMAGE_DIR}", flush=True)
    print(f"[WATCHER] Detection.py : {DETECTION_SCRIPT}", flush=True)
    print(f"[WATCHER] frame_rate={args.frame_rate}, score_threshold={args.score_threshold}", flush=True)

    RAWIMAGE_DIR.mkdir(parents=True, exist_ok=True)

    last_processed_path: Path | None = None
    last_processed_mtime: float = 0.0

    while True:
        latest = find_latest_jpg(RAWIMAGE_DIR)

        if latest is None:
            print("[WATCHER] No jpg found, sleep...", flush=True)
            time.sleep(args.interval)
            continue

        latest_mtime = latest.stat().st_mtime
        print(
            f"[WATCHER] Latest jpg: {latest.name}, mtime={latest_mtime}, "
            f"last_processed={getattr(last_processed_path, 'name', None)}, "
            f"last_mtime={last_processed_mtime}",
            flush=True,
        )

        # 只要「文件不同」或者「mtime 更大」，就认为是新图，跑一次
        if (last_processed_path is None) or (latest != last_processed_path) or (latest_mtime > last_processed_mtime):
            success = run_detection_on_image(latest, args.frame_rate, args.score_threshold)
            if success:
                last_processed_path = latest
                last_processed_mtime = latest_mtime
            else:
                print("[WATCHER] detection failed, will retry this image later.", flush=True)
        else:
            print("[WATCHER] No new image to process.", flush=True)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
