#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
checkreslabel_findnx.py

功能：
1) 日志：RotatingFileHandler，logs/checkreslabel_findnx_log.log，5MB，最多 3 个备份
2) 在每天 10:00-22:00（Australia/Sydney）保持 active，其余时间 sleep
3) 扫描 reslabel 中昨晚 22:00 到今日 06:00 的文件（按文件名时间戳顺序重放）
4) “场景级违停”判定（不区分车辆ID）：
   - 只要画面检测到车辆类 label 连续存在 >= 60 秒，则打印一次：
     [WARNING] illegally parked vehicles: confirm_ts=... start_ts=... file=...
   - 防抖：
     - 允许短暂 miss（检测不到车）不立刻结束事件：MISS_TOLERANCE_SEC
     - 若没有新文件产生，不进行“消失判定”（只做 stall 日志）
5) 记录：每次 confirmed 追加写入 logs/illegally_parked_events.jsonl（便于后续调取）
"""

import time
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo


# =========================
# 路径配置（与 readraw_detection.py 同级）
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
RESLABEL_DIR = PROJECT_ROOT / "reslabel"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "checkreslabel_findnx_log.log"
EVENTS_FILE = LOG_DIR / "illegally_parked_events.jsonl"

# =========================
# 时区与时间窗口
# =========================
TZ = ZoneInfo("Australia/Sydney")

ACTIVE_START = dtime(10, 0, 0)  # 10:00
ACTIVE_END   = dtime(22, 0, 0)  # 22:00（不含 22:00 之后）

# 需要检查的“夜间”区间：昨晚 22:00 -> 今日 06:00
NIGHT_START = dtime(22, 0, 0)
NIGHT_END   = dtime(6, 0, 0)

# =========================
# 车辆 label（可按需扩展）
# =========================
VEHICLE_LABELS = {
    "car", "truck", "bus",
    "van", "motorcycle", "motorbike", "bike", "bicycle",
    "pickup", "ute", "suv", "taxi",
    "trailer",
}

# =========================
# 扫描间隔（active 时）
# =========================
POLL_INTERVAL_SEC = 2

# =========================
# 违停判定参数
# =========================
PARK_THRESHOLD_SEC = 60          # 连续存在 >= 60s 触发一次违停
MISS_TOLERANCE_SEC = 20          # 允许短暂 miss（检测不到车辆）的最大时长
STALL_WARN_SEC = 60              # 最久没有新文件则记录 stall 警告（不影响状态机）


# =========================
# 日志配置：5MB * 3 个轮转
# =========================
logger = logging.getLogger("checkreslabel_findnx")
logger.setLevel(logging.INFO)

fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

file_handler = RotatingFileHandler(
    LOG_FILE,
    maxBytes=5 * 1024 * 1024,  # 5MB
    backupCount=3,
    encoding="utf-8",
)
file_handler.setFormatter(fmt)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(fmt)

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

logger.propagate = False


# =========================
# 工具函数
# =========================
def now_syd() -> datetime:
    return datetime.now(TZ)


def is_active_window(dt: datetime) -> bool:
    t = dt.timetz().replace(tzinfo=None)
    return ACTIVE_START <= t < ACTIVE_END


def seconds_until_next_active(dt: datetime) -> int:
    today_10 = dt.replace(hour=10, minute=0, second=0, microsecond=0)
    if dt < today_10:
        target = today_10
    else:
        target = today_10 + timedelta(days=1)
    sec = int((target - dt).total_seconds())
    return max(sec, 1)


def night_window_range(dt: datetime) -> tuple[datetime, datetime]:
    today = dt.date()
    start = datetime.combine(today - timedelta(days=1), NIGHT_START, tzinfo=TZ)
    end   = datetime.combine(today, NIGHT_END, tzinfo=TZ)
    return start, end


def parse_ts_from_reslabel_name(p: Path) -> datetime | None:
    """
    文件名类似：snapshot_20260106033135.txt
    解析出 2026-01-06 03:31:35（悉尼时间）
    """
    stem = p.stem  # snapshot_20260106033135
    parts = stem.split("_")
    for part in reversed(parts):
        if part.isdigit() and len(part) == 14:
            try:
                ts = datetime.strptime(part, "%Y%m%d%H%M%S")
                return ts.replace(tzinfo=TZ)
            except ValueError:
                return None
    return None


def file_contains_vehicle_label(file_path: Path) -> tuple[bool, set[str]]:
    hit_labels: set[str] = set()
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                first = line.split()[0].strip().lower()
                if first in VEHICLE_LABELS:
                    hit_labels.add(first)
    except Exception as e:
        logger.warning("Failed to read %s: %s", file_path.name, e)
        return False, set()

    return (len(hit_labels) > 0), hit_labels


def append_event_record(record: dict) -> None:
    """
    追加写入 JSONL，便于后续调取。
    """
    try:
        with open(EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("Failed to append event record: %s", e)


# =========================
# 状态机（场景级违停）
# =========================
class ParkingState:
    IDLE = "IDLE"
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"


class ParkingEventFSM:
    """
    仅基于 (ts, has_vehicle) 序列判定“车辆持续存在 >= PARK_THRESHOLD_SEC”
    - 防抖：允许 miss 但 miss 时长 <= MISS_TOLERANCE_SEC 不结束事件
    - 仅在 PENDING -> CONFIRMED 时刻报警/记录一次
    """

    def __init__(self):
        self.state = ParkingState.IDLE
        self.event_start_ts: datetime | None = None
        self.last_seen_vehicle_ts: datetime | None = None
        self.confirmed_ts: datetime | None = None

        # 用于记录“事件确认时”对应的文件信息（便于追溯）
        self.start_file: str | None = None
        self.confirm_file: str | None = None

        # 事件标签聚合（可选）
        self.labels_seen: set[str] = set()

    def reset(self):
        self.state = ParkingState.IDLE
        self.event_start_ts = None
        self.last_seen_vehicle_ts = None
        self.confirmed_ts = None
        self.start_file = None
        self.confirm_file = None
        self.labels_seen = set()

    def update(self, ts: datetime, has_vehicle: bool, labels: set[str], filename: str):
        """
        输入一条“新观测”（来自一个新 reslabel 文件）
        可能触发：
        - 确认违停（返回一条 record dict）
        - 或者无输出（返回 None）
        """
        if has_vehicle:
            self.labels_seen |= set(labels)

            if self.state == ParkingState.IDLE:
                self.state = ParkingState.PENDING
                self.event_start_ts = ts
                self.last_seen_vehicle_ts = ts
                self.start_file = filename
                logger.info("FSM: IDLE -> PENDING (start_ts=%s file=%s)", ts.isoformat(), filename)
                return None

            # PENDING / CONFIRMED：更新 last_seen
            if self.last_seen_vehicle_ts is None or ts > self.last_seen_vehicle_ts:
                self.last_seen_vehicle_ts = ts

            # 若是 PENDING，检查是否达到阈值
            if self.state == ParkingState.PENDING:
                if self.event_start_ts is not None and self.last_seen_vehicle_ts is not None:
                    duration = (self.last_seen_vehicle_ts - self.event_start_ts).total_seconds()
                    if duration >= PARK_THRESHOLD_SEC:
                        self.state = ParkingState.CONFIRMED
                        self.confirmed_ts = ts
                        self.confirm_file = filename

                        record = {
                            "type": "illegally_parked_vehicle",
                            "start_ts": self.event_start_ts.isoformat(),
                            "confirm_ts": self.confirmed_ts.isoformat(),
                            "start_file": self.start_file,
                            "confirm_file": self.confirm_file,
                            "labels": sorted(self.labels_seen),
                            "duration_sec_at_confirm": int(duration),
                        }
                        return record

            # CONFIRMED：不重复记录
            return None

        # has_vehicle == False：仅在有新观测时处理“可能结束”
        if self.state in (ParkingState.PENDING, ParkingState.CONFIRMED):
            if self.last_seen_vehicle_ts is not None:
                gap = (ts - self.last_seen_vehicle_ts).total_seconds()
                if gap > MISS_TOLERANCE_SEC:
                    # 认为事件结束
                    logger.info(
                        "FSM: %s -> IDLE (end_ts=%s gap=%.1fs last_seen=%s)",
                        self.state,
                        ts.isoformat(),
                        gap,
                        self.last_seen_vehicle_ts.isoformat(),
                    )
                    self.reset()
            return None

        # IDLE 且无车：无事发生
        return None


# =========================
# 扫描与处理
# =========================
def scan_reslabel_for_night(dt: datetime, processed: set[str], fsm: ParkingEventFSM, last_latest_ts_holder: dict) -> None:
    """
    扫描 reslabel 中属于昨晚 22:00 -> 今日 06:00 的文件；
    对未处理过的文件，按时间顺序喂给状态机。
    """
    start, end = night_window_range(dt)
    RESLABEL_DIR.mkdir(parents=True, exist_ok=True)

    # 仅记录一次窗口信息，避免每2秒刷屏
    # 但仍然在日志里可见扫描行为
    logger.debug("Night window: %s -> %s", start.isoformat(), end.isoformat())

    files = list(RESLABEL_DIR.glob("*.txt"))
    if not files:
        logger.info("No reslabel txt files found.")
        return

    # 过滤：能解析出 ts 且落在窗口内的
    candidates: list[tuple[datetime, Path]] = []
    latest_ts_in_dir: datetime | None = None

    for fp in files:
        ts = parse_ts_from_reslabel_name(fp)
        if ts is None:
            continue
        if latest_ts_in_dir is None or ts > latest_ts_in_dir:
            latest_ts_in_dir = ts
        if start <= ts <= end:
            candidates.append((ts, fp))

    # stall 检测（不影响 FSM）
    if latest_ts_in_dir is not None:
        last_latest = last_latest_ts_holder.get("latest_ts")
        last_latest_ts_holder["latest_ts"] = latest_ts_in_dir

        # 如果 latest_ts 没变，且 now - latest_ts 很久，则提示可能卡住
        now_dt = dt
        stall_age = (now_dt - latest_ts_in_dir).total_seconds()
        if stall_age >= STALL_WARN_SEC:
            logger.warning(
                "reslabel may be stalled: now=%s latest_file_ts=%s age=%.1fs",
                now_dt.isoformat(),
                latest_ts_in_dir.isoformat(),
                stall_age,
            )

        # 若 latest_ts 回退（理论不应发生），也记录一下
        if last_latest is not None and latest_ts_in_dir < last_latest:
            logger.warning("Latest reslabel ts moved backwards: %s -> %s", last_latest.isoformat(), latest_ts_in_dir.isoformat())

    if not candidates:
        logger.info("No reslabel files within night window.")
        return

    # 时间顺序处理；同一秒多个文件也能稳定处理
    candidates.sort(key=lambda x: (x[0], x[1].name))

    for ts, fp in candidates:
        if fp.name in processed:
            continue

        hit, labels = file_contains_vehicle_label(fp)
        processed.add(fp.name)

        # 喂给状态机
        record = fsm.update(ts=ts, has_vehicle=hit, labels=labels, filename=fp.name)

        if record is not None:
            # 仅在确认时刻输出一次
            msg = (
                f"[WARNING] illegally parked vehicles: "
                f"confirm_ts={record['confirm_ts']} start_ts={record['start_ts']} file={record['confirm_file']}"
            )
            print(msg)
            logger.warning(
                "Illegally parked vehicles CONFIRMED. start_ts=%s confirm_ts=%s start_file=%s confirm_file=%s labels=%s duration=%ss",
                record["start_ts"],
                record["confirm_ts"],
                record["start_file"],
                record["confirm_file"],
                ",".join(record["labels"]),
                record["duration_sec_at_confirm"],
            )
            append_event_record(record)
        else:
            # 你如果不想刷太多 info，可把这行改为 logger.debug
            logger.info("Processed file=%s ts=%s has_vehicle=%s", fp.name, ts.isoformat(), hit)


# =========================
# 主循环
# =========================
def main():
    logger.info("Project root : %s", PROJECT_ROOT)
    logger.info("Reslabel dir  : %s", RESLABEL_DIR)
    logger.info("Log file      : %s", LOG_FILE)
    logger.info("Events file   : %s", EVENTS_FILE)
    logger.info("Vehicle labels: %s", ",".join(sorted(VEHICLE_LABELS)))
    logger.info("Active window : %s-%s (Sydney)", ACTIVE_START, ACTIVE_END)
    logger.info("Night window  : lastday %s -> today %s (Sydney)", NIGHT_START, NIGHT_END)
    logger.info("Params: PARK_THRESHOLD_SEC=%s MISS_TOLERANCE_SEC=%s POLL_INTERVAL_SEC=%s STALL_WARN_SEC=%s",
                PARK_THRESHOLD_SEC, MISS_TOLERANCE_SEC, POLL_INTERVAL_SEC, STALL_WARN_SEC)

    processed: set[str] = set()
    fsm = ParkingEventFSM()
    last_latest_ts_holder: dict = {"latest_ts": None}

    # 避免一开始就刷很多窗口日志：这里先打印一次
    dt0 = now_syd()
    w_start, w_end = night_window_range(dt0)
    logger.info("Night window (current run): %s -> %s", w_start.isoformat(), w_end.isoformat())

    while True:
        dt = now_syd()

        if not is_active_window(dt):
            sec = seconds_until_next_active(dt)
            logger.info("Outside active window. Sleep %d sec until next 10:00.", sec)
            time.sleep(sec)
            continue

        try:
            scan_reslabel_for_night(dt, processed, fsm, last_latest_ts_holder)
        except Exception as e:
            logger.exception("Scan failed: %s", e)

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
