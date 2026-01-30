#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
checkreslabel_findnx.py

功能：
1) 日志：RotatingFileHandler，logs/checkreslabel_findnx_log.log，5MB，最多 3 个备份
2) 在每天 22:00-06:00（Australia/Sydney）保持 active，其余时间 sleep
3) 夜间实时监控“当前这晚”的窗口（22:00 -> 次日 06:00），按文件名时间戳顺序处理
4) “场景级违停”判定（不区分车辆ID）：
   - 只要画面检测到车辆类 label 连续存在 >= 60 秒，则打印一次 CONFIRMED（不写 events.jsonl）
   - 当车辆离开（或连续 miss 超过容忍阈值）事件结束时，写入一条完整时间段记录到 events.jsonl
   - 防抖：
     - 允许短暂 miss（检测不到车）不立刻结束事件：MISS_TOLERANCE_SEC
     - 若没有新文件产生，不进行“消失判定”（只做 stall 日志）
5) 记录：logs/illegally_parked_events.jsonl 每行都是完整事件段（start/confirm/end）
"""

import time
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
import socket
import paho.mqtt.client as mqtt


# =========================
# MQTT 配置（用于 CONFIRMED 推送）
# =========================
MQTT_HOST = "13.238.189.183"
MQTT_PORT = 1883
MQTT_USERNAME = "test"
MQTT_PASSWORD = "2143test"

MQTT_TOPIC = "/AI_monitor/illegally_parked_vehicle/BaysideTrailer12"
MQTT_CLIENT_ID = f"checkreslabel_findnx_{socket.gethostname()}"
MQTT_QOS = 1
MQTT_KEEPALIVE = 30

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

# Active window：夜间实时监控 22:00 -> 06:00（跨午夜）
ACTIVE_START = dtime(21, 0, 0)
ACTIVE_END   = dtime(7, 0, 0)

# 夜间检查窗口：同样是 22:00 -> 06:00（实时“当前这晚”）
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
POLL_INTERVAL_SEC = 10

# =========================
# 违停判定参数
# =========================
PARK_THRESHOLD_SEC = 60          # 连续存在 >= 60s 触发一次违停确认
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
def mqtt_publish_confirm(payload: dict) -> None:
    """
    在 CONFIRMED 时发布 MQTT 消息。
    设计原则：失败不影响主流程，只打日志。
    """
    try:
        client = mqtt.Client(client_id=MQTT_CLIENT_ID, clean_session=True)
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

        # 连接（阻塞式，快速失败）
        client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE)

        # 发布
        msg = json.dumps(payload, ensure_ascii=False)
        info = client.publish(MQTT_TOPIC, msg, qos=MQTT_QOS, retain=False)

        # 等待发送结果（避免立刻退出导致消息没发出去）
        info.wait_for_publish(timeout=3)

        client.disconnect()
        logger.info("MQTT published CONFIRMED to topic=%s payload=%s", MQTT_TOPIC, msg)
    except Exception as e:
        logger.warning("MQTT publish failed: %s", e)


def now_syd() -> datetime:
    return datetime.now(TZ)


def is_active_window(dt: datetime) -> bool:
    # t = dt.timetz().replace(tzinfo=None)
    #
    # # 非跨午夜
    # if ACTIVE_START < ACTIVE_END:
    #     return ACTIVE_START <= t < ACTIVE_END
    #
    # # 跨午夜：22:00 -> 06:00
    # return (t >= ACTIVE_START) or (t < ACTIVE_END)
    return True


def seconds_until_next_active(dt: datetime) -> int:
    # 睡到下一次 ACTIVE_START（22:00）
    target = dt.replace(hour=ACTIVE_START.hour, minute=ACTIVE_START.minute, second=0, microsecond=0)
    if dt >= target:
        target = target + timedelta(days=1)
    sec = int((target - dt).total_seconds())
    return max(sec, 1)


def night_window_range(dt: datetime) -> tuple[datetime, datetime]:
    """
    夜间实时窗口（当前这晚）：
    - 若当前时间在 22:00-24:00：窗口为 今天 22:00 -> 明天 06:00
    - 若当前时间在 00:00-06:00：窗口为 昨天 22:00 -> 今天 06:00
    - 其它时间：返回“下一次夜间窗口”（今天 22:00 -> 明天 06:00），用于日志/调试不出错
    """
    t = dt.timetz().replace(tzinfo=None)
    today = dt.date()

    if t >= NIGHT_START:
        start = datetime.combine(today, NIGHT_START, tzinfo=TZ)
        end = datetime.combine(today + timedelta(days=1), NIGHT_END, tzinfo=TZ)
        return start, end

    if t < NIGHT_END:
        start = datetime.combine(today - timedelta(days=1), NIGHT_START, tzinfo=TZ)
        end = datetime.combine(today, NIGHT_END, tzinfo=TZ)
        return start, end

    # 白天：给一个“下一次夜间窗口”，不影响，因为白天本来也不扫描
    start = datetime.combine(today, NIGHT_START, tzinfo=TZ)
    end = datetime.combine(today + timedelta(days=1), NIGHT_END, tzinfo=TZ)
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
    追加写入 JSONL（每行一个完整事件段）
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
    def __init__(self):
        self.state = ParkingState.IDLE
        self.event_start_ts: datetime | None = None
        self.last_seen_vehicle_ts: datetime | None = None
        self.confirmed_ts: datetime | None = None

        self.start_file: str | None = None
        self.confirm_file: str | None = None

        self.labels_seen: set[str] = set()

    def reset(self):
        self.state = ParkingState.IDLE
        self.event_start_ts = None
        self.last_seen_vehicle_ts = None
        self.confirmed_ts = None
        self.start_file = None
        self.confirm_file = None
        self.labels_seen = set()

    def update(self, ts: datetime, has_vehicle: bool, labels: set[str], filename: str) -> dict | None:
        if has_vehicle:
            self.labels_seen |= set(labels)

            if self.state == ParkingState.IDLE:
                self.state = ParkingState.PENDING
                self.event_start_ts = ts
                self.last_seen_vehicle_ts = ts
                self.start_file = filename
                logger.info("FSM: IDLE -> PENDING (start_ts=%s file=%s)", ts.isoformat(), filename)
                return None

            if self.last_seen_vehicle_ts is None or ts > self.last_seen_vehicle_ts:
                self.last_seen_vehicle_ts = ts

            if self.state == ParkingState.PENDING:
                if self.event_start_ts is not None and self.last_seen_vehicle_ts is not None:
                    duration = (self.last_seen_vehicle_ts - self.event_start_ts).total_seconds()
                    if duration >= PARK_THRESHOLD_SEC:
                        self.state = ParkingState.CONFIRMED
                        self.confirmed_ts = ts
                        self.confirm_file = filename
                        return {
                            "type": "confirmed",
                            "start_ts": self.event_start_ts,
                            "confirm_ts": self.confirmed_ts,
                            "start_file": self.start_file,
                            "confirm_file": self.confirm_file,
                            "labels": sorted(self.labels_seen),
                            "duration_sec_at_confirm": int(duration),
                        }

            return None

        # has_vehicle == False
        if self.state in (ParkingState.PENDING, ParkingState.CONFIRMED):
            if self.last_seen_vehicle_ts is not None:
                gap = (ts - self.last_seen_vehicle_ts).total_seconds()
                if gap > MISS_TOLERANCE_SEC:
                    if self.state == ParkingState.CONFIRMED and self.event_start_ts and self.confirmed_ts:
                        end_ts = ts
                        record = {
                            "type": "illegally_parked_vehicle",
                            "start_ts": self.event_start_ts.isoformat(),
                            "confirm_ts": self.confirmed_ts.isoformat(),
                            "end_ts": end_ts.isoformat(),
                            "start_file": self.start_file,
                            "confirm_file": self.confirm_file,
                            "end_file": filename,
                            "labels": sorted(self.labels_seen),
                            "duration_sec_total": int((end_ts - self.event_start_ts).total_seconds()),
                        }
                        logger.info(
                            "FSM: CONFIRMED -> IDLE (end_ts=%s gap=%.1fs last_seen=%s)",
                            end_ts.isoformat(),
                            gap,
                            self.last_seen_vehicle_ts.isoformat(),
                        )
                        self.reset()
                        return {"type": "ended", "record": record}

                    logger.info(
                        "FSM: %s -> IDLE (end_ts=%s gap=%.1fs last_seen=%s)",
                        self.state,
                        ts.isoformat(),
                        gap,
                        self.last_seen_vehicle_ts.isoformat(),
                    )
                    self.reset()
            return None

        return None


# =========================
# 扫描与处理
# =========================
def scan_reslabel_for_night(dt: datetime, processed: set[str], fsm: ParkingEventFSM, last_latest_ts_holder: dict) -> None:
    start, end = night_window_range(dt)
    RESLABEL_DIR.mkdir(parents=True, exist_ok=True)

    files = list(RESLABEL_DIR.glob("*.txt"))
    if not files:
        logger.info("No reslabel txt files found.")
        return

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
        last_latest_ts_holder["latest_ts"] = latest_ts_in_dir
        stall_age = (dt - latest_ts_in_dir).total_seconds()
        if stall_age >= STALL_WARN_SEC:
            logger.warning(
                "reslabel may be stalled: now=%s latest_file_ts=%s age=%.1fs",
                dt.isoformat(),
                latest_ts_in_dir.isoformat(),
                stall_age,
            )

    if not candidates:
        # 夜间实时：如果窗口内暂时没文件，这是正常情况（上游还没生成）
        logger.info("No reslabel files within current night window: %s -> %s", start.isoformat(), end.isoformat())
        return

    candidates.sort(key=lambda x: (x[0], x[1].name))

    for ts, fp in candidates:
        if fp.name in processed:
            continue

        hit, labels = file_contains_vehicle_label(fp)
        processed.add(fp.name)

        out = fsm.update(ts=ts, has_vehicle=hit, labels=labels, filename=fp.name)

        logger.info("Processed file=%s ts=%s has_vehicle=%s", fp.name, ts.isoformat(), hit)

        if out is None:
            continue

        if out.get("type") == "confirmed":
            msg = (
                f"[WARNING] illegally parked vehicles (CONFIRMED): "
                f"confirm_ts={out['confirm_ts'].isoformat()} start_ts={out['start_ts'].isoformat()} "
                f"file={out['confirm_file']}"
            )
            print(msg)
            logger.warning(
                "Illegally parked vehicles CONFIRMED. start_ts=%s confirm_ts=%s start_file=%s confirm_file=%s labels=%s duration=%ss",
                out["start_ts"].isoformat(),
                out["confirm_ts"].isoformat(),
                out["start_file"],
                out["confirm_file"],
                ",".join(out["labels"]),
                out["duration_sec_at_confirm"],
            )
            # 发送 MQTT（CONFIRMED 一次）
            mqtt_payload = {
                "type": "illegally_parked_vehicle_confirmed",
                "start_ts": out["start_ts"].isoformat(),
                "confirm_ts": out["confirm_ts"].isoformat(),
                "start_file": out["start_file"],
                "confirm_file": out["confirm_file"],
                "labels": out["labels"],
                "duration_sec_at_confirm": out["duration_sec_at_confirm"],
            }
            mqtt_publish_confirm(mqtt_payload)
            continue

        if out.get("type") == "ended":
            record = out["record"]
            append_event_record(record)

            print(
                f"[INFO] illegally parked vehicles (ENDED): "
                f"start_ts={record['start_ts']} end_ts={record['end_ts']} duration={record['duration_sec_total']}s"
            )

            logger.warning(
                "Illegally parked vehicles ENDED. start_ts=%s confirm_ts=%s end_ts=%s duration_total=%ss labels=%s",
                record["start_ts"],
                record["confirm_ts"],
                record["end_ts"],
                record["duration_sec_total"],
                ",".join(record["labels"]),
            )
            continue


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
    logger.info("Night window  : %s -> %s (Sydney)", NIGHT_START, NIGHT_END)
    logger.info("Params: PARK_THRESHOLD_SEC=%s MISS_TOLERANCE_SEC=%s POLL_INTERVAL_SEC=%s STALL_WARN_SEC=%s",
                PARK_THRESHOLD_SEC, MISS_TOLERANCE_SEC, POLL_INTERVAL_SEC, STALL_WARN_SEC)

    processed: set[str] = set()
    fsm = ParkingEventFSM()
    last_latest_ts_holder: dict = {"latest_ts": None}

    dt0 = now_syd()
    w_start, w_end = night_window_range(dt0)
    logger.info("Night window (current run): %s -> %s", w_start.isoformat(), w_end.isoformat())

    while True:
        dt = now_syd()

        if not is_active_window(dt):
            sec = seconds_until_next_active(dt)
            logger.info("Outside active window. Sleep %d sec until next 22:00.", sec)
            time.sleep(sec)
            continue

        try:
            scan_reslabel_for_night(dt, processed, fsm, last_latest_ts_holder)
        except Exception as e:
            logger.exception("Scan failed: %s", e)

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
