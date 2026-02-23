#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
parking_overstay_nx.py

Reads existing outputs from your current pipeline:
  - rawimage/snapshot_YYYYMMDDHHMMSS.jpg
  - reslabel/snapshot_YYYYMMDDHHMMSS.txt

Per camera config (JSON):
  - confidence threshold
  - labels list
  - AOI polygon (normalized)
  - schedule windows (e.g. 21:00 -> 05:00)
  - overstay: max_park_minutes + grace_minutes

When overstay happens:
  - send NX Generic Event via /api/createEvent
    caption = parking.enter / parking.violation / (optional) parking.leave
  - optional: email violation snapshot

Designed for single-channel Pi+HAT now, but config supports multiple cameras.
"""

import os
import sys
import json
import time
import base64
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import smtplib
from email.message import EmailMessage


# -----------------------
# Utility: parsing time
# -----------------------

def parse_hhmm(s: str) -> dtime:
    hh, mm = s.strip().split(":")
    return dtime(int(hh), int(mm), 0)


def in_any_window(now_local: datetime, windows: list[dict]) -> dict | None:
    """
    Return the matching window dict if now_local is within any configured window; otherwise None.
    Windows can cross midnight (e.g. 21:00 -> 05:00).
    """
    t = now_local.timetz().replace(tzinfo=None)
    for w in windows:
        start = parse_hhmm(w["start"])
        end = parse_hhmm(w["end"])
        if start < end:
            if start <= t < end:
                return w
        else:
            # cross midnight
            if (t >= start) or (t < end):
                return w
    return None


def parse_ts_from_snapshot_stem(stem: str, tz: ZoneInfo) -> datetime | None:
    """
    stem like: snapshot_20260130094512
    """
    parts = stem.split("_")
    for part in reversed(parts):
        if part.isdigit() and len(part) == 14:
            try:
                dt = datetime.strptime(part, "%Y%m%d%H%M%S")
                return dt.replace(tzinfo=tz)
            except ValueError:
                return None
    return None


# -----------------------
# Geometry: point in polygon
# -----------------------

def polygon_norm_to_px(points_norm: list[list[float]], w: int, h: int) -> list[tuple[float, float]]:
    pts = []
    for x, y in points_norm:
        pts.append((x * w, y * h))
    return pts


def point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """
    Ray casting algorithm.
    poly: list of (x,y) vertices
    """
    inside = False
    n = len(poly)
    if n < 3:
        return False
    x1, y1 = poly[0]
    for i in range(1, n + 1):
        x2, y2 = poly[i % n]
        # check if point is between y1 and y2
        if ((y1 > y) != (y2 > y)):
            # compute x intersection
            x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_int:
                inside = not inside
        x1, y1 = x2, y2
    return inside


# -----------------------
# Parse reslabel format
# -----------------------

def parse_reslabel_file(fp: Path) -> list[dict]:
    """
    Expected line format (YOLO-like text):
      <label> <conf> <x_center> <y_center> <w> <h>
    Example:
      car 0.634674 0.450807 0.169684 0.195683 0.8284
      traffic light 0.079164 0.353132 0.016694 0.052754 0.4172

    Note: label can contain spaces. We'll parse from the end:
      last 5 tokens are numeric, the rest joined as label.
    """
    dets = []
    try:
        text = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return dets

    for line in text:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) < 6:
            continue
        tail = tokens[-5:]
        head = tokens[:-5]
        try:
            conf = float(tail[0])
            xc = float(tail[1])
            yc = float(tail[2])
            bw = float(tail[3])
            bh = float(tail[4])
        except ValueError:
            continue
        label = " ".join(head).strip().lower()
        dets.append({
            "label": label,
            "conf": conf,
            "xc": xc,
            "yc": yc,
            "w": bw,
            "h": bh
        })
    return dets


def det_bottom_center_px(det: dict, W: int, H: int) -> tuple[float, float]:
    cx = det["xc"] * W
    cy = det["yc"] * H
    bw = det["w"] * W
    bh = det["h"] * H
    px = cx
    py = cy + bh / 2.0
    return px, py


# -----------------------
# NX Generic Event publisher
# -----------------------

def nx_create_event(nx_cfg: dict, caption: str, description: str, logger: logging.Logger) -> bool:
    """
    Send NX Generic Event via:
      GET /api/createEvent?source=...&caption=...&description=...
    Basic auth with nx username/password.
    """
    server_url = nx_cfg["server_url"].rstrip("/")
    user = nx_cfg["username"]
    pwd = nx_cfg["password"]
    source = nx_cfg.get("source", "ParkingDemo")

    qs = urlencode({
        "source": source,
        "caption": caption,
        "description": description
    }, safe=":/,._- ")

    url = f"{server_url}/api/createEvent?{qs}"

    auth = base64.b64encode(f"{user}:{pwd}".encode("utf-8")).decode("ascii")
    req = Request(url, method="GET", headers={"Authorization": f"Basic {auth}"})

    try:
        with urlopen(req, timeout=5) as resp:
            status = getattr(resp, "status", 200)
            body = resp.read(200).decode("utf-8", errors="ignore")
            logger.info("[NX] createEvent ok status=%s caption=%s body=%s", status, caption, body)
            return True
    except HTTPError as e:
        logger.error("[NX] HTTPError status=%s reason=%s", e.code, e.reason)
        return False
    except URLError as e:
        logger.error("[NX] URLError: %s", e)
        return False
    except Exception as e:
        logger.exception("[NX] createEvent exception: %s", e)
        return False


# -----------------------
# Email notifier (optional)
# -----------------------

def send_violation_email(email_cfg: dict, camera_id: str, subject: str, body: str, jpg_path: Path | None, logger: logging.Logger) -> None:
    if not email_cfg.get("enabled", False):
        return

    msg = EmailMessage()
    msg["From"] = email_cfg["from"]
    msg["To"] = ", ".join(email_cfg["to"])
    msg["Subject"] = subject
    msg.set_content(body)

    if jpg_path is not None and jpg_path.exists():
        try:
            data = jpg_path.read_bytes()
            msg.add_attachment(data, maintype="image", subtype="jpeg", filename=jpg_path.name)
        except Exception as e:
            logger.warning("[EMAIL] attach failed: %s", e)

    host = email_cfg["smtp_host"]
    port = int(email_cfg.get("smtp_port", 587))
    user = email_cfg.get("username")
    pwd = email_cfg.get("password")

    try:
        with smtplib.SMTP(host, port, timeout=10) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            if user and pwd:
                s.login(user, pwd)
            s.send_message(msg)
        logger.info("[EMAIL] sent to=%s camera=%s", email_cfg.get("to"), camera_id)
    except Exception as e:
        logger.warning("[EMAIL] send failed: %s", e)


# -----------------------
# Parking FSM per camera
# -----------------------

class ParkingFSM:
    """
    States:
      IDLE -> OCCUPIED -> VIOLATED
      leaving resets to IDLE (optionally emits leave event)

    Uses:
      confirm_after_sec: entering must be stable for N sec before we consider "enter"
      miss_tolerance_sec: leaving only if no hit for N sec
    """
    IDLE = "IDLE"
    OCCUPIED = "OCCUPIED"
    VIOLATED = "VIOLATED"

    def __init__(self):
        self.state = self.IDLE
        self.first_seen_ts: datetime | None = None
        self.enter_ts: datetime | None = None
        self.last_hit_ts: datetime | None = None
        self.violation_ts: datetime | None = None

        self.enter_sent = False
        self.violation_sent = False
        self.leave_sent = False

    def reset(self):
        self.state = self.IDLE
        self.first_seen_ts = None
        self.enter_ts = None
        self.last_hit_ts = None
        self.violation_ts = None
        self.enter_sent = False
        self.violation_sent = False
        self.leave_sent = False

    def update(self, ts: datetime, has_hit: bool, confirm_after_sec: int, miss_tolerance_sec: int) -> dict | None:
        """
        Returns event dict:
          {"type":"enter"} / {"type":"violation"} / {"type":"leave"} or None
        """
        if has_hit:
            self.last_hit_ts = ts
            if self.state == self.IDLE:
                # start pending
                self.first_seen_ts = ts
                self.state = self.OCCUPIED  # treat as OCCUPIED, but enter_ts set only after confirm
                return None

            # state is OCCUPIED or VIOLATED
            if self.enter_ts is None and self.first_seen_ts is not None:
                if (ts - self.first_seen_ts).total_seconds() >= confirm_after_sec:
                    self.enter_ts = self.first_seen_ts
                    return {"type": "enter", "enter_ts": self.enter_ts}

            return None

        # no hit
        if self.state in (self.OCCUPIED, self.VIOLATED) and self.last_hit_ts is not None:
            gap = (ts - self.last_hit_ts).total_seconds()
            if gap > miss_tolerance_sec:
                # leaving
                leave_ts = ts
                # capture useful info before reset
                enter_ts = self.enter_ts
                violation_ts = self.violation_ts
                was_violated = (self.state == self.VIOLATED)
                self.reset()
                return {
                    "type": "leave",
                    "leave_ts": leave_ts,
                    "enter_ts": enter_ts,
                    "violation_ts": violation_ts,
                    "was_violated": was_violated
                }
        return None

    def maybe_mark_violation(self, ts: datetime) -> dict | None:
        if self.state == self.VIOLATED:
            return None
        self.state = self.VIOLATED
        self.violation_ts = ts
        return {"type": "violation", "violation_ts": ts, "enter_ts": self.enter_ts}


# -----------------------
# Main worker per camera
# -----------------------

def setup_logger(log_dir: Path, camera_id: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"parking_overstay[{camera_id}]")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    file_handler = RotatingFileHandler(
        log_dir / f"parking_overstay_{camera_id}.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    # avoid duplicate handlers if re-init
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def load_config(cfg_path: Path) -> dict:
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def list_reslabel_files(reslabel_dir: Path) -> list[Path]:
    return sorted(reslabel_dir.glob("snapshot_*.txt"))


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("--config", "-c"):
        print("Usage: python3 parking_overstay_nx.py --config /path/to/config_parking.json")
        sys.exit(2)

    cfg_path = Path(sys.argv[2]).expanduser().resolve()
    cfg = load_config(cfg_path)

    project_root = Path(cfg.get("project_root", Path(__file__).resolve().parent)).resolve()
    nx_cfg = cfg["nx"]
    email_cfg = cfg.get("email", {"enabled": False})

    cameras = cfg.get("cameras", [])
    if not cameras:
        print("No cameras configured in JSON.")
        sys.exit(2)

    # single process loops all cameras (OK for Pi single-channel; future can fork per camera)
    workers = []
    for cam in cameras:
        camera_id = cam["camera_id"]
        paths = cam.get("paths", {})
        rawimage_dir = (project_root / paths.get("rawimage_dir", "rawimage")).resolve()
        reslabel_dir = (project_root / paths.get("reslabel_dir", "reslabel")).resolve()
        logs_dir = (project_root / paths.get("logs_dir", "logs")).resolve()

        tz = ZoneInfo(cam.get("schedule", {}).get("timezone", "Australia/Sydney"))
        frame_w = int(cam.get("frame", {}).get("width", 640))
        frame_h = int(cam.get("frame", {}).get("height", 360))

        conf_th = float(cam.get("confidence_threshold", 0.5))
        labels = set([s.lower() for s in cam.get("labels", [])])

        aoi = cam.get("aoi", {})
        if aoi.get("type") != "polygon_norm":
            raise ValueError(f"camera {camera_id}: aoi.type must be polygon_norm")
        aoi_norm = aoi.get("points", [])
        if len(aoi_norm) < 3:
            raise ValueError(f"camera {camera_id}: aoi polygon must have >= 3 points")

        logic = cam.get("logic", {})
        confirm_after_sec = int(logic.get("confirm_after_sec", 5))
        miss_tolerance_sec = int(logic.get("miss_tolerance_sec", 20))
        poll_interval_sec = float(logic.get("poll_interval_sec", 2))
        emit_leave_event = bool(logic.get("emit_leave_event", False))

        schedule = cam.get("schedule", {})
        windows = schedule.get("windows", [])
        if not windows:
            raise ValueError(f"camera {camera_id}: schedule.windows cannot be empty")

        logger = setup_logger(logs_dir, camera_id)

        worker = {
            "cam": cam,
            "camera_id": camera_id,
            "rawimage_dir": rawimage_dir,
            "reslabel_dir": reslabel_dir,
            "logger": logger,
            "tz": tz,
            "frame_w": frame_w,
            "frame_h": frame_h,
            "conf_th": conf_th,
            "labels": labels,
            "aoi_px": polygon_norm_to_px(aoi_norm, frame_w, frame_h),
            "confirm_after_sec": confirm_after_sec,
            "miss_tolerance_sec": miss_tolerance_sec,
            "poll_interval_sec": poll_interval_sec,
            "emit_leave_event": emit_leave_event,
            "windows": windows,
            "fsm": ParkingFSM(),
            "processed": set()  # reslabel filename set
        }
        logger.info("Project root=%s", project_root)
        logger.info("rawimage_dir=%s reslabel_dir=%s logs_dir=%s", rawimage_dir, reslabel_dir, logs_dir)
        logger.info("frame=%dx%d conf_th=%.3f labels=%s", frame_w, frame_h, conf_th, ",".join(sorted(labels)))
        logger.info("AOI(px)=%s", worker["aoi_px"])
        logger.info("schedule windows=%s", windows)
        logger.info("logic confirm_after_sec=%s miss_tolerance_sec=%s poll_interval_sec=%s emit_leave_event=%s",
                    confirm_after_sec, miss_tolerance_sec, poll_interval_sec, emit_leave_event)
        workers.append(worker)

    while True:
        for w in workers:
            cam = w["cam"]
            camera_id = w["camera_id"]
            logger = w["logger"]
            tz = w["tz"]

            now_local = datetime.now(tz)
            win = in_any_window(now_local, w["windows"])
            if win is None:
                # outside monitoring windows
                # still update FSM with "no hit" to allow leave reset if needed (optional)
                time.sleep(0.01)
                continue

            # current overstay settings (per window)
            max_park_min = int(win.get("max_park_minutes", 0))
            grace_min = int(win.get("grace_minutes", 0))
            allowed_sec = (max_park_min + grace_min) * 60

            # scan reslabel
            reslabel_dir = w["reslabel_dir"]
            rawimage_dir = w["rawimage_dir"]
            reslabel_dir.mkdir(parents=True, exist_ok=True)
            rawimage_dir.mkdir(parents=True, exist_ok=True)

            files = list_reslabel_files(reslabel_dir)
            if not files:
                logger.info("No reslabel files yet. waiting...")
                continue

            for fp in files:
                if fp.name in w["processed"]:
                    continue

                stem = fp.stem  # snapshot_YYYYMMDDHHMMSS
                ts = parse_ts_from_snapshot_stem(stem, tz)
                if ts is None:
                    w["processed"].add(fp.name)
                    continue

                dets = parse_reslabel_file(fp)

                # filter detections: label + confidence + AOI
                hit = False
                best = None  # best matching det for logging
                for d in dets:
                    if d["label"] not in w["labels"]:
                        continue
                    if d["conf"] < w["conf_th"]:
                        continue
                    px, py = det_bottom_center_px(d, w["frame_w"], w["frame_h"])
                    if point_in_polygon(px, py, w["aoi_px"]):
                        hit = True
                        if best is None or d["conf"] > best["conf"]:
                            best = {**d, "px": px, "py": py}
                w["processed"].add(fp.name)

                # update FSM with hit/no-hit
                evt = w["fsm"].update(
                    ts=ts,
                    has_hit=hit,
                    confirm_after_sec=w["confirm_after_sec"],
                    miss_tolerance_sec=w["miss_tolerance_sec"]
                )

                logger.info("file=%s ts=%s hit=%s best=%s",
                            fp.name, ts.isoformat(), hit,
                            None if best is None else f"{best['label']} conf={best['conf']:.3f} p=({best['px']:.1f},{best['py']:.1f})")

                # if enter event
                if evt and evt["type"] == "enter":
                    enter_ts = evt["enter_ts"]
                    desc = json.dumps({
                        "camera_id": camera_id,
                        "event": "enter",
                        "enter_ts": enter_ts.isoformat(),
                        "rtsp": cam.get("rtsp_url", ""),
                        "window": {"start": win["start"], "end": win["end"]},
                        "allowed_sec": allowed_sec
                    }, ensure_ascii=False)
                    nx_create_event(nx_cfg, "parking.enter", desc, logger)

                # violation check (only if we have a confirmed enter_ts)
                if w["fsm"].enter_ts is not None and w["fsm"].state != ParkingFSM.VIOLATED:
                    duration = (ts - w["fsm"].enter_ts).total_seconds()
                    if duration >= allowed_sec:
                        vio = w["fsm"].maybe_mark_violation(ts)
                        if vio:
                            enter_ts = vio.get("enter_ts")
                            vio_ts = vio["violation_ts"]

                            # try to attach snapshot jpg for email
                            jpg_path = rawimage_dir / f"{stem}.jpg"
                            if not jpg_path.exists():
                                jpg_path = None

                            payload = {
                                "camera_id": camera_id,
                                "event": "violation",
                                "enter_ts": enter_ts.isoformat() if enter_ts else None,
                                "violation_ts": vio_ts.isoformat(),
                                "duration_sec": int(duration),
                                "max_park_minutes": max_park_min,
                                "grace_minutes": grace_min,
                                "confidence_threshold": w["conf_th"],
                                "best_det": None if best is None else {
                                    "label": best["label"],
                                    "conf": best["conf"],
                                    "xc": best["xc"], "yc": best["yc"], "w": best["w"], "h": best["h"],
                                    "point_px": [best["px"], best["py"]]
                                },
                                "snapshot_jpg": None if jpg_path is None else jpg_path.name
                            }
                            desc = json.dumps(payload, ensure_ascii=False)
                            nx_create_event(nx_cfg, "parking.violation", desc, logger)

                            # optional email
                            subject = f"{email_cfg.get('subject_prefix','[ParkingDemo]')} violation {camera_id}"
                            body = (
                                f"Parking violation detected.\n\n"
                                f"camera_id: {camera_id}\n"
                                f"enter_ts: {payload['enter_ts']}\n"
                                f"violation_ts: {payload['violation_ts']}\n"
                                f"duration_sec: {payload['duration_sec']}\n"
                                f"threshold: max={max_park_min}min grace={grace_min}min\n"
                                f"snapshot: {payload['snapshot_jpg']}\n"
                            )
                            send_violation_email(email_cfg, camera_id, subject, body, jpg_path, logger)

                # optional leave event
                if evt and evt["type"] == "leave" and w["emit_leave_event"]:
                    desc = json.dumps({
                        "camera_id": camera_id,
                        "event": "leave",
                        "leave_ts": evt["leave_ts"].isoformat(),
                        "enter_ts": None if evt["enter_ts"] is None else evt["enter_ts"].isoformat(),
                        "violation_ts": None if evt["violation_ts"] is None else evt["violation_ts"].isoformat(),
                        "was_violated": evt["was_violated"]
                    }, ensure_ascii=False)
                    nx_create_event(nx_cfg, "parking.leave", desc, logger)

        time.sleep(min(w["poll_interval_sec"] for w in workers))


if __name__ == "__main__":
    main()
