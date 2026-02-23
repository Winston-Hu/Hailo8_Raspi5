#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export snapshots for Nx bookmarks in a time window (Route B).
- Auto login (Bearer token)
- Auto discover bookmark list endpoint (REST v2/v1 + EC2 fallbacks)
- Filter bookmarks by time window + keywords/tags
- For each bookmark, request a snapshot at bookmark central time and save to local disk
"""

import os
import re
import json
import time
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =========================
# USER CONFIG (edit here)
# =========================
NX_HOST = "192.168.72.176"
NX_PORT = 7001
USE_HTTPS = True
VERIFY_TLS = False  # self-signed -> False; production should use proper CA

NX_USERNAME = "admin"
NX_PASSWORD = "3.1415926Pi2143"

# Output folder
OUT_DIR = "./bookmark_snapshots"

# ---- Time window mode ----
# MODE = "last_hours"  -> use LOOKBACK_HOURS
# MODE = "day"         -> use DAY_YYYY_MM_DD
TIME_WINDOW_MODE = "last_hours"
LOOKBACK_HOURS = 6
DAY_YYYY_MM_DD = "2026-02-02"

# Your local timezone offset (Sydney usually +10/+11; you can hardcode +11 like below)
# If you leave it None, script uses your OS local time.
LOCAL_TZ_UTC_OFFSET_HOURS = 11

# ---- Bookmark filter ----
# Recommend: put tags in the Nx rule. If you do, set FILTER_TAG.
FILTER_TAG = None  # e.g. "illegal_parking"

# Fallback keyword matching (works even if you didn't set tags)
KEYWORDS_ANY = ["illegal_parking_confirmed", "BaysideTrailer12_AI"]  # any of these matches
KEYWORDS_ALL = []  # all of these must match

# Max bookmarks to process (safety)
MAX_ITEMS = 50

# Snapshot size hint (some endpoints accept it; safe to ignore if not supported)
SNAPSHOT_WIDTH = 1280

# =========================

def base_url() -> str:
    scheme = "https" if USE_HTTPS else "http"
    return f"{scheme}://{NX_HOST}:{NX_PORT}"

def login_get_token() -> str:
    """
    POST /web/rest/v1/login/sessions -> { token: ... }
    (You already confirmed this works.)
    """
    url = f"{base_url()}/web/rest/v1/login/sessions"
    payload = {"username": NX_USERNAME, "password": NX_PASSWORD, "setCookie": False}
    r = requests.post(url, json=payload, timeout=10, verify=VERIFY_TLS)
    if r.status_code != 200:
        raise RuntimeError(f"Login failed: {r.status_code} {r.text[:500]}")
    data = r.json()
    token = data.get("token") or data.get("accessToken") or data.get("id")
    if not token:
        raise RuntimeError(f"Login ok but token not found in response: {data}")
    return token

def make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.verify = VERIFY_TLS
    s.headers.update({"Authorization": f"Bearer {token}"})
    return s

def get_local_tz() -> timezone:
    if LOCAL_TZ_UTC_OFFSET_HOURS is None:
        # Use OS local timezone (best effort)
        return datetime.now().astimezone().tzinfo or timezone.utc
    return timezone(timedelta(hours=LOCAL_TZ_UTC_OFFSET_HOURS))

def to_epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def parse_day_window(day_str: str, tz: timezone) -> Tuple[int, int]:
    start = datetime.strptime(day_str, "%Y-%m-%d").replace(tzinfo=tz)
    end = start + timedelta(days=1)
    return to_epoch_ms(start), to_epoch_ms(end)

def last_hours_window(hours: int, tz: timezone) -> Tuple[int, int]:
    end = datetime.now(tz=tz)
    start = end - timedelta(hours=hours)
    return to_epoch_ms(start), to_epoch_ms(end)

def safe_json(resp: requests.Response) -> Optional[Any]:
    try:
        return resp.json()
    except Exception:
        return None

def try_get_json(s: requests.Session, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    r = s.get(url, params=params, timeout=15)
    if r.status_code != 200:
        return None
    return safe_json(r)

def discover_bookmark_list(s: requests.Session) -> Tuple[str, str]:
    """
    Try multiple endpoints to list bookmarks.
    Returns (kind, url_template):
      kind = "rest" or "ec2"
      url_template = full url for GET list
    """
    # REST candidates (seen across Nx versions; docs mention devices/*/bookmarks for rest v1/v2)  :contentReference[oaicite:3]{index=3}
    rest_candidates = [
        f"{base_url()}/rest/v2/devices/*/bookmarks",
        f"{base_url()}/rest/v1/devices/*/bookmarks",
        f"{base_url()}/rest/v2/bookmarks",
        f"{base_url()}/rest/v1/bookmarks",
    ]

    for url in rest_candidates:
        data = try_get_json(s, url)
        if isinstance(data, list):
            return ("rest", url)
        if isinstance(data, dict) and ("items" in data or "data" in data):
            return ("rest", url)

    # EC2 / deprecated candidates (community confirms /ec2/bookmarks/... exists in some form) :contentReference[oaicite:4]{index=4}
    ec2_candidates = [
        f"{base_url()}/ec2/bookmarks",
        f"{base_url()}/ec2/bookmarks/list",
        f"{base_url()}/ec2/getBookmarks",
    ]
    for url in ec2_candidates:
        data = try_get_json(s, url)
        if isinstance(data, list) or isinstance(data, dict):
            return ("ec2", url)

    raise RuntimeError("Cannot discover bookmark listing endpoint on this Nx server.")

def normalize_bookmark_items(data: Any) -> List[Dict[str, Any]]:
    """
    Normalize server responses to list[dict].
    """
    if data is None:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            return [x for x in data["items"] if isinstance(x, dict)]
        if isinstance(data.get("data"), list):
            return [x for x in data["data"] if isinstance(x, dict)]
        # Sometimes response is wrapped
        if "reply" in data and isinstance(data["reply"], list):
            return [x for x in data["reply"] if isinstance(x, dict)]
    return []

def bookmark_time_fields(bm: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Try to extract (startMs, durationMs, centralMs) from various schema variants.
    """
    start = bm.get("startTimeMs") or bm.get("startTime") or bm.get("startMs")
    dur = bm.get("durationMs") or bm.get("duration") or bm.get("lengthMs")
    central = bm.get("centralTimePointMs") or bm.get("centralTimeMs") or None
    # Some schemas store microseconds; attempt to detect and convert
    for k in ("startTimeUs", "timestampUs"):
        if start is None and bm.get(k):
            start = int(bm[k] / 1000)
    if isinstance(start, str) and start.isdigit():
        start = int(start)
    if isinstance(dur, str) and dur.isdigit():
        dur = int(dur)
    if isinstance(central, str) and central.isdigit():
        central = int(central)
    return (start if isinstance(start, int) else None,
            dur if isinstance(dur, int) else None,
            central if isinstance(central, int) else None)

def bookmark_text_blob(bm: Dict[str, Any]) -> str:
    parts = []
    for k in ("name", "title", "caption", "description", "comment"):
        v = bm.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    # Tags sometimes are list[str]
    tags = bm.get("tags")
    if isinstance(tags, list):
        parts.extend([str(t) for t in tags])
    return "\n".join(parts).lower()

def keyword_match(bm: Dict[str, Any]) -> bool:
    blob = bookmark_text_blob(bm)

    if FILTER_TAG:
        tags = bm.get("tags")
        if isinstance(tags, list) and any(str(t).lower() == FILTER_TAG.lower() for t in tags):
            return True
        # If server doesn't return tags, fallback to blob search
        if FILTER_TAG.lower() in blob:
            return True
        return False

    if KEYWORDS_ALL:
        for kw in KEYWORDS_ALL:
            if kw.lower() not in blob:
                return False

    if KEYWORDS_ANY:
        return any(kw.lower() in blob for kw in KEYWORDS_ANY)

    # No filters -> accept all
    return True

def in_time_window(bm: Dict[str, Any], start_ms: int, end_ms: int) -> bool:
    st, dur, central = bookmark_time_fields(bm)
    if central is not None:
        t = central
    elif st is not None and dur is not None:
        t = st + dur // 2
    elif st is not None:
        t = st
    else:
        return False
    return start_ms <= t < end_ms

def extract_device_id(bm: Dict[str, Any]) -> Optional[str]:
    # Different schemas: deviceId / cameraId / resourceId
    for k in ("deviceId", "cameraId", "resourceId", "resource", "id"):
        v = bm.get(k)
        if isinstance(v, str) and len(v) >= 6:
            return v
    # Some store nested
    if isinstance(bm.get("device"), dict):
        v = bm["device"].get("id")
        if isinstance(v, str):
            return v
    return None

def discover_snapshot_getter(s: requests.Session) -> str:
    """
    Return a url template that can fetch an image, using device/camera id + timestampMs.
    We'll try common REST endpoints first (v2), then v1, then EC2 thumbnails.
    REST v2 includes /rest/v2/devices/{id}/image and timestampMs naming changes are mentioned in changelog. :contentReference[oaicite:5]{index=5}
    """
    # We'll just return a mode and try in order per request
    return "auto"

def fetch_snapshot_bytes(s: requests.Session, device_id: str, ts_ms: int) -> bytes:
    """
    Try multiple snapshot endpoints until one returns image/*
    """
    candidates = [
        # REST v2
        (f"{base_url()}/rest/v2/devices/{device_id}/image", {"timestampMs": ts_ms, "width": SNAPSHOT_WIDTH}),
        (f"{base_url()}/rest/v2/devices/{device_id}/image", {"time": ts_ms, "width": SNAPSHOT_WIDTH}),
        # REST v1
        (f"{base_url()}/rest/v1/devices/{device_id}/image", {"timestampMs": ts_ms, "width": SNAPSHOT_WIDTH}),
        (f"{base_url()}/rest/v1/devices/{device_id}/image", {"time": ts_ms, "width": SNAPSHOT_WIDTH}),
        # Older web/rest style sometimes used in some builds
        (f"{base_url()}/web/rest/v1/cameras/{device_id}/image", {"timestamp": ts_ms, "width": SNAPSHOT_WIDTH}),
        (f"{base_url()}/web/rest/v1/cameras/{device_id}/image", {"time": ts_ms, "width": SNAPSHOT_WIDTH}),
        # EC2 fallbacks
        (f"{base_url()}/ec2/cameraThumbnail", {"id": device_id, "timestamp": ts_ms}),
        (f"{base_url()}/ec2/cameraThumbnail", {"id": device_id, "time": ts_ms}),
        (f"{base_url()}/ec2/getCameraSnapshot", {"id": device_id, "timestamp": ts_ms}),
        (f"{base_url()}/ec2/getCameraSnapshot", {"id": device_id, "time": ts_ms}),
    ]

    last_err = None
    for url, params in candidates:
        try:
            r = s.get(url, params=params, timeout=20)
            ct = (r.headers.get("Content-Type") or "").lower()
            if r.status_code == 200 and ct.startswith("image/"):
                return r.content
            last_err = f"{r.status_code} {ct} {r.text[:200]}"
        except Exception as e:
            last_err = str(e)

    raise RuntimeError(f"Snapshot failed for device={device_id}, ts_ms={ts_ms}. Last error: {last_err}")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def format_ts(ts_ms: int, tz: timezone) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=tz)
    return dt.strftime("%Y-%m-%d_%H-%M-%S")

def main():
    tz = get_local_tz()

    if TIME_WINDOW_MODE == "day":
        start_ms, end_ms = parse_day_window(DAY_YYYY_MM_DD, tz)
    else:
        start_ms, end_ms = last_hours_window(LOOKBACK_HOURS, tz)

    ensure_dir(OUT_DIR)

    token = login_get_token()
    s = make_session(token)

    kind, list_url = discover_bookmark_list(s)
    print(f"[INFO] Bookmark list endpoint: {list_url} ({kind})")

    # Try requesting list; some endpoints support server-side filter, but we keep it client-side for compatibility.
    data = try_get_json(s, list_url)
    items = normalize_bookmark_items(data)

    if not items:
        print("[WARN] No bookmarks returned by endpoint. Exiting.")
        return

    # Filter by time + keywords/tags
    filtered = []
    for bm in items:
        if in_time_window(bm, start_ms, end_ms) and keyword_match(bm):
            filtered.append(bm)

    filtered = filtered[:MAX_ITEMS]

    print(f"[INFO] Total bookmarks: {len(items)} | matched: {len(filtered)} | window: {format_ts(start_ms, tz)} ~ {format_ts(end_ms, tz)}")

    if not filtered:
        print("[INFO] No matching bookmarks found.")
        return

    ok = 0
    for i, bm in enumerate(filtered, 1):
        st, dur, central = bookmark_time_fields(bm)
        if central is not None:
            snap_ts = central
        elif st is not None and dur is not None:
            snap_ts = st + dur // 2
        elif st is not None:
            snap_ts = st
        else:
            print("[WARN] Skip bookmark without time fields.")
            continue

        device_id = extract_device_id(bm)
        if not device_id:
            print("[WARN] Skip bookmark without device/camera id fields.")
            continue

        bm_id = bm.get("id") or bm.get("bookmarkId") or f"idx{i}"
        ts_str = format_ts(snap_ts, tz)
        out_path = os.path.join(OUT_DIR, f"{ts_str}__{device_id}__{bm_id}.jpg")

        try:
            img = fetch_snapshot_bytes(s, device_id, snap_ts)
            with open(out_path, "wb") as f:
                f.write(img)
            ok += 1
            print(f"[OK] {i}/{len(filtered)} saved: {out_path}")
        except Exception as e:
            print(f"[ERR] {i}/{len(filtered)} bookmark={bm_id} -> {e}")

    print(f"[DONE] saved {ok}/{len(filtered)} snapshots into: {os.path.abspath(OUT_DIR)}")

if __name__ == "__main__":
    main()
