#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nx_generic_event.py

Goal:
  1) Send Nx Witness HTTP Generic Event (CreateEvent)
  2) (Optional) List cameras to find camera IDs (UUID) for metadata.cameraRefs

How to run:
  python nx_generic_event.py send
  python nx_generic_event.py list-cameras

Notes:
  - Some Nx servers allow unauthenticated createEvent; some require auth (401/403).
  - If you get 401/403, switch AUTH_MODE and implement token login for your server.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import requests
from requests.auth import HTTPBasicAuth


# =========================
# User Config (hard-coded)
# =========================

NX_HOST = "192.168.72.176"
NX_PORT = 7001

USE_HTTPS = True               # True if your server is https
VERIFY_TLS = False              # False for self-signed cert (testing)

# Authentication:
#   "none"   -> no auth header
#   "basic"  -> HTTP Basic (may not work depending on Nx setup)
#   "bearer" -> Authorization: Bearer <token>  (token getter is placeholder)
AUTH_MODE = "bearer"

NX_USERNAME = "admin"
NX_PASSWORD = "3.1415926Pi2143"

# If AUTH_MODE == "bearer", you can paste a token here for quick test,
# or implement get_bearer_token() for your environment.
BEARER_TOKEN = ""


# Default event content (you can override via CLI args too)
EVENT_SOURCE = "BaysideTrailer12_AI"
EVENT_CAPTION = "illegal_parking_confirmed"
EVENT_DESCRIPTION = "plate=ABC123;score=0.91;duration=12m"

# Bind event to cameras (so clicking notification shows streams)
# Put Nx camera UUID(s) here (strings). Empty list is allowed.
CAMERA_REFS = [
    # "0f3d9b2e-....-....-....-............",
]


# =========================
# Helpers
# =========================

def build_base_url() -> str:
    scheme = "https" if USE_HTTPS else "http"
    return f"{scheme}://{NX_HOST}:{NX_PORT}"


def now_iso_no_ms_local() -> str:
    """
    Nx docs show: YYYY-MM-DDTHH:MM:SS
    We'll use local time without milliseconds.
    """
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def make_session() -> requests.Session:
    s = requests.Session()
    s.verify = VERIFY_TLS

    # Auth header setup
    if AUTH_MODE == "basic":
        s.auth = HTTPBasicAuth(NX_USERNAME, NX_PASSWORD)

    elif AUTH_MODE == "bearer":
        token = BEARER_TOKEN.strip() or get_bearer_token()
        if not token:
            raise RuntimeError("AUTH_MODE=bearer but no token available.")
        s.headers.update({"Authorization": f"Bearer {token}"})

    elif AUTH_MODE == "none":
        pass

    else:
        raise ValueError(f"Unknown AUTH_MODE: {AUTH_MODE}")

    return s


def get_bearer_token() -> str:
    """
    Login to Nx Witness REST API and get a bearer/session token.
    Endpoint seen in Nx examples:
      POST https://<server>:7001/web/rest/v1/login/sessions
    """
    url = f"{build_base_url()}/web/rest/v1/login/sessions"

    payload = {
        "username": NX_USERNAME,
        "password": NX_PASSWORD,
        "setCookie": False
    }

    # 用 requests 直接发
    r = requests.post(url, json=payload, timeout=10, verify=VERIFY_TLS)

    if r.status_code != 200:
        raise RuntimeError(f"Login failed: {r.status_code} {r.text[:500]}")

    data = r.json()
    token = data.get("token") or data.get("accessToken") or data.get("id")

    if not token:
        raise RuntimeError(f"Login ok but token not found in response: {data}")

    return token


# =========================
# Core: Create Event
# =========================

def send_generic_event(
    session: requests.Session,
    timestamp: Optional[str] = None,
    source: str = EVENT_SOURCE,
    caption: str = EVENT_CAPTION,
    description: str = EVENT_DESCRIPTION,
    camera_refs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Send Nx Witness Generic Event using POST JSON to /api/createEvent
    (more robust than URL query encoding).

    If your server only supports query-string version, tell me and I’ll provide that fallback.
    """
    url = f"{build_base_url()}/api/createEvent"

    payload: Dict[str, Any] = {
        "timestamp": timestamp or now_iso_no_ms_local(),
        "source": source,
        "caption": caption,
        "description": description,
    }

    camera_refs = camera_refs or []
    if camera_refs:
        payload["metadata"] = {"cameraRefs": camera_refs}

    r = session.post(url, json=payload, timeout=10)
    return {
        "url": url,
        "status_code": r.status_code,
        "response_text": r.text[:2000],
        "request_payload": payload,
    }


# =========================
# Optional: List Cameras
# =========================

def list_cameras(session: requests.Session) -> Dict[str, Any]:
    """
    Try to list cameras (to discover camera UUID).
    The doc you pasted mentions:
      /ec2/getCamerasEx?extraFormatting
    We'll call that.

    If your server requires auth, configure AUTH_MODE accordingly.
    """
    url = f"{build_base_url()}/ec2/getCamerasEx?extraFormatting"
    r = session.get(url, timeout=10)

    # Response might be XML-ish / JSON-ish depending on system.
    # We'll just return raw text and you can search for "id".
    return {
        "url": url,
        "status_code": r.status_code,
        "response_text": r.text[:4000],
    }


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="Nx Witness HTTP Generic Event sender")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_send = sub.add_parser("send", help="Send a Generic Event")
    p_send.add_argument("--source", default=EVENT_SOURCE)
    p_send.add_argument("--caption", default=EVENT_CAPTION)
    p_send.add_argument("--description", default=EVENT_DESCRIPTION)
    p_send.add_argument("--timestamp", default=None, help="YYYY-MM-DDTHH:MM:SS")
    p_send.add_argument("--camera-ref", action="append", default=None,
                        help="Camera UUID; can be repeated. If not provided, uses CAMERA_REFS in script.")

    p_list = sub.add_parser("list-cameras", help="Query camera list to find camera IDs")

    args = parser.parse_args()

    try:
        session = make_session()

        if args.cmd == "send":
            camera_refs = args.camera_ref if args.camera_ref is not None else CAMERA_REFS
            result = send_generic_event(
                session=session,
                timestamp=args.timestamp,
                source=args.source,
                caption=args.caption,
                description=args.description,
                camera_refs=camera_refs,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))

            if result["status_code"] in (401, 403):
                print("\n[Hint] Server rejected request (401/403). Set AUTH_MODE and provide token/login.", file=sys.stderr)

        elif args.cmd == "list-cameras":
            result = list_cameras(session=session)
            print(json.dumps(result, ensure_ascii=False, indent=2))

            if result["status_code"] in (401, 403):
                print("\n[Hint] list-cameras needs auth on your server. Set AUTH_MODE accordingly.", file=sys.stderr)

    except requests.exceptions.SSLError as e:
        print(f"[SSL Error] {e}", file=sys.stderr)
        print("Try setting USE_HTTPS correctly and/or VERIFY_TLS=False for self-signed cert.", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
