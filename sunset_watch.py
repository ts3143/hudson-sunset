#!/usr/bin/env python3
"""
sunset_watch.py — Phase 0 sunset alert

Grabs a frame from an RTSP camera every minute around sunset, scores how
colorful the sky is, saves every frame + a CSV log, and sends a push alert
via ntfy when it's worth going outside (plus an "afterglow" alert).

Setup:
    pip install opencv-python numpy requests imageio-ffmpeg
    Install the ntfy app on your phone and subscribe to a topic, then:
    export NTFY_TOPIC=your-topic        # add to ~/.zshrc so it sticks

Usage:
    python sunset_watch.py --once            # grab 1 frame, score it, save it (connection test)
    python sunset_watch.py --score img.jpg   # score an existing photo (tune thresholds)
    python sunset_watch.py --now             # run the loop right now, ignoring sunset times
    python sunset_watch.py                   # normal mode: runs every evening around sunset
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import requests

# ---------------------------------------------------------------- config ----
RTSP_URL = os.environ.get("RTSP_URL", "rtsp://192.168.12.137")   # or: export RTSP_URL=...
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")   # set via: export NTFY_TOPIC=your-topic (keep it out of git)

LAT, LON = 40.80, -73.97                  # Riverside Dr, UWS
try:                                      # Python 3.9+
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("America/New_York")
except ImportError:                       # older Python: use the computer's own time zone
    TZ = None

WINDOW_BEFORE_MIN = 45                    # start watching this long before sunset
WINDOW_AFTER_MIN = 40                     # keep watching this long after (afterglow)
INTERVAL_SEC = 60                         # one frame per minute

SKY_TOP_FRACTION = None                   # None = auto-detect horizon; or fix it, e.g. 0.60 = top 60% is sky
MASK_RECTS = []                           # extra areas to ignore, as fractions (x0, y0, x1, y1)
                                          # e.g. app watermark top-left: [(0.0, 0.0, 0.25, 0.06)]

ALERT_SCORE = 5.0                         # 0–10; tune after a week of data
AFTERGLOW_START_MIN = 8                   # "afterglow" = this many min after sunset or later

DATA_DIR = Path("sunset_data")

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def now_local() -> datetime:
    return now_local() if TZ else datetime.now().astimezone()


def to_local(dt: datetime) -> datetime:
    return dt.astimezone(TZ) if TZ else dt.astimezone()


# ------------------------------------------------------------ sunset time ---
def sunset_utc(d: date, lat: float = LAT, lon: float = LON) -> datetime:
    """NOAA solar calculation; accurate to ~1 min. Returns sunset in UTC."""
    jd = d.toordinal() + 1721424.5 + 0.5          # Julian day at noon UTC
    for _ in range(2):                            # iterate once to refine
        t = (jd - 2451545.0) / 36525.0
        l0 = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360
        m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
        e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
        mr = math.radians(m)
        c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t))
             + math.sin(2 * mr) * (0.019993 - 0.000101 * t)
             + math.sin(3 * mr) * 0.000289)
        app = l0 + c - 0.00569 - 0.00478 * math.sin(math.radians(125.04 - 1934.136 * t))
        eps0 = 23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
        eps = eps0 + 0.00256 * math.cos(math.radians(125.04 - 1934.136 * t))
        decl = math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(app)))
        y = math.tan(math.radians(eps / 2)) ** 2
        l0r = math.radians(l0)
        eqtime = 4 * math.degrees(
            y * math.sin(2 * l0r) - 2 * e * math.sin(mr)
            + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
            - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr))
        latr = math.radians(lat)
        cos_ha = (math.cos(math.radians(90.833)) / (math.cos(latr) * math.cos(decl))
                  - math.tan(latr) * math.tan(decl))
        ha = math.degrees(math.acos(max(-1.0, min(1.0, cos_ha))))
        minutes = 720 - 4 * (lon - ha) - eqtime     # minutes after 00:00 UTC
        jd = d.toordinal() + 1721424.5 + minutes / 1440
    base = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return base + timedelta(minutes=minutes)


def sunset_local(d: date) -> datetime:
    return to_local(sunset_utc(d))


# ---------------------------------------------------------------- camera ----
def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg                 # pip install imageio-ffmpeg (bundles an ffmpeg binary)
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


FFMPEG = find_ffmpeg()
LAST_ERROR = ""


def grab_frame_ffmpeg(url: str, transport: str):
    global LAST_ERROR
    cmd = [FFMPEG, "-loglevel", "error", "-rtsp_transport", transport, "-i", url,
           "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=25)
    except subprocess.TimeoutExpired:
        LAST_ERROR = f"ffmpeg timed out ({transport})"
        return None
    if out.returncode != 0 or not out.stdout:
        LAST_ERROR = f"ffmpeg ({transport}): " + out.stderr.decode(errors="ignore").strip()[-300:]
        return None
    return cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)


def grab_frame_cv2(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    frame = None
    if cap.isOpened():
        for _ in range(5):                    # flush stale buffered frames
            ok, f = cap.read()
            if ok:
                frame = f
    cap.release()
    return frame


def grab_frame(url: str = RTSP_URL, tries: int = 2):
    """Return the latest frame (BGR) or None. Uses ffmpeg if available, else OpenCV."""
    for _ in range(tries):
        if FFMPEG:
            for transport in ("tcp", "udp"):
                frame = grab_frame_ffmpeg(url, transport)
                if frame is not None:
                    return frame
        else:
            frame = grab_frame_cv2(url)
            if frame is not None:
                return frame
        time.sleep(3)
    return None


# --------------------------------------------------------------- scoring ----
def horizon_row(bgr: np.ndarray) -> int:
    """Find the skyline: the row with the sharpest bright-sky → darker-land drop."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float)
    h = gray.shape[0]
    rows = np.convolve(np.median(gray, axis=1), np.ones(9) / 9, mode="same")
    win = max(3, h // 30)
    best, best_r = -1e9, int(h * 0.6)
    for r in range(int(h * 0.25), int(h * 0.85)):
        drop = rows[r - win:r].mean() - rows[r:r + win].mean()
        if drop > best:
            best, best_r = drop, r
    return best_r


def sky_mask(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    top = int(h * SKY_TOP_FRACTION) if SKY_TOP_FRACTION else horizon_row(bgr) - int(0.01 * h)
    mask = np.zeros((h, w), dtype=bool)
    mask[: max(1, top), :] = True
    for x0, y0, x1, y1 in MASK_RECTS:
        mask[int(y0 * h): int(y1 * h), int(x0 * w): int(x1 * w)] = False
    return mask


def score_frame(bgr: np.ndarray) -> dict:
    """
    0–10 sunset score from sky color (v2).
      warmth    = per-pixel weight for red/orange/yellow/pink/purple, scaled by saturation,
                  so pastel skies count partially and vivid ones fully
      coverage  = average warmth over the whole sky   (big colorful skies)
      band      = warmth of the strongest horizontal strip (thin glowing horizon bands)
      score     = 40% coverage + 60% band
    """
    small = cv2.resize(bgr, (640, int(640 * bgr.shape[0] / bgr.shape[1])))
    h = small.shape[0]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(float)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]      # OpenCV hue is 0–179

    sky = sky_mask(small)
    valid = sky & (val >= 40)                                   # ignore near-black
    if valid.sum() < 100:
        return {"score": 0.0, "coverage": 0.0, "band": 0.0, "brightness": 0.0}

    warm_hue = (hue <= 35) | (hue >= 125)                       # red→yellow, purple→pink
    warmth = warm_hue * np.clip((sat - 30) / 120, 0, 1) * np.clip(val / 120, 0, 1)
    warmth = np.where(valid, warmth, 0.0)

    coverage = float(warmth[valid].mean())
    band_h = max(4, int(0.06 * h))
    sky_bottom = int(np.where(sky.any(axis=1))[0].max()) + 1
    bands = [float(warmth[y:y + band_h][valid[y:y + band_h]].mean())
             for y in range(0, max(1, sky_bottom - band_h + 1), max(1, band_h // 2))
             if valid[y:y + band_h].sum() > 50]
    band = max(bands) if bands else 0.0

    raw = 0.4 * min(coverage / 0.30, 1.0) + 0.6 * min(band / 0.45, 1.0)
    return {"score": round(10 * raw, 1), "coverage": round(coverage, 3),
            "band": round(band, 3), "brightness": round(float(val[valid].mean() / 255), 3)}


# ----------------------------------------------------------------- alerts ---
def notify(title: str, message: str, jpg: bytes | None = None):
    if not NTFY_TOPIC:
        print("[warn] NTFY_TOPIC not set - skipping push. Run: export NTFY_TOPIC=your-topic")
        return
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    headers = {"Title": title, "Tags": "sunrise", "Priority": "high"}
    try:
        if jpg:
            headers.update({"Message": message, "Filename": "sunset.jpg"})
            requests.put(url, data=jpg, headers=headers, timeout=20)
        else:
            requests.post(url, data=message.encode(), headers=headers, timeout=20)
    except requests.RequestException as e:
        print(f"[warn] notify failed: {e}")


# ------------------------------------------------------------------ loop ----
def save(frame, ts: datetime, result: dict) -> Path:
    day_dir = DATA_DIR / ts.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{ts.strftime('%H%M%S')}_s{result['score']:.1f}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    log = DATA_DIR / "log.csv"
    new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "min_from_sunset", "score", "coverage",
                        "band", "brightness", "file"])
        w.writerow([ts.isoformat(), result.get("min_from_sunset"), result["score"],
                    result["coverage"], result["band"], result["brightness"], path.name])
    return path


def watch_evening(start: datetime, end: datetime, sunset: datetime):
    print(f"Watching {start:%H:%M}–{end:%H:%M} (sunset {sunset:%H:%M})")
    alerted_main = alerted_after = False
    peak = 0.0
    history: list[float] = []

    while now_local() < end:
        loop_start = time.time()
        now = now_local()
        frame = grab_frame()
        if frame is None:
            print(f"{now:%H:%M:%S}  [no frame — check the phone/app]")
        else:
            r = score_frame(frame)
            r["min_from_sunset"] = round((now - sunset).total_seconds() / 60, 1)
            path = save(frame, now, r)
            history.append(r["score"])
            rising = len(history) >= 3 and history[-1] >= history[-3]
            print(f"{now:%H:%M:%S}  {r['min_from_sunset']:+6.1f}m  score {r['score']:4.1f}  "
                  f"cov {r['coverage']:.2f}  band {r['band']:.2f}")

            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            jpg = jpg.tobytes() if ok else None

            is_afterglow = r["min_from_sunset"] >= AFTERGLOW_START_MIN
            if not alerted_main and not is_afterglow and r["score"] >= ALERT_SCORE and rising:
                notify(f"Sunset {r['score']}/10 - go look west",
                       f"Sky is lighting up. Sunset at {sunset:%-I:%M}.", jpg)
                alerted_main = True
            elif (is_afterglow and not alerted_after and r["score"] >= ALERT_SCORE
                  and r["score"] >= peak):
                notify(f"Afterglow {r['score']}/10 - don't leave yet",
                       "Second wave of color after sunset.", jpg)
                alerted_after = True
            peak = max(peak, r["score"])

        time.sleep(max(0, INTERVAL_SEC - (time.time() - loop_start)))

    print(f"Done for tonight. Peak score {peak:.1f}")


def run_forever():
    while True:
        today = now_local().date()
        ss = sunset_local(today)
        start = ss - timedelta(minutes=WINDOW_BEFORE_MIN)
        end = ss + timedelta(minutes=WINDOW_AFTER_MIN)
        now = now_local()
        if now >= end:                                    # tonight is over → tomorrow
            ss = sunset_local(today + timedelta(days=1))
            start = ss - timedelta(minutes=WINDOW_BEFORE_MIN)
            end = ss + timedelta(minutes=WINDOW_AFTER_MIN)
        if now < start:
            wait = (start - now).total_seconds()
            print(f"Next sunset {ss:%a %H:%M}. Sleeping {wait/3600:.1f}h until {start:%H:%M}.")
            time.sleep(wait)
        watch_evening(max(start, now_local()), end, ss)


# ------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser(description="Sunset alert camera")
    ap.add_argument("--once", action="store_true", help="grab + score one frame, then exit")
    ap.add_argument("--score", metavar="IMG", help="score an image file, then exit")
    ap.add_argument("--now", action="store_true", help="run a 45-min watch starting now")
    ap.add_argument("--test-alert", action="store_true", help="send a test push")
    args = ap.parse_args()

    if args.score:
        img = cv2.imread(args.score)
        if img is None:
            sys.exit(f"Can't read {args.score}")
        print(score_frame(img))
    elif args.once:
        frame = grab_frame()
        if frame is None:
            sys.exit(f"No frame from {RTSP_URL}.\n"
                     f"ffmpeg: {FFMPEG or 'NOT FOUND - run: pip install imageio-ffmpeg'}\n"
                     f"last error: {LAST_ERROR or '-'}")
        now = now_local()
        r = score_frame(frame)
        r["min_from_sunset"] = round((now - sunset_local(now.date())).total_seconds() / 60, 1)
        print(r, "->", save(frame, now, r))
    elif args.test_alert:
        notify("Test - sunset watch", "If you see this, alerts work.")
        print("Sent.")
    elif args.now:
        now = now_local()
        watch_evening(now, now + timedelta(minutes=45), sunset_local(now.date()))
    else:
        run_forever()


if __name__ == "__main__":
    main()
