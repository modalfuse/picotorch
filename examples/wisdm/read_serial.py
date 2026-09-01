#!/usr/bin/env python3
"""Reset ESP32-S3 over CH340 and print serial until a ready line."""

from __future__ import annotations

import argparse
import os
import sys
import time

import serial


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="serial device; or set PICOTORCH_SERIAL")
    ap.add_argument("--ready", default="ready")
    ap.add_argument("--seconds", type=float, default=20)
    args = ap.parse_args()
    port = args.port or os.environ.get("PICOTORCH_SERIAL")
    if not port:
        raise SystemExit("set --port or PICOTORCH_SERIAL to the board serial device")
    ser = serial.Serial(port, 115200, timeout=0.2)
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.05)
    ser.setRTS(False)
    deadline = time.time() + args.seconds
    buf = ""
    while time.time() < deadline:
        chunk = ser.read(4096).decode("utf-8", errors="replace")
        if chunk:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            buf += chunk
            if args.ready in buf:
                time.sleep(0.3)
                extra = ser.read(4096).decode("utf-8", errors="replace")
                if extra:
                    sys.stdout.write(extra)
                return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
