#!/usr/bin/env bash
# One-time setup on the Pi 5 (Pi OS Bookworm 64-bit). Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

sudo apt update
sudo apt install -y python3-picamera2 python3-venv python3-dev build-essential \
                    ffmpeg alsa-utils libsdl2-ttf-2.0-0 fonts-dejavu-core

# --system-site-packages: picamera2/libcamera only exist as apt packages
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip wheel
.venv/bin/pip install -r requirements.txt

# Download the face models now (needs internet once; cached in ~/.insightface for the party)
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "mirror")
from common import load_config
from faces import FaceEngine
FaceEngine(load_config().recognition)
print("face models ready")
PY

mkdir -p data audio/source audio/sd logs
echo
echo "Done. Next:"
echo "  .venv/bin/python mirror/display.py --windowed     # build step 1-3: keys drive the show"
echo "  .venv/bin/python mirror/recognize.py --preview    # camera + detection check"
