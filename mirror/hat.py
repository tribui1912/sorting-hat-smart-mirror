"""Hat client: tells the ESP32 what to play, falls back to the mirror's own speaker.

Standalone:
    python mirror/hat.py status
    python mirror/hat.py mutter
    python mirror/hat.py play 3           # person id 3 -> 3.wav on the SD card
    python mirror/hat.py play house_slytherin
    python mirror/hat.py stop
    python mirror/hat.py jaw 105          # move the servo, for finding closed/open angles
    python mirror/hat.py play 3 --local   # test the fallback path without the hat
"""

from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import json
import logging
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from common import Config, load_config, setup_logging

log = logging.getLogger("sortinghat.hat")


class LocalAudio:
    """Plays clips through the Pi's own output (HDMI / USB speaker) when the hat is unreachable."""

    def __init__(self, audio_dir: Path, device: str | None = None):
        self.audio_dir = audio_dir
        self.device = device
        self._proc: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        if shutil.which("aplay"):
            self._cmd = ["aplay", "-q"] + (["-D", device] if device else [])
        elif shutil.which("afplay"):  # macOS dev
            self._cmd = ["afplay"]
        else:
            self._cmd = []
            log.warning("no aplay/afplay found: fallback audio disabled")

    def play(self, clip: str) -> bool:
        path = self.audio_dir / f"{clip}.wav"
        if not path.exists():
            log.warning("fallback audio: %s missing", path)
            return False
        if not self._cmd:
            return False
        with self._lock:
            self._stop_locked()
            try:
                self._proc = subprocess.Popen(self._cmd + [str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as e:
                log.error("fallback audio failed: %s", e)
                return False
        log.info("fallback audio: playing %s on the mirror", path.name)
        return True

    def mutter(self) -> bool:
        clips = sorted(self.audio_dir.glob("mutter*.wav"))
        return bool(clips) and self.play(random.choice(clips).stem)

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None


class HatClient:
    def __init__(self, cfg: Config, enabled: bool = True):
        self.cfg = cfg.hat
        self.enabled = enabled
        self.base = f"http://{self.cfg.host}:{self.cfg.port}"
        self.local = LocalAudio(cfg.resolve(cfg.paths.audio), self.cfg.audio_device)
        self.online: bool | None = None  # None = never checked
        self.last_status: dict[str, object] = {}
        self._ip: str | None = None
        self._ip_at = 0.0
        self._dns_failed_at = -1e9
        self._resolver = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="dns")

    def _address(self) -> str | None:
        """Hostname -> IP with a hard timeout, cached.

        urllib's timeout doesn't cover DNS: an unanswered mDNS lookup for sortinghat.local blocks
        for ~5 s, which would blow straight through the 1.5 s ACK window.
        """
        host = self.cfg.host
        try:
            ipaddress.ip_address(host)
            return host
        except ValueError:
            pass
        if self._ip and time.monotonic() - self._ip_at < 60:
            return self._ip
        if time.monotonic() - self._dns_failed_at < self.cfg.ping_every_s:
            return None  # failed moments ago: fail fast and let the show fall back immediately
        fut = self._resolver.submit(socket.gethostbyname, host)
        try:
            self._ip = fut.result(timeout=min(1.0, self.cfg.timeout_s))
            self._ip_at = time.monotonic()
        except (concurrent.futures.TimeoutError, OSError) as e:
            log.warning("hat: can't resolve %s (%s) - use its IP in config hat.host", host, str(e) or "timeout")
            self._ip = None
            self._dns_failed_at = time.monotonic()
        return self._ip

    # -- raw endpoints. Each returns True only on an explicit ACK (HTTP 200).

    def _get(self, path: str, params: dict[str, str] | None = None, timeout: float | None = None) -> str | None:
        if not self.enabled:
            return None
        ip = self._address()
        if ip is None:
            self.online = False
            return None
        url = f"http://{ip}:{self.cfg.port}{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
        try:
            with urllib.request.urlopen(url, timeout=timeout or self.cfg.timeout_s) as resp:
                body = resp.read().decode("utf-8", "replace")
                self.online = True
                return body
        except urllib.error.HTTPError as e:
            self.online = True  # it answered, it just didn't like the request
            log.warning("hat %s -> HTTP %s %s", path, e.code, e.read().decode("utf-8", "replace").strip())
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self.online = False
            self._ip = None  # re-resolve next time, in case DHCP moved it
            log.warning("hat %s -> no answer (%s)", path, getattr(e, "reason", e))
        return None

    def play_remote(self, clip: str) -> bool:
        return self._get("/play", {"id": clip}) is not None

    def mutter_remote(self) -> bool:
        return self._get("/mutter") is not None

    def stop_remote(self) -> bool:
        return self._get("/stop") is not None

    def jaw(self, deg: int) -> bool:
        return self._get("/jaw", {"deg": str(deg)}) is not None

    def status(self, timeout: float | None = None) -> dict[str, object] | None:
        body = self._get("/status", timeout=timeout)
        if body is None:
            return None
        try:
            self.last_status = json.loads(body)
        except json.JSONDecodeError:
            self.last_status = {"raw": body}
        return self.last_status

    # -- what the show uses: hat first, mirror speaker as fallback

    def play(self, clip: str) -> str:
        """Returns 'hat', 'local' or 'none' depending on where the clip ended up playing."""
        self.local.stop()
        if self.play_remote(clip):
            return "hat"
        if self.cfg.fallback_audio and self.local.play(clip):
            return "local"
        return "none"

    def mutter(self) -> str:
        self.local.stop()
        if self.mutter_remote():
            return "hat"
        if self.cfg.fallback_audio and self.local.mutter():
            return "local"
        return "none"

    def stop(self) -> None:
        self.local.stop()
        if self.online is not False:
            self.stop_remote()

    def monitor(self, stop: threading.Event) -> None:
        """Background health ping so the override page can show whether the hat is alive."""
        while not stop.is_set():
            was = self.online
            self.status(timeout=2.0)
            if was is not None and was != self.online:
                log.warning("hat is now %s", "ONLINE" if self.online else "OFFLINE")
            stop.wait(self.cfg.ping_every_s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["status", "play", "mutter", "stop", "jaw"])
    ap.add_argument("arg", nargs="?", help="clip id for play, degrees for jaw")
    ap.add_argument("--config")
    ap.add_argument("--host", help="override hat.host")
    ap.add_argument("--local", action="store_true", help="skip the hat, play through this machine")
    args = ap.parse_args()

    cfg = load_config(args.config)
    setup_logging()
    if args.host:
        cfg.hat.host = args.host
    hat = HatClient(cfg, enabled=not args.local)

    if args.command == "status":
        st = hat.status()
        print(json.dumps(st, indent=2) if st else f"no answer from {hat.base}")
        sys.exit(0 if st else 1)
    if args.command == "play":
        if not args.arg:
            ap.error("play needs a clip id")
        where = hat.play(args.arg)
    elif args.command == "mutter":
        where = hat.mutter()
    elif args.command == "stop":
        hat.stop()
        return
    else:
        if not args.arg:
            ap.error("jaw needs degrees")
        sys.exit(0 if hat.jaw(int(args.arg)) else 1)
    print(f"played on: {where}")
    if where == "local" and hat.local._proc:
        hat.local._proc.wait()  # keep the process alive until the clip ends
    sys.exit(0 if where != "none" else 1)


if __name__ == "__main__":
    main()
