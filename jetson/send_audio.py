#!/usr/bin/env python3
"""
send_audio.py -- PC/Jetson side of the sorting-hat audio link.

Converts any audio file to 16-bit mono 22050 Hz PCM (via ffmpeg) and
streams it to the hat firmware over TCP. The hat plays it through the
MAX98357A and flaps the mouth with the loudness. Returns/blocks until
the hat replies DONE, so queued files play in order.

For now this is a standalone bench tool; later the Jetson's
facial-recognition flow can `from send_audio import command` and call
it with the right clip for the person it recognised.

Usage:
  python jetson/send_audio.py --host <hat-ip> --ping
  python jetson/send_audio.py --host <hat-ip> --clip
  python jetson/send_audio.py --host <hat-ip> line1.mp3 line2.mp3
  # one utterance from parts (opening + name + ending) with 250 ms pauses:
  python jetson/send_audio.py --host <hat-ip> --gap 250 open.mp3 name.mp3 end.mp3

The --gap mode concatenates the files with silence and sends ONE PLAY
stream, so the hat's mouth tracking stays continuous across the parts.
Without --gap, each file is a separate utterance.

To test without a hat: python jetson/mock_hat.py, then --host 127.0.0.1.
Protocol: PING->PONG ; CLIP->DONE ; "PLAY <n>" + n PCM bytes -> OK ... DONE.
"""

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path

RATE = 22050
PORT = 5005
CONFIG = Path(__file__).resolve().parent / "hat.json"   # {"host": "...", "port": 5005}


def load_config() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def to_pcm(path: Path) -> bytes:
    """Any audio file -> raw signed 16-bit LE mono PCM at RATE."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-ar", str(RATE), "-ac", "1", "-f", "s16le", "-"],
            capture_output=True, check=True)
    except FileNotFoundError:
        sys.exit("error: ffmpeg not found on PATH")
    except subprocess.CalledProcessError as e:
        sys.exit(f"error: ffmpeg failed on {path}: {e.stderr.decode(errors='replace').strip()}")
    return r.stdout


def recv_line(sock) -> str:
    buf = b""
    while not buf.endswith(b"\n"):
        d = sock.recv(64)
        if not d:
            break
        buf += d
    return buf.decode(errors="replace").strip()


def command(host: str, cmd: str, payload: bytes = b"", port: int = PORT) -> str:
    """Send one protocol command, return the hat's final reply (PONG/DONE/...)."""
    with socket.create_connection((host, port), timeout=5) as s:
        s.settimeout(120)          # DONE arrives after playback ends
        s.sendall(cmd.encode() + b"\n")
        if payload:
            ack = recv_line(s)
            if ack != "OK":
                return ack or "no reply"
            s.sendall(payload)
        return recv_line(s)


def silence(ms: int) -> bytes:
    return b"\x00" * (RATE * 2 * ms // 1000)


def play_line(host: str, parts, gap_ms: int = 250, port: int = PORT) -> str:
    """Play several audio files as ONE utterance: parts joined by gap_ms of
    silence, streamed as a single PLAY. The vision code will call this with
    e.g. (opening_for(person), ending_for(house))."""
    pcms = [to_pcm(Path(p)) for p in parts]
    pcm = silence(gap_ms).join(pcms)
    print(f"line: {' + '.join(f'{len(p) / (RATE * 2):.2f}s' for p in pcms)}"
          f" (+{gap_ms} ms gaps) = {len(pcm) / (RATE * 2):.2f} s, {len(pcm)} B ... ",
          end="", flush=True)
    return command(host, f"PLAY {len(pcm)}", pcm, port)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(
        description="Send audio to the sorting hat over WiFi.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--host", default=cfg.get("host") or None,
                    help="hat IP from its Serial Monitor "
                         f"(default: host in {CONFIG.name})")
    ap.add_argument("--port", type=int, default=int(cfg.get("port", PORT)))
    ap.add_argument("--ping", action="store_true", help="connectivity check only")
    ap.add_argument("--clip", action="store_true", help="play the hat's embedded bench clip")
    ap.add_argument("--gap", type=int, default=None, metavar="MS",
                    help="join files into ONE utterance with MS ms of silence between parts")
    ap.add_argument("files", nargs="*", help="audio files to convert and stream, in order")
    args = ap.parse_args()

    if args.ping:
        print(command(args.host, "PING", port=args.port))
    if args.clip:
        print(command(args.host, "CLIP", port=args.port))
    if args.gap is not None and args.files:
        print(play_line(args.host, args.files, gap_ms=args.gap, port=args.port))
    else:
        for f in args.files:
            pcm = to_pcm(Path(f))
            print(f"{f}: {len(pcm) / (RATE * 2):.2f} s, {len(pcm)} B ... ",
                  end="", flush=True)
            print(command(args.host, f"PLAY {len(pcm)}", pcm, port=args.port))
    if not (args.ping or args.clip or args.files):
        ap.error("nothing to do: give files, --ping or --clip")


if __name__ == "__main__":
    main()
