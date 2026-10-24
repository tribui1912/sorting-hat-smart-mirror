"""Turn voice clips into ESP32-ready audio + jaw envelopes.

For each input clip writes, into the output folder (= the SD card root):
    <name>.wav   16-bit mono 22050 Hz PCM, canonical 44-byte header
    <name>.env   raw bytes, one servo angle (degrees) per 20 ms frame

Envelope pipeline:
    RMS per 20 ms -> rolling average (~3 frames) -> noise floor (jaw shuts below it)
    -> map normal speech to 60-70% of travel, peaks up to 100% -> shift ~40 ms earlier

Usage:
    python mirror/make_envelope.py clip.wav                  # one file -> audio/sd/
    python mirror/make_envelope.py audio/source/             # batch a folder
    python mirror/make_envelope.py audio/source/ --preview   # print an ASCII jaw trace

Naming (batch mode): files named after a person in people.json (e.g. "Alice.m4a") are written
as their id ("3.wav", "3.env"). Everything else keeps its name, so record:
    mutter1.wav, mutter2.wav ...           -> /mutter (random pick)
    house_gryffindor.wav ... (one per house) -> guests and the hat's physical button
Non-WAV input (mp3/m4a/...) needs ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

from common import EnvelopeCfg, People, load_config

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".aiff", ".aif"}


# ---------------------------------------------------------------------------- audio io


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Any PCM WAV -> float32 mono in [-1, 1], sample rate."""
    with wave.open(str(path), "rb") as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        i = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        i = np.where(i & 0x800000, i - 0x1000000, i)
        x = i.astype(np.float32) / 8388608
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError(f"{path}: unsupported sample width {width}")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, rate


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix.lower() == ".wav":
        try:
            return read_wav(path)
        except (wave.Error, ValueError):
            pass  # float WAV, WAVE_FORMAT_EXTENSIBLE etc. -> let ffmpeg handle it
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"{path.name}: needs ffmpeg to decode (sudo apt install ffmpeg / brew install ffmpeg)")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "x.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-ac", "1", "-c:a", "pcm_s16le", str(tmp)],
                       check=True)
        return read_wav(tmp)


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(src, dst)
        return resample_poly(x, dst // g, src // g).astype(np.float32)
    except ImportError:  # linear interpolation: fine for speech on a 3W speaker
        n = int(round(len(x) * dst / src))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def write_wav(path: Path, x: np.ndarray, rate: int) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:  # stdlib writes the plain 44-byte header the firmware expects
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------- envelope


def compute_envelope(x: np.ndarray, rate: int, cfg: EnvelopeCfg) -> np.ndarray:
    """Audio -> uint8 servo angle per frame."""
    hop = int(rate * cfg.frame_ms / 1000)
    n_frames = int(np.ceil(len(x) / hop))
    padded = np.zeros(n_frames * hop, np.float32)
    padded[: len(x)] = x
    rms = np.sqrt(np.mean(padded.reshape(n_frames, hop) ** 2, axis=1))

    # smooth: centred rolling mean over smooth_frames
    k = max(1, cfg.smooth_frames)
    if k > 1:
        rms = np.convolve(rms, np.ones(k) / k, mode="same")

    # reference "normal speech" level = 90th percentile of the frames that have any sound
    voiced = rms[rms > rms.max() * 0.02] if rms.max() > 0 else rms
    ref = float(np.percentile(voiced, 90)) if len(voiced) else 1.0
    peak = float(rms.max()) or 1.0
    ref = max(min(ref, peak), 1e-6)
    level = rms / ref  # ~1.0 for normal speech, >1 for shouts

    floor = cfg.noise_floor
    # speech: floor..1 -> 0..speech_travel ; peaks: 1..peak/ref -> speech_travel..1
    speech = np.clip((level - floor) / (1 - floor), 0, 1) ** cfg.gamma * cfg.speech_travel
    over = peak / ref - 1
    extra = np.clip((level - 1) / over, 0, 1) * (1 - cfg.speech_travel) if over > 1e-3 else 0
    openness = np.where(level > 1, cfg.speech_travel + extra, speech)
    openness[level < floor] = 0

    # lead: the jaw starts opening before the sound arrives
    lead = int(round(cfg.lead_ms / cfg.frame_ms))
    if lead > 0:
        openness = np.concatenate([openness[lead:], np.zeros(lead)])

    angles = cfg.closed_deg + openness * (cfg.open_deg - cfg.closed_deg)
    angles = np.append(angles, cfg.closed_deg)  # always end shut
    return np.clip(np.round(angles), 0, 180).astype(np.uint8)


def preview(env: np.ndarray, cfg: EnvelopeCfg, width: int = 50) -> str:
    lo, hi = sorted((cfg.closed_deg, cfg.open_deg))
    rows = []
    step = max(1, len(env) // 60)
    for i in range(0, len(env), step):
        frac = (abs(int(env[i]) - cfg.closed_deg)) / max(1, hi - lo)
        rows.append(f"{i * cfg.frame_ms / 1000:6.2f}s |{'#' * int(frac * width):<{width}}| {env[i]}")
    return "\n".join(rows)


# ---------------------------------------------------------------------------- driver


def output_name(src: Path, people: People | None) -> str:
    if people:
        p = people.get(src.stem) or next((q for q in people.all() if q.name.lower() == src.stem.lower()), None)
        if p and p.id is not None:
            return str(p.id)
    return src.stem


def process(src: Path, out_dir: Path, cfg: EnvelopeCfg, people: People | None, show_preview: bool) -> None:
    x, rate = load_audio(src)
    x = resample(x, rate, cfg.sample_rate)
    x = x - float(np.mean(x))  # remove DC so silence really is silent
    if cfg.normalize_dbfs is not None and np.abs(x).max() > 0:
        x = x * (10 ** (cfg.normalize_dbfs / 20) / np.abs(x).max())
    env = compute_envelope(x, cfg.sample_rate, cfg)
    name = output_name(src, people)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_wav(out_dir / f"{name}.wav", x, cfg.sample_rate)
    (out_dir / f"{name}.env").write_bytes(env.tobytes())
    dur = len(x) / cfg.sample_rate
    moving = float(np.mean(env != cfg.closed_deg)) * 100
    print(f"{src.name:28s} -> {name}.wav/.env  {dur:5.1f}s  {len(env)} frames  jaw moving {moving:3.0f}% of the time")
    if moving < 10:
        print("   ! jaw barely moves: lower envelope.noise_floor or check the clip isn't mostly silence")
    if show_preview:
        print(preview(env, cfg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="audio files or folders")
    ap.add_argument("-o", "--out", type=Path, help="output folder (default: paths.audio from config)")
    ap.add_argument("--config")
    ap.add_argument("--preview", action="store_true", help="print an ASCII plot of each envelope")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = args.out or cfg.resolve(cfg.paths.audio)
    people_path = cfg.resolve(cfg.paths.people)
    people = People(people_path, cfg.houses) if people_path.exists() else None

    files: list[Path] = []
    for inp in args.inputs:
        if inp.is_dir():
            files += sorted(p for p in inp.iterdir() if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."))
        else:
            files.append(inp)
    if not files:
        sys.exit("no audio files found")

    failed = 0
    for f in files:
        try:
            process(f, out, cfg.envelope, people, args.preview)
        except Exception as e:  # noqa: BLE001 - keep going through the batch
            failed += 1
            print(f"{f.name}: FAILED - {e}", file=sys.stderr)
    print(f"\n{len(files) - failed}/{len(files)} clips written to {out}  (copy the folder's contents to the SD card root)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
