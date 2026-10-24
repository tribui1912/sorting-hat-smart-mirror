"""Enroll friends from the mirror camera itself (same lens, angle and light as the party).

    python mirror/enroll.py capture --name Alice --house gryffindor     # live preview window
    python mirror/enroll.py capture --name Alice --headless             # over SSH: Enter = capture
    python mirror/enroll.py folder photos/alice --name Alice --house gryffindor   # fallback: existing photos
    python mirror/enroll.py rebuild                                     # re-embed everything (after a model change)
    python mirror/enroll.py list                                        # quality report for everyone
    python mirror/enroll.py remove --name Alice
    python mirror/enroll.py snap data/strangers                         # save raw frames (for calibrate.py)

Capture keys:  SPACE capture  BACKSPACE undo last  ENTER save & finish  ESC abort
Get 5-10 shots per person standing on the mark: look at the lens, then slightly left/right/up/down,
neutral face and a smile. Glasses on if they'll wear them at the party.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from common import Config, People, load_config, setup_logging
from faces import Face, FaceEngine, Gallery

log = logging.getLogger("sortinghat.enroll")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------- quality report


def report(cfg: Config, gallery: Gallery, names: list[str] | None = None) -> int:
    """Print per-person consistency and lookalike warnings. Returns the number of warnings."""
    ec = cfg.enroll
    warnings = 0
    for name in names or gallery.names:
        if name not in gallery.names:
            continue
        sims = gallery.self_consistency(name)
        n = len(sims)
        flag = ""
        if sims.mean() < ec.warn_mean_sim:
            flag = f"  !! HIGH VARIANCE (mean {sims.mean():.2f} < {ec.warn_mean_sim}): recapture under party lighting"
            warnings += 1
        elif n < 5:
            flag = "  ! fewer than 5 shots"
            warnings += 1
        print(f"  {name:20s} {n:2d} shots  self-sim mean {sims.mean():.2f}  min {sims.min():.2f}{flag}")
        for i, s in enumerate(sims):
            if s < ec.warn_shot_sim:
                print(f"      ! shot {i} looks unlike the others ({s:.2f}): blurry, turned, or the wrong person?")
                warnings += 1
    # lookalikes: centroids too close together
    if len(gallery.names) > 1:
        sim = gallery.centroids @ gallery.centroids.T
        np.fill_diagonal(sim, -1)
        idx = np.argwhere(np.triu(sim > ec.warn_cross_sim))
        for i, j in idx:
            print(f"  ! {gallery.names[i]} and {gallery.names[j]} look alike ({sim[i, j]:.2f}): "
                  f"recognition.margin protects them; consider more shots of both")
            warnings += 1
        nearest = sim.max(axis=1)
        print(f"  closest pair similarity: {nearest.max():.2f}  (threshold {cfg.recognition.threshold})")
    return warnings


# ---------------------------------------------------------------------------- helpers


def check_face(cfg: Config, face: Face | None) -> str | None:
    if face is None:
        return "no face found"
    if face.height < cfg.recognition.min_face_px:
        return f"face too small ({face.height:.0f}px < {cfg.recognition.min_face_px}) - step closer"
    if face.score < cfg.recognition.det_threshold:
        return f"low detection confidence ({face.score:.2f})"
    return None


def save_person(cfg: Config, gallery: Gallery, people: People, name: str, house: str | None,
                embs: list[np.ndarray], frames: list[np.ndarray] | None, append: bool) -> None:
    import cv2

    img_dir = cfg.resolve(cfg.paths.enroll_images) / name
    if append and name in gallery.samples:
        all_embs = np.concatenate([gallery.samples[name], np.stack(embs)])
    else:
        all_embs = np.stack(embs)
        if img_dir.exists() and frames is not None:  # replacing: archive the old shots, don't delete them
            archive = img_dir.parent / ".old" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img_dir), str(archive))
    if frames is not None:
        img_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for i, f in enumerate(frames):
            cv2.imwrite(str(img_dir / f"{stamp}-{i:02d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
    person = people.upsert(name, house)
    gallery.set(name, all_embs)
    gallery.save(cfg.resolve(cfg.paths.embeddings))
    people.save()
    print(f"\nsaved {name} (id {person.id}, {person.house}) with {len(all_embs)} embeddings")
    report(cfg, gallery, [name])
    others = [n for n in gallery.names if n != name]
    if others:
        i = gallery.names.index(name)
        sims = gallery.centroids @ gallery.centroids[i]
        nearest = max(others, key=lambda n: sims[gallery.names.index(n)])
        print(f"  nearest other person: {nearest} ({sims[gallery.names.index(nearest)]:.2f})")


def embed_images(engine: FaceEngine, cfg: Config, paths: list[Path]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    import cv2

    embs, frames = [], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  {p.name}: unreadable, skipped")
            continue
        face = engine.largest(img)
        # no min-size check here: arbitrary photos have arbitrary resolutions
        if face is None or face.score < 0.5:
            print(f"  {p.name}: {'no face found' if face is None else 'low confidence'}, skipped")
            continue
        embs.append(engine.embed(img, face))
        frames.append(img)
        print(f"  {p.name}: ok ({face.height:.0f}px)")
    return embs, frames


# ---------------------------------------------------------------------------- capture modes


class Preview:
    """pygame window showing the camera (mirrored, like a mirror) with the detected face box."""

    def __init__(self, fullscreen: bool):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import pygame

        self.pg = pygame
        pygame.display.init()
        pygame.font.init()
        self.fullscreen = fullscreen
        self.screen: pygame.Surface | None = None
        self.font = pygame.font.SysFont("dejavusans,helvetica,arial", 26)

    def show(self, frame: np.ndarray, face: Face | None, lines: list[str], ok: bool) -> None:
        pg = self.pg
        h, w = frame.shape[:2]
        if self.screen is None:
            if self.fullscreen:
                self.screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
            else:
                k = min(1.0, 900 / max(w, h))
                self.screen = pg.display.set_mode((int(w * k), int(h * k)))
            pg.display.set_caption("enroll.py")
        sc = self.screen
        sw, sh = sc.get_size()
        k = min(sw / w, sh / h)
        dw, dh = int(w * k), int(h * k)
        ox, oy = (sw - dw) // 2, (sh - dh) // 2
        mirrored = frame[:, ::-1, ::-1]  # flip horizontally + BGR->RGB
        surf = pg.surfarray.make_surface(np.ascontiguousarray(mirrored.swapaxes(0, 1)))
        sc.fill((0, 0, 0))
        sc.blit(pg.transform.scale(surf, (dw, dh)), (ox, oy))
        if face is not None:
            x1, y1, x2, y2 = face.bbox
            x1, x2 = w - x2, w - x1  # mirrored
            pg.draw.rect(sc, (80, 220, 120) if ok else (240, 90, 80),
                         (ox + x1 * k, oy + y1 * k, (x2 - x1) * k, (y2 - y1) * k), 3)
        for i, line in enumerate(lines):
            img = self.font.render(line, True, (255, 255, 255), (0, 0, 0))
            sc.blit(img, (12, 12 + i * 32))
        pg.display.flip()

    def keys(self) -> list[int]:
        pg = self.pg
        out = []
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                out.append(pg.K_ESCAPE)
            elif ev.type == pg.KEYDOWN:
                out.append(ev.key)
        return out

    def close(self) -> None:
        self.pg.quit()


def capture_interactive(cfg: Config, engine: FaceEngine, name: str, target: int, fullscreen: bool
                        ) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    from camera import Camera

    preview = Preview(fullscreen)
    pg = preview.pg
    embs: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    msg, msg_until = "", 0.0
    try:
        with Camera(cfg.camera) as cam:
            while True:
                frame = cam.read()
                face = engine.largest(frame)
                problem = check_face(cfg, face)
                status = problem or f"face {face.height:.0f}px  yaw {face.yaw:.2f}"  # type: ignore[union-attr]
                lines = [f"{name}: {len(embs)}/{target} shots", status,
                         "SPACE capture  BKSP undo  ENTER save  ESC abort"]
                if time.monotonic() < msg_until:
                    lines.append(msg)
                preview.show(frame, face, lines, problem is None)
                for key in preview.keys():
                    if key == pg.K_ESCAPE:
                        print("aborted, nothing saved")
                        return None
                    if key in (pg.K_RETURN, pg.K_KP_ENTER):
                        if not embs:
                            msg, msg_until = "no shots yet", time.monotonic() + 2
                            continue
                        return embs, frames
                    if key == pg.K_BACKSPACE and embs:
                        embs.pop()
                        frames.pop()
                        msg, msg_until = "removed last shot", time.monotonic() + 2
                    if key == pg.K_SPACE:
                        if problem:
                            msg, msg_until = f"not captured: {problem}", time.monotonic() + 2
                            continue
                        assert face is not None
                        e = engine.embed(frame, face)
                        note = ""
                        if embs:
                            c = np.mean(embs, axis=0)
                            sim = float(e @ (c / np.linalg.norm(c)))
                            note = f"  similarity to previous shots {sim:.2f}"
                            if sim < cfg.enroll.warn_shot_sim:
                                note += "  !! unlike the others - undo?"
                        embs.append(e)
                        frames.append(frame.copy())
                        msg, msg_until = f"captured #{len(embs)}{note}", time.monotonic() + 3
                        print(msg)
    finally:
        preview.close()


def capture_headless(cfg: Config, engine: FaceEngine, name: str, target: int
                     ) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    from camera import Camera

    embs: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    print(f"Enrolling {name}. Enter = capture, u = undo, s = save, q = abort.")
    with Camera(cfg.camera) as cam:
        for _ in range(5):
            cam.read()  # let exposure settle
        while True:
            cmd = input(f"[{len(embs)}/{target}] > ").strip().lower()
            if cmd == "q":
                print("aborted, nothing saved")
                return None
            if cmd == "s":
                return (embs, frames) if embs else None
            if cmd == "u":
                if embs:
                    embs.pop()
                    frames.pop()
                continue
            for _ in range(3):
                frame = cam.read()  # drain buffered frames so we get "now"
            face = engine.largest(frame)
            problem = check_face(cfg, face)
            if problem:
                print(f"  not captured: {problem}")
                continue
            assert face is not None
            e = engine.embed(frame, face)
            if embs:
                c = np.mean(embs, axis=0)
                print(f"  ok ({face.height:.0f}px, similarity to previous {float(e @ (c / np.linalg.norm(c))):.2f})")
            else:
                print(f"  ok ({face.height:.0f}px)")
            embs.append(e)
            frames.append(frame.copy())
            if len(embs) >= target:
                print("  target reached: 's' to save or keep capturing")


def snap(cfg: Config, out: Path, headless: bool) -> None:
    """Save raw camera frames (for strangers / held-out genuine sets used by calibrate.py)."""
    import cv2

    from camera import Camera

    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with Camera(cfg.camera) as cam:
        if headless:
            print(f"Saving to {out}. Enter = save frame, q = quit.")
            while input("> ").strip().lower() != "q":
                for _ in range(3):
                    frame = cam.read()
                p = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{n:03d}.jpg"
                cv2.imwrite(str(p), frame)
                n += 1
                print(f"  {p.name}")
            return
        preview = Preview(False)
        pg = preview.pg
        try:
            while True:
                frame = cam.read()
                preview.show(frame, None, [f"{out}: {n} saved", "SPACE save  ESC quit"], True)
                for key in preview.keys():
                    if key == pg.K_ESCAPE:
                        return
                    if key == pg.K_SPACE:
                        cv2.imwrite(str(out / f"{time.strftime('%Y%m%d-%H%M%S')}-{n:03d}.jpg"), frame)
                        n += 1
        finally:
            preview.close()


# ---------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("capture", help="enroll from the mirror camera")
    c.add_argument("--name", required=True)
    c.add_argument("--house", help="required for a new person")
    c.add_argument("--shots", type=int)
    c.add_argument("--append", action="store_true", help="add to existing shots instead of replacing them")
    c.add_argument("--headless", action="store_true", help="terminal only (over SSH)")
    c.add_argument("--fullscreen", action="store_true", help="preview fullscreen on the mirror")

    f = sub.add_parser("folder", help="enroll from existing photos (fallback)")
    f.add_argument("path", type=Path)
    f.add_argument("--name", required=True)
    f.add_argument("--house")
    f.add_argument("--append", action="store_true")
    f.add_argument("--det-size", type=int, default=640, help="detector size for arbitrary photos")

    sub.add_parser("rebuild", help="re-embed every saved enrollment image (after changing model)")
    sub.add_parser("list", help="quality report")
    r = sub.add_parser("remove")
    r.add_argument("--name", required=True)
    s = sub.add_parser("snap", help="save raw frames to a folder")
    s.add_argument("path", type=Path)
    s.add_argument("--headless", action="store_true")

    args = ap.parse_args()
    cfg = load_config(args.config)
    setup_logging(name="enroll")
    people = People(cfg.resolve(cfg.paths.people), cfg.houses)
    gallery_path = cfg.resolve(cfg.paths.embeddings)
    gallery = Gallery.load(gallery_path)

    if args.cmd == "list":
        for p in people.all():
            mark = "" if p.name in gallery.names else "   (NOT ENROLLED - override only)"
            print(f"  #{p.id:<3d} {p.name:20s} {p.house}{mark}")
        print()
        n = report(cfg, gallery)
        print(f"\n{len(gallery.names)} enrolled, {n} warning(s)")
        return
    if args.cmd == "remove":
        gallery.remove(args.name)
        gallery.save(gallery_path)
        people.remove(args.name)
        people.save()
        print(f"removed {args.name} (enrollment images left in {cfg.paths.enroll_images}/{args.name})")
        return
    if args.cmd == "snap":
        snap(cfg, args.path, args.headless)
        return

    if args.cmd in ("capture", "folder") and args.house is None and people.get(args.name) is None:
        sys.exit(f"{args.name} is new: pass --house ({', '.join(cfg.houses)})")
    if args.cmd in ("capture", "folder") and args.house and args.house not in cfg.houses:
        sys.exit(f"unknown house {args.house}: choose from {', '.join(cfg.houses)}")

    if args.cmd == "capture":
        engine = FaceEngine(cfg.recognition)
        target = args.shots or cfg.enroll.shots
        if args.headless:
            result = capture_headless(cfg, engine, args.name, target)
        else:
            result = capture_interactive(cfg, engine, args.name, target, args.fullscreen)
        if result:
            save_person(cfg, gallery, people, args.name, args.house, result[0], result[1], args.append)
        return

    if args.cmd == "folder":
        engine = FaceEngine(cfg.recognition, det_size=(args.det_size, args.det_size))
        paths = sorted(p for p in args.path.iterdir() if p.suffix.lower() in IMG_EXTS)
        print(f"{len(paths)} images in {args.path}")
        embs, frames = embed_images(engine, cfg, paths)
        if not embs:
            sys.exit("no usable faces")
        print("note: photos from other cameras match less well than mirror-camera captures")
        save_person(cfg, gallery, people, args.name, args.house, embs, frames, args.append)
        return

    if args.cmd == "rebuild":
        engine = FaceEngine(cfg.recognition, det_size=(640, 640))
        root = cfg.resolve(cfg.paths.enroll_images)
        new = Gallery()
        for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            print(d.name)
            embs, _ = embed_images(engine, cfg, sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS))
            if embs:
                new.set(d.name, np.stack(embs))
            if people.get(d.name) is None:
                print(f"  ! {d.name} is not in people.json: add them with a house or they won't be sorted")
        new.save(gallery_path)
        print()
        report(cfg, new)


if __name__ == "__main__":
    main()
