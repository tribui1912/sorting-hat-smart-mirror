"""Camera source: picamera2 on the Pi, OpenCV webcam as a dev fallback on a laptop.

Frames come out as upright BGR uint8 arrays (what InsightFace expects).
"""

from __future__ import annotations

import logging

import numpy as np

from common import CameraCfg

log = logging.getLogger("sortinghat.camera")


class Camera:
    def __init__(self, cfg: CameraCfg):
        self.cfg = cfg
        self._picam = None
        self._cv = None
        try:
            self._open_picamera2()
        except ImportError:
            log.warning("picamera2 not available, falling back to OpenCV webcam %d (dev mode)", cfg.webcam_index)
            self._open_webcam()

    def _open_picamera2(self) -> None:
        from libcamera import Transform, controls  # type: ignore[import-not-found]
        from picamera2 import Picamera2  # type: ignore[import-not-found]

        cam = Picamera2()
        w, h = self.cfg.size
        # "RGB888" in picamera2 is BGR byte order in numpy, which is what OpenCV/InsightFace want.
        conf = cam.create_video_configuration(
            main={"size": (w, h), "format": "RGB888"},
            transform=Transform(hflip=self.cfg.hflip, vflip=self.cfg.vflip),
            buffer_count=3,
        )
        cam.configure(conf)
        cam.start()
        if self.cfg.lens_position is None:
            cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        else:
            cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(self.cfg.lens_position)})
        self._picam = cam
        log.info("picamera2 started at %dx%d", w, h)

    def _open_webcam(self) -> None:
        import cv2

        cap = cv2.VideoCapture(self.cfg.webcam_index)
        if not cap.isOpened():
            raise RuntimeError(f"no camera: picamera2 missing and webcam {self.cfg.webcam_index} won't open")
        w, h = self.cfg.size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cv = cap

    def read(self) -> np.ndarray:
        if self._picam is not None:
            frame = self._picam.capture_array("main")
        else:
            ok, frame = self._cv.read()  # type: ignore[union-attr]
            if not ok:
                raise RuntimeError("webcam read failed")
        if frame.shape[2] == 4:  # some picamera2 formats carry an alpha/padding channel
            frame = frame[:, :, :3]
        rot = self.cfg.rotate % 360
        if rot:
            # np.rot90 is counter-clockwise; config rotate is clockwise
            frame = np.rot90(frame, k=-(rot // 90))
        return np.ascontiguousarray(frame)

    def close(self) -> None:
        try:
            if self._picam is not None:
                self._picam.stop()
                self._picam.close()
            if self._cv is not None:
                self._cv.release()
        except Exception:  # noqa: BLE001
            log.exception("camera close failed")

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
