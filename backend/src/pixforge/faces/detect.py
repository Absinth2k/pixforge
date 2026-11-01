"""Face detection with YuNet (OpenCV Zoo, MIT) through OpenCV's FaceDetectorYN.

YuNet is trained on small to medium faces: a portrait-filling face is missed at full
resolution, while tiny faces in a group shot are missed at low resolution. Pixforge therefore
detects at two scales and merges the results with non-maximum suppression.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# silence "Targets are not supported by the new graph engine" warnings
with contextlib.suppress(AttributeError):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

DETECT_SIDES = (640, 1600)
# YuNet's own demo uses 0.9. Real faces in old group photos scored 0.86-0.92 in our tests,
# a window niche 0.72; restoring a false positive paints a face into the background.
SCORE_THRESHOLD = 0.8
NMS_IOU = 0.4


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]  # x, y, w, h in image pixels
    landmarks: np.ndarray  # (5, 2): eye (image left), eye (image right), nose, mouth l, mouth r
    score: float

    @property
    def size(self) -> float:
        return max(self.box[2], self.box[3])

    @property
    def eye_distance(self) -> float:
        return float(np.linalg.norm(self.landmarks[1] - self.landmarks[0]))

    def scaled(self, f: float) -> Face:
        x, y, w, h = self.box
        return Face((x * f, y * f, w * f, h * f), self.landmarks * f, self.score)

    def to_dict(self) -> dict[str, object]:
        return {
            "box": [round(v, 1) for v in self.box],
            "landmarks": np.round(self.landmarks, 1).tolist(),
            "score": round(self.score, 3),
        }


class FaceDetector:
    def __init__(self, model_path: str | Path | None = None) -> None:
        if model_path is None:
            from pixforge.models.registry import model_path as resolve

            model_path = resolve("face-detect", "fp32")
        self.model_path = str(model_path)

    def detect(self, rgb: np.ndarray, min_size: float = 0.0) -> list[Face]:
        """Faces in ``rgb`` (float 0..1 or uint8), largest first."""
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        rows: list[np.ndarray] = []
        done_native = False
        for side in DETECT_SIDES:
            s = side / max(h, w)
            if s >= 1.0:
                if done_native:
                    continue
                s, done_native = 1.0, True
            img = (
                bgr
                if s == 1.0
                else cv2.resize(
                    bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA
                )
            )
            det = cv2.FaceDetectorYN.create(
                self.model_path, "", (img.shape[1], img.shape[0]), SCORE_THRESHOLD, 0.3, 5000
            )
            _, found = det.detect(img)
            if found is not None:
                found = found.copy()
                found[:, :14] /= s
                rows.append(found)
        if not rows:
            return []
        allf = np.concatenate(rows)
        keep = cv2.dnn.NMSBoxes(
            allf[:, :4].tolist(), allf[:, 14].tolist(), SCORE_THRESHOLD, NMS_IOU
        )
        faces = []
        for i in np.array(keep).reshape(-1):
            r = allf[int(i)]
            lm = r[4:14].reshape(5, 2).astype(np.float32)
            lm = _order_landmarks(lm)
            face = Face((float(r[0]), float(r[1]), float(r[2]), float(r[3])), lm, float(r[14]))
            if face.size >= min_size and plausible(face):
                faces.append(face)
        faces.sort(key=lambda f: f.size, reverse=True)
        return faces


def plausible(face: Face) -> bool:
    """Reject detections whose landmarks do not look like an (upright or tilted) face."""
    lm = face.landmarks
    eyes, nose, mouth = lm[:2].mean(0), lm[2], lm[3:].mean(0)
    axis = mouth - eyes  # eyes -> mouth direction
    length = float(np.linalg.norm(axis))
    if length < 1e-3:
        return False
    # nose must lie between eyes and mouth along that axis
    t = float(np.dot(nose - eyes, axis) / length**2)
    if not 0.2 <= t <= 0.95:
        return False
    ratio = face.eye_distance / max(face.box[2], 1e-3)
    return 0.2 <= ratio <= 0.8


def _order_landmarks(lm: np.ndarray) -> np.ndarray:
    """Eyes and mouth corners ordered left-to-right in the image (FFHQ template order)."""
    out = lm.copy()
    if out[0, 0] > out[1, 0]:
        out[[0, 1]] = out[[1, 0]]
    if out[3, 0] > out[4, 0]:
        out[[3, 4]] = out[[4, 3]]
    return out
