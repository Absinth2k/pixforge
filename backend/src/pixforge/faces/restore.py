"""Face restoration: align to the FFHQ template, restore at 512 px, paste back softly."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np

from pixforge.backends.base import InferenceBackend
from pixforge.faces.blend import face_mask, paste
from pixforge.faces.detect import Face

FACE_SIZE = 512
# Five-point FFHQ alignment template at 512 px (as used by facexlib / GFPGAN / RestoreFormer).
FFHQ_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.1936],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    np.float32,
)


def align_matrix(landmarks: np.ndarray) -> np.ndarray:
    m, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), FFHQ_512, method=cv2.LMEDS)
    if m is None:
        raise ValueError("face alignment failed")
    return m.astype(np.float32)


def crop_size_in_image(m: np.ndarray) -> float:
    """Side length (image px) of the 512 template square."""
    scale = float(np.sqrt(abs(np.linalg.det(m[:, :2]))))
    return FACE_SIZE / max(scale, 1e-6)


def size_factor(crop_px: float) -> float:
    """Faces that are already larger than the restorer's 512 px would get softer, not sharper:
    full effect up to 512 px, fading out to none at 1024 px."""
    return float(np.clip((1024.0 - crop_px) / 512.0, 0.0, 1.0))


class FaceRestorer:
    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend
        self._mask = face_mask(FACE_SIZE)

    def restore(
        self,
        img: np.ndarray,
        faces: Sequence[Face],
        strength: float = 80.0,
        exclude: Sequence[int] = (),
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Restore ``faces`` (landmarks in ``img`` coordinates). strength: 0..100."""
        skip = set(exclude)
        jobs = []
        # Crop every face first: pasting modifies ``img`` in place, and neighbouring faces
        # must be restored from the untouched pixels.
        for i, face in enumerate(faces):
            if i in skip:
                continue
            m = align_matrix(face.landmarks)
            crop_px = crop_size_in_image(m)
            eff = strength / 100.0 * size_factor(crop_px)
            if eff <= 0.01:
                continue
            interp = cv2.INTER_AREA if crop_px > FACE_SIZE * 1.5 else cv2.INTER_LINEAR
            crop = cv2.warpAffine(
                img, m, (FACE_SIZE, FACE_SIZE), flags=interp, borderMode=cv2.BORDER_REFLECT_101
            )
            jobs.append((m, eff, np.ascontiguousarray(crop)))
        out = img
        for n, (m, eff, crop) in enumerate(jobs, 1):
            restored = np.clip(self.backend.run_tile(crop), 0.0, 1.0)
            out = paste(out, restored, self._mask, m, eff)
            if progress:
                progress(n, len(jobs))
        return out
