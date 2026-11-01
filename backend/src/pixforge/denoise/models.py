"""Denoising decisions: when, how strong, with which model.

Calibrated on 5 unprocessed Canon RAWs and 3 DxO PureRAW (DeepPRIME 3) outputs, see
docs/backend.md section 4.
"""

from __future__ import annotations

from dataclasses import dataclass

from pixforge.denoise.estimate import NoiseEstimate

#: below this luma sigma (8-bit units) an image counts as clean
CLEAN_LUMA = 1.5
#: an AI- or phone-denoised image has almost no colour noise left, even if grain was kept.
#: Measured at half resolution: unprocessed RAWs 1.8-8.9 (ratio >= 1.7), DxO 0.07-0.33
#: (ratio <= 0.43), Pixel 8 Pro JPEGs 0.0.
DENOISED_CHROMA = 0.6
DENOISED_RATIO = 0.6
#: luma noise (8-bit sigma) that may remain after mixing; more than ~1.5 is amplified into
#: worm-like texture by the upscaler (measured, docs/backend.md section 4)
RESIDUAL_TARGET = 0.8
#: models that remove noise themselves (Real-HAT-GAN, trained on real degradations)
SELF_DENOISING = ("photo",)


@dataclass(frozen=True)
class DenoiseSuggestion:
    strength: float  # 0..100, start value of the slider
    reason: str  # "already_denoised" | "clean" | "noisy"
    estimate: NoiseEstimate

    def message(self, lang: str = "de") -> str:
        de = {
            "already_denoised": "Bild ist bereits entrauscht",
            "clean": "kein Rauschen erkannt",
            "model_handles": "Rauschen erkannt, das Fotomodell entfernt es selbst",
            "noisy": "Rauschen erkannt",
        }
        en = {
            "already_denoised": "Image is already denoised",
            "clean": "no noise detected",
            "model_handles": "Noise detected, the photo model removes it itself",
            "noisy": "noise detected",
        }
        return (de if lang == "de" else en)[self.reason]


def suggest(
    est: NoiseEstimate, meta: dict[str, object] | None = None, upscale_model: str = "photo"
) -> DenoiseSuggestion:
    """Start value for the noise slider.

    ``upscale_model``: the Real-HAT-GAN models ("photo", "photo-sharp") remove even strong
    sensor noise themselves; a partial pre-denoise only leaves residual noise that the GAN
    turns into worms. The automatic stays off for them (the user can still enable it).

    Otherwise the strength is chosen so that the residual luma noise after mixing,
    (1 - strength) * sigma, stays at about RESIDUAL_TARGET.
    """
    meta = meta or {}
    ratio = est.chroma_sigma / max(est.luma_sigma, 1e-6)
    # Black-and-white images never have colour noise, so that signal means nothing there.
    if not est.is_grayscale and est.chroma_sigma < DENOISED_CHROMA and ratio < DENOISED_RATIO:
        return DenoiseSuggestion(0.0, "already_denoised", est)
    if est.luma_sigma < CLEAN_LUMA:
        return DenoiseSuggestion(0.0, "clean", est)

    if upscale_model.startswith(SELF_DENOISING):
        return DenoiseSuggestion(0.0, "model_handles", est)
    strength = (1.0 - RESIDUAL_TARGET / est.luma_sigma) * 100.0
    iso = meta.get("iso")
    if isinstance(iso, int) and iso >= 3200:
        strength += 5.0
    if meta.get("is_phone"):
        strength += 5.0  # phone noise is blotchy and survives upscaling badly
    return DenoiseSuggestion(float(min(100.0, max(0.0, strength))), "noisy", est)
