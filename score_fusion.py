"""
utils/score_fusion.py — Multimodal Score Fusion
================================================
Combines text, image, and video fake-probability scores into a single
credibility assessment using configurable weighted averaging.

The weights auto-scale so they always sum to 1, even when some modalities
are not present in a given run.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AppConfig


class ScoreFusion:
    """
    Fuses per-modality fake scores into one overall score.

    Usage
    -----
    fuser = ScoreFusion()
    overall = fuser.fuse(results)   # results is the dict from run_analysis()
    """

    # Base weights for each modality (re-normalised when some are absent)
    _BASE_WEIGHTS = {
        "text":  AppConfig.WEIGHT_TEXT,
        "image": AppConfig.WEIGHT_IMAGE,
        "video": AppConfig.WEIGHT_VIDEO,
    }

    def fuse(self, results: dict) -> dict:
        """
        Parameters
        ----------
        results : dict
            Keys: "text", "image", "video" (any subset).
            Each value is the result dict from the respective analyzer.

        Returns
        -------
        dict with keys:
            overall_score   float 0–1  (fake probability)
            overall_label   str        FAKE | SUSPICIOUS | REAL
            confidence      float 0–1
            modalities      list[str]  which modalities contributed
            weights_used    dict       actual weight per modality
            explanation     str
        """
        active   = {k: v for k, v in results.items()
                    if k in self._BASE_WEIGHTS and "fake_score" in v}

        if not active:
            return self._empty_result()

        # Re-normalise weights for present modalities
        raw_weights = {k: self._BASE_WEIGHTS[k] for k in active}
        total_w     = sum(raw_weights.values())
        weights     = {k: w / total_w for k, w in raw_weights.items()}

        # Weighted average
        overall = sum(
            weights[k] * active[k]["fake_score"]
            for k in active
        )
        overall = max(0.0, min(1.0, float(overall)))

        label      = _score_to_label(overall)
        confidence = abs(overall - 0.5) * 2

        return {
            "overall_score":  round(overall, 4),
            "overall_label":  label,
            "confidence":     round(confidence, 4),
            "modalities":     list(active.keys()),
            "weights_used":   {k: round(w, 3) for k, w in weights.items()},
            "explanation":    self._explain(overall, label, active, weights),
            "individual": {
                k: {
                    "fake_score": active[k]["fake_score"],
                    "label":      active[k].get("label", _score_to_label(active[k]["fake_score"])),
                }
                for k in active
            },
        }

    # ── Internal helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _explain(overall: float, label: str,
                 active: dict, weights: dict) -> str:
        parts = [f"Overall credibility assessment: {label} (score={overall:.2%})."]

        for mod, res in active.items():
            fs  = res["fake_score"]
            lbl = _score_to_label(fs)
            wt  = weights[mod]
            parts.append(
                f"  • {mod.capitalize()} analysis ({wt:.0%} weight): "
                f"{lbl} ({fs:.2%})"
            )

        if label == "FAKE":
            parts.append(
                "⚠️  Multiple modalities indicate fabrication or manipulation."
            )
        elif label == "SUSPICIOUS":
            parts.append(
                "⚠️  Some signals suggest manipulation; independent verification recommended."
            )
        else:
            parts.append(
                "✅  No strong indicators of fake content detected."
            )

        return "\n".join(parts)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "overall_score": 0.5,
            "overall_label": "UNCERTAIN",
            "confidence":    0.0,
            "modalities":    [],
            "weights_used":  {},
            "explanation":   "No analysis results to fuse.",
            "individual":    {},
        }


# ── Helpers ──────────────────────────────────────────────────────────────────
def _score_to_label(score: float) -> str:
    if score >= AppConfig.THRESHOLD_FAKE:
        return "FAKE"
    if score >= AppConfig.THRESHOLD_SUSPICIOUS:
        return "SUSPICIOUS"
    return "REAL"


def get_label_color(label: str) -> str:
    """Return hex color for a credibility label."""
    return {
        "FAKE":        AppConfig.COLOR_FAKE,
        "SUSPICIOUS":  AppConfig.COLOR_SUSPICIOUS,
        "REAL":        AppConfig.COLOR_REAL,
        "UNCERTAIN":   AppConfig.COLOR_NEUTRAL,
    }.get(label, AppConfig.COLOR_NEUTRAL)


def get_label_emoji(label: str) -> str:
    return {
        "FAKE":       "🔴",
        "SUSPICIOUS": "🟡",
        "REAL":       "🟢",
        "UNCERTAIN":  "⚪",
    }.get(label, "⚪")
