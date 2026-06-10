"""
models/text_model.py — BERT-Based Fake News Text Classifier
=============================================================
Wraps a HuggingFace text-classification pipeline (BERT / RoBERTa)
fine-tuned for fake-news detection.

Loads automatically on first call; subsequent calls use the cached model.
Falls back to zero-shot classification if the primary model is unavailable.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fake / Credibility keyword lists ────────────────────────────────────────
_FAKE_KEYWORDS = {
    "breaking", "urgent", "exclusive", "shocking", "bombshell", "exposed",
    "secret", "banned", "censored", "viral", "share before deleted",
    "must share", "wake up", "they don't want you to know", "crisis actor",
    "false flag", "deep state", "cover-up", "hoax", "nwo", "illuminati",
    "mainstream media lies", "fake news", "you won't believe",
    "doctors hate", "one weird trick", "miracle", "conspiracy",
}

_REAL_KEYWORDS = {
    "according to", "study", "research", "published", "peer-reviewed",
    "university", "scientist", "expert", "data", "evidence", "confirmed",
    "official", "government", "verified", "spokesperson", "report",
    "journal", "findings", "analysis", "investigation", "statistics",
}


class TextAnalyzer:
    """
    Wraps a HuggingFace text-classification pipeline for fake-news detection.

    Methods
    -------
    load_model()          Download / load model (called once at startup).
    analyze(text)         Run full analysis; returns a result dict.
    """

    def __init__(self):
        self._pipeline   = None
        self._mode       = None      # "bert" | "zsc"
        self.model_name  = None
        self.is_loaded   = False

    # ── Model Loading ────────────────────────────────────────────────────────
    def load_model(self) -> bool:
        """
        Try to load in order:
          1. mrm8488/bert-tiny-finetuned-fake-news-detection
          2. hamzab/roberta-fake-news-classification
          3. Zero-shot via facebook/bart-large-mnli
        Returns True if any model loaded successfully.
        """
        if self.is_loaded:
            return True

        import torch
        device = 0 if torch.cuda.is_available() else -1

        candidates = [
            ("mrm8488/bert-tiny-finetuned-fake-news-detection", "bert"),
            ("hamzab/roberta-fake-news-classification", "bert"),
        ]

        for name, mode in candidates:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "text-classification",
                    model=name,
                    device=device,
                    truncation=True,
                    max_length=512,
                )
                # smoke-test
                self._pipeline("This is a test sentence.", truncation=True)
                self.model_name = name
                self._mode = mode
                self.is_loaded = True
                return True
            except Exception:
                continue

        # Last resort: zero-shot
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device,
            )
            self.model_name = "facebook/bart-large-mnli (zero-shot)"
            self._mode = "zsc"
            self.is_loaded = True
            return True
        except Exception:
            return False

    # ── Main Inference ───────────────────────────────────────────────────────
    def analyze(self, text: str) -> dict:
        """
        Analyze text for fake news.

        Parameters
        ----------
        text : str  Raw article / paragraph to evaluate.

        Returns
        -------
        dict with keys:
            fake_score      float 0–1
            real_score      float 0–1
            label           "FAKE" | "REAL" | "SUSPICIOUS"
            confidence      float 0–1
            word_importance list[(word, score)]
            analyzed_text   str
            model_used      str
            explanation     str
            sensationalism  float 0–1
            credibility_cues list[str]
        """
        if not self.is_loaded:
            ok = self.load_model()
            if not ok:
                return self._fallback_result(text, "Model could not be loaded")

        text = text.strip()
        if len(text.split()) < 3:
            return self._fallback_result(text, "Text too short for reliable analysis")

        try:
            if self._mode == "zsc":
                raw = self._zsc_predict(text)
            else:
                raw = self._bert_predict(text)

            # Blend with keyword-based heuristics
            kw_score   = self._keyword_score(text)
            final_fake = 0.75 * raw["fake_score"] + 0.25 * kw_score

            label = _score_to_label(final_fake)

            return {
                "fake_score":      round(final_fake, 4),
                "real_score":      round(1 - final_fake, 4),
                "label":           label,
                "confidence":      round(abs(final_fake - 0.5) * 2, 4),
                "word_importance": self._word_importance(text, final_fake),
                "analyzed_text":   text[:800],
                "model_used":      self.model_name or "unknown",
                "explanation":     self._generate_explanation(text, final_fake, label),
                "sensationalism":  round(kw_score, 4),
                "credibility_cues":self._credibility_cues(text),
            }
        except Exception as exc:
            return self._fallback_result(text, str(exc))

    # ── Internal helpers ─────────────────────────────────────────────────────
    def _bert_predict(self, text: str) -> dict:
        result = self._pipeline(text[:1024], truncation=True)[0]
        lbl    = result["label"].upper()
        score  = float(result["score"])

        # Normalise label variants: LABEL_0, FAKE, 0 …
        is_fake = any(x in lbl for x in ("FAKE", "LABEL_0", "0"))
        fake_score = score if is_fake else (1 - score)
        return {"fake_score": fake_score}

    def _zsc_predict(self, text: str) -> dict:
        result = self._pipeline(
            text[:1024],
            candidate_labels=["fake news", "misinformation", "real news", "factual reporting"],
            multi_label=False,
        )
        fake_score = sum(
            s for lbl, s in zip(result["labels"], result["scores"])
            if "fake" in lbl or "misinfo" in lbl
        )
        return {"fake_score": float(fake_score)}

    def _keyword_score(self, text: str) -> float:
        """Simple 0–1 fake-likelihood from keyword matching."""
        lower = text.lower()
        fake_hits = sum(1 for kw in _FAKE_KEYWORDS if kw in lower)
        real_hits = sum(1 for kw in _REAL_KEYWORDS if kw in lower)
        total = fake_hits + real_hits + 1e-6
        return min(fake_hits / (total + 2), 0.95)

    def _credibility_cues(self, text: str) -> list[str]:
        """Return list of positive credibility signals found in the text."""
        lower = text.lower()
        return [kw for kw in _REAL_KEYWORDS if kw in lower][:8]

    def _word_importance(self, text: str, fake_score: float) -> list[tuple]:
        """
        Assign an importance score to each unique token.
        Positive = pushes toward FAKE, Negative = pushes toward REAL.
        """
        words  = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        unique = list(dict.fromkeys(words))          # preserve order, dedup

        scored = []
        for w in unique[:60]:
            if w in {kw.split()[-1] for kw in _FAKE_KEYWORDS}:
                imp = 0.7 + 0.3 * fake_score
            elif w in {kw.split()[-1] for kw in _REAL_KEYWORDS}:
                imp = -(0.5 + 0.2 * (1 - fake_score))
            else:
                freq = words.count(w) / max(len(words), 1)
                imp  = (freq * 5 - 0.1) * fake_score
            scored.append((w, round(float(imp), 4)))

        scored.sort(key=lambda x: abs(x[1]), reverse=True)
        return scored[:25]

    def _generate_explanation(self, text: str, score: float, label: str) -> str:
        lower = text.lower()
        reasons = []

        if score > 0.6:
            reasons.append("The text exhibits linguistic patterns commonly associated with misinformation.")
        if any(kw in lower for kw in ["breaking", "urgent", "exclusive", "shocking"]):
            reasons.append("Sensationalist language (e.g. 'BREAKING', 'SHOCKING') was detected.")
        if "share" in lower and "delete" in lower:
            reasons.append("Urgency-to-share manipulation tactic detected.")
        if any(kw in lower for kw in _REAL_KEYWORDS):
            reasons.append("Some credibility signals (citations, expert references) are present.")
        if len(text.split(".")) < 3:
            reasons.append("Short content with limited verifiable detail.")
        if score < 0.4:
            reasons.append("The text uses measured language typical of factual reporting.")

        return " ".join(reasons) if reasons else (
            "Analysis complete. Review individual scores for details."
        )

    def _fallback_result(self, text: str, reason: str) -> dict:
        return {
            "fake_score":      0.5,
            "real_score":      0.5,
            "label":           "UNCERTAIN",
            "confidence":      0.0,
            "word_importance": [],
            "analyzed_text":   text[:200],
            "model_used":      "fallback",
            "explanation":     f"Could not complete analysis: {reason}",
            "sensationalism":  0.0,
            "credibility_cues": [],
            "error":           reason,
        }


# ── Module-level helper ──────────────────────────────────────────────────────
def _score_to_label(fake_score: float) -> str:
    from config import AppConfig
    if fake_score >= AppConfig.THRESHOLD_FAKE:
        return "FAKE"
    if fake_score >= AppConfig.THRESHOLD_SUSPICIOUS:
        return "SUSPICIOUS"
    return "REAL"
