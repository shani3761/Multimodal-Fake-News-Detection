"""
inference.py — FakeShield AI Unified Inference Orchestrator
============================================================
Single entry-point for all prediction tasks.  Models are loaded lazily
and cached as singletons so each process loads them at most once.

Quick usage:
    engine = FakeNewsInference.get_instance()
    result = engine.predict(text="…", image=some_pil, video_url="…")

CLI:
    python inference.py --text "Breaking news …"
    python inference.py --text "…" --image photo.jpg --video https://youtu.be/xxx
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ── Ensure project root on path ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import AppConfig
from utils.score_fusion import ScoreFusion
from utils.text_utils import clean_text, fetch_url_content, is_valid_url

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("inference")


# ════════════════════════════════════════════════════════════════════════════
# Inference engine
# ════════════════════════════════════════════════════════════════════════════

class FakeNewsInference:
    """
    Orchestrates text, image and video models, fuses their scores, and
    returns a rich result dictionary.

    Design choices
    --------------
    * Singleton pattern — call ``FakeNewsInference.get_instance()`` to avoid
      re-loading models across multiple calls in the same process.
    * Models load lazily the first time they are needed.
    * ``predict()`` accepts any combination of text / image / video_url.
    * ``predict_batch()`` accepts a list of input dicts for bulk scoring.
    """

    _instance: Optional["FakeNewsInference"] = None

    # ── Constructor ──────────────────────────────────────────────────────────
    def __init__(self) -> None:
        self._text_analyzer:  Any = None
        self._image_analyzer: Any = None
        self._video_analyzer: Any = None
        self._score_fusion        = ScoreFusion()
        self._db:             Any = None
        self._load_times:     dict = {}
        AppConfig.ensure_dirs()

    # ── Singleton factory ────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "FakeNewsInference":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Model accessors (lazy) ───────────────────────────────────────────────
    def _text(self):
        if self._text_analyzer is None:
            from models.text_model import TextAnalyzer
            logger.info("Loading text model …")
            t0 = time.time()
            self._text_analyzer = TextAnalyzer()
            self._text_analyzer.load_model()
            self._load_times["text"] = round(time.time() - t0, 2)
            logger.info("Text model ready in %.1fs", self._load_times["text"])
        return self._text_analyzer

    def _image(self):
        if self._image_analyzer is None:
            from models.image_model import ImageAnalyzer
            logger.info("Loading image model …")
            t0 = time.time()
            self._image_analyzer = ImageAnalyzer()
            self._image_analyzer.load_model()
            self._load_times["image"] = round(time.time() - t0, 2)
            logger.info("Image model ready in %.1fs", self._load_times["image"])
        return self._image_analyzer

    def _video(self):
        if self._video_analyzer is None:
            from models.video_model import VideoAnalyzer
            logger.info("Loading video model …")
            self._video_analyzer = VideoAnalyzer()
            self._load_times["video"] = 0.0
        return self._video_analyzer

    def _db_conn(self):
        if self._db is None:
            from database import AnalysisDatabase
            self._db = AnalysisDatabase()
        return self._db

    # ── Core prediction ──────────────────────────────────────────────────────
    def predict(
        self,
        text:       Optional[str]                        = None,
        image:      Optional[Any]                        = None,   # PIL.Image or path
        video_url:  Optional[str]                        = None,
        save_to_db: bool                                 = True,
        analysis_type: Optional[str]                     = None,
    ) -> Dict[str, Any]:
        """
        Run multimodal fake-news analysis.

        Parameters
        ----------
        text       : article text, or a URL (will be scraped automatically).
        image      : PIL.Image or str path to an image file, or None.
        video_url  : URL to a video (YouTube, direct .mp4, …), or None.
        save_to_db : persist the result to SQLite.
        analysis_type : override the DB label; inferred if None.

        Returns
        -------
        dict with keys:
          final_score    float 0–1  (fake probability — top-level alias)
          verdict        str        FAKE | SUSPICIOUS | REAL
          confidence     float 0–1
          individual     dict       per-modality scores
          explanations   dict       per-modality explanations
          modalities     list[str]  which modalities ran
          fusion         dict       full fusion output
          meta           dict       timing, model names, etc.
        """
        t_start  = time.perf_counter()
        results  = {}
        errors   = {}

        # ── Resolve image ────────────────────────────────────────────────────
        pil_image = _resolve_image(image)

        # ── Resolve text (URL → body) ─────────────────────────────────────────
        article_text = _resolve_text(text)

        if not any([article_text, pil_image, video_url]):
            return _empty_result("No inputs provided.")

        # ── Text analysis ────────────────────────────────────────────────────
        if article_text:
            try:
                results["text"] = self._text().analyze(article_text)
            except Exception as exc:
                logger.error("Text analysis failed: %s", exc)
                errors["text"] = str(exc)

        # ── Image analysis ───────────────────────────────────────────────────
        if pil_image is not None:
            try:
                results["image"] = self._image().analyze(pil_image)
            except Exception as exc:
                logger.error("Image analysis failed: %s", exc)
                errors["image"] = str(exc)

        # ── Video analysis ───────────────────────────────────────────────────
        if video_url:
            try:
                results["video"] = self._video().analyze(video_url)
            except Exception as exc:
                logger.error("Video analysis failed: %s", exc)
                errors["video"] = str(exc)

        if not results:
            return _empty_result("All modality analyses failed.", errors)

        # ── Score fusion ──────────────────────────────────────────────────────
        fusion = self._score_fusion.fuse(results)

        # ── Assemble rich output ─────────────────────────────────────────────
        elapsed  = round(time.perf_counter() - t_start, 3)
        a_type   = analysis_type or _infer_type(results)
        verdict  = fusion["overall_label"]
        f_score  = fusion["overall_score"]
        conf     = fusion["confidence"]

        output = {
            "final_score":  f_score,
            "verdict":      verdict,
            "confidence":   conf,
            "individual":   {
                mod: {
                    "fake_score":  r.get("fake_score",  0.5),
                    "label":       r.get("label",       "UNCERTAIN"),
                    "confidence":  r.get("confidence",  0.0),
                }
                for mod, r in results.items()
            },
            "explanations": {
                mod: r.get("explanation", "") or r.get("explanation_text", "")
                for mod, r in results.items()
            },
            "modalities":   list(results.keys()),
            "fusion":       fusion,
            "errors":       errors,
            "meta": {
                "elapsed_sec":   elapsed,
                "analysis_type": a_type,
                "model_versions": {
                    "text":  getattr(self._text_analyzer, "model_name", "N/A")
                             if self._text_analyzer else "not loaded",
                    "image": AppConfig.IMAGE_MODEL_ARCH,
                    "video": "VideoAnalyzer v1.0",
                },
                "load_times": self._load_times,
                "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            # Raw per-modality results (may include PIL images — handle carefully)
            "_raw": results,
        }

        # ── Persist ───────────────────────────────────────────────────────────
        if save_to_db:
            try:
                safe_details = _sanitise_for_db(results, fusion)
                row_id = self._db_conn().save_analysis({
                    "analysis_type":   a_type,
                    "input_summary":   (article_text or "")[:300],
                    "input_video_url": video_url or "",
                    "overall_score":   f_score,
                    "overall_label":   verdict,
                    "text_score":      results.get("text",  {}).get("fake_score"),
                    "image_score":     results.get("image", {}).get("fake_score"),
                    "video_score":     results.get("video", {}).get("fake_score"),
                    "confidence":      conf,
                    "explanation_text":fusion.get("explanation", ""),
                    "details":         safe_details,
                })
                output["db_record_id"] = row_id
            except Exception as exc:
                logger.warning("DB save failed: %s", exc)

        return output

    # ── Batch inference ──────────────────────────────────────────────────────
    def predict_batch(
        self,
        items: List[Dict[str, Any]],
        save_to_db: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run ``predict()`` on a list of input dicts.

        Each dict may contain: text, image, video_url.

        Parameters
        ----------
        items      : list of dicts, e.g. [{"text": "…"}, {"image": img, …}]
        save_to_db : passed through to each ``predict()`` call

        Returns
        -------
        list of result dicts in the same order as *items*.
        """
        results = []
        for i, item in enumerate(items):
            logger.info("Batch item %d/%d", i + 1, len(items))
            try:
                r = self.predict(
                    text      = item.get("text"),
                    image     = item.get("image"),
                    video_url = item.get("video_url"),
                    save_to_db= save_to_db,
                )
            except Exception as exc:
                r = _empty_result(f"Item {i} failed: {exc}")
            results.append(r)
        return results

    # ── Pre-warm ─────────────────────────────────────────────────────────────
    def warm_up(self, modalities: tuple = ("text", "image")) -> None:
        """Pre-load selected models so the first predict() call is fast."""
        if "text"  in modalities:
            self._text()
        if "image" in modalities:
            self._image()
        if "video" in modalities:
            self._video()


# ════════════════════════════════════════════════════════════════════════════
# Private helpers
# ════════════════════════════════════════════════════════════════════════════

def _resolve_text(text: Optional[str]) -> Optional[str]:
    if not text or not text.strip():
        return None
    text = text.strip()
    if is_valid_url(text):
        logger.info("Fetching article from URL: %s", text)
        _, body = fetch_url_content(text)
        return clean_text(body) if body else None
    return clean_text(text)


def _resolve_image(image: Optional[Any]) -> Optional[Any]:
    if image is None:
        return None
    from PIL import Image as PILImage
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)) and Path(image).is_file():
        try:
            return PILImage.open(str(image)).convert("RGB")
        except Exception as exc:
            logger.warning("Could not open image %s: %s", image, exc)
    return None


def _infer_type(results: dict) -> str:
    mods = set(results.keys())
    if mods == {"text"}:   return "text"
    if mods == {"image"}:  return "image"
    if mods == {"video"}:  return "video"
    if "text" in mods and "image" in mods and "video" in mods:
        return "combined"
    if "text" in mods and "image" in mods:  return "text+image"
    if "text" in mods and "video" in mods:  return "text+video"
    return "combined"


def _sanitise_for_db(results: dict, fusion: dict) -> dict:
    """Remove PIL images and non-serialisable objects before JSON encoding."""
    safe = {}
    for mod, r in results.items():
        safe[mod] = {k: v for k, v in r.items()
                     if not hasattr(v, "save") and not hasattr(v, "read")}
    safe["fusion"] = fusion
    return safe


def _empty_result(reason: str, errors: dict | None = None) -> dict:
    return {
        "final_score": 0.5,
        "verdict":     "UNCERTAIN",
        "confidence":  0.0,
        "individual":  {},
        "explanations":{},
        "modalities":  [],
        "fusion":      {},
        "errors":      errors or {"general": reason},
        "meta":        {"reason": reason, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "_raw":        {},
    }


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="inference.py",
        description="FakeShield AI — command-line inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --text "Breaking: Scientists discover miracle cure"
  python inference.py --text "…" --image photo.jpg
  python inference.py --text "…" --image photo.jpg --video https://youtu.be/xxx
  python inference.py --text "https://example.com/article"
  python inference.py --batch items.json --output results.json
""",
    )
    p.add_argument("--text",   type=str, help="Article text or URL")
    p.add_argument("--image",  type=str, help="Path to image file")
    p.add_argument("--video",  type=str, help="Video URL")
    p.add_argument("--no-db",  action="store_true", help="Skip saving to database")
    p.add_argument("--batch",  type=str, help="Path to JSON file with list of inputs")
    p.add_argument("--output", type=str, help="Path to write JSON output (stdout if omitted)")
    p.add_argument("--warmup", action="store_true", help="Pre-load all models then exit")
    p.add_argument("--verbose","-v", action="store_true")
    return p


def _cli_main():
    parser = _build_cli_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    engine = FakeNewsInference.get_instance()

    if args.warmup:
        engine.warm_up(("text", "image", "video"))
        print("✅ All models pre-loaded.")
        return

    # ── Batch mode ────────────────────────────────────────────────────────────
    if args.batch:
        with open(args.batch) as f:
            items = json.load(f)
        results = engine.predict_batch(items, save_to_db=not args.no_db)
        output  = json.dumps(results, indent=2, default=str)
    else:
        # ── Single prediction ──────────────────────────────────────────────────
        if not any([args.text, args.image, args.video]):
            parser.print_help()
            sys.exit(1)

        result = engine.predict(
            text      = args.text,
            image     = args.image,
            video_url = args.video,
            save_to_db= not args.no_db,
        )

        # Render a readable summary
        _print_summary(result)

        # Serialise (strip PIL images)
        safe = {k: v for k, v in result.items() if k != "_raw"}
        output = json.dumps(safe, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output)
        print(f"✅ Results written to {args.output}")
    else:
        print(output)


def _print_summary(result: dict):
    """Print a human-readable summary to stdout."""
    v  = result.get("verdict", "?")
    fs = result.get("final_score", 0.5)
    c  = result.get("confidence", 0.0)
    emoji = {"FAKE": "🔴", "REAL": "🟢", "SUSPICIOUS": "🟡"}.get(v, "⚪")

    print("\n" + "═" * 52)
    print(f"  {emoji}  Verdict:    {v}")
    print(f"     Fake score: {fs:.1%}")
    print(f"     Confidence: {c:.1%}")
    print()

    for mod, ind in result.get("individual", {}).items():
        print(f"  {mod.capitalize():<8}  {ind.get('fake_score',0):.1%}  "
              f"({ind.get('label','?')})")

    elapsed = result.get("meta", {}).get("elapsed_sec", 0)
    print(f"\n  ⏱  Elapsed: {elapsed:.2f}s")
    print("═" * 52 + "\n")


if __name__ == "__main__":
    _cli_main()
