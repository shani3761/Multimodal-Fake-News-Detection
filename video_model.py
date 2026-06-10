"""
models/video_model.py — Video Frame Extraction & Analysis
==========================================================
Pipeline:
  1. Download video from YouTube URL or direct URL (yt-dlp)
  2. Extract N evenly-spaced key frames (OpenCV)
  3. Run ImageAnalyzer on each frame
  4. Aggregate per-frame scores → overall video manipulation score
  5. Cleanup temp files
"""

from __future__ import annotations
import os
import sys
import time
import uuid
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AppConfig
from models.image_model import ImageAnalyzer


class VideoAnalyzer:
    """
    Downloads a video, extracts frames, and scores each frame for manipulation.

    Methods
    -------
    analyze(url)   Full pipeline — returns a result dict.
    """

    def __init__(self):
        self._image_analyzer = ImageAnalyzer()
        self._temp_dir       = AppConfig.TEMP_DIR
        AppConfig.ensure_dirs()

    # ── Public API ───────────────────────────────────────────────────────────
    def analyze(self, url: str) -> dict:
        """
        Parameters
        ----------
        url : str  YouTube URL, Vimeo URL, or direct .mp4 / .avi link.

        Returns
        -------
        dict with keys:
            fake_score      float 0–1
            real_score      float 0–1
            label           str
            confidence      float 0–1
            frame_scores    list[float]
            frame_images    list[PIL.Image]   (key frames for display)
            frames_analyzed int
            explanation     str
        """
        # Ensure image model is loaded
        if not self._image_analyzer.is_loaded:
            self._image_analyzer.load_model()

        job_dir = self._temp_dir / f"video_{uuid.uuid4().hex[:8]}"
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Download
            video_path = self._download(url, job_dir)
            if video_path is None:
                return self._error_result(f"Could not download video from: {url}")

            # 2. Extract frames
            frames, fps = self._extract_frames(video_path)
            if not frames:
                return self._error_result("No frames could be extracted from the video.")

            # 3. Analyse each frame
            frame_results  = []
            frame_images   = []
            for frame in frames:
                res = self._image_analyzer.analyze(frame)
                frame_results.append(res)
                frame_images.append(frame)

            # 4. Aggregate
            scores      = [r["fake_score"] for r in frame_results]
            fake_score  = float(np.mean(scores))
            fake_score  = max(0.0, min(1.0, fake_score))

            label = _score_to_label(fake_score)

            return {
                "fake_score":      round(fake_score, 4),
                "real_score":      round(1 - fake_score, 4),
                "label":           label,
                "confidence":      round(abs(fake_score - 0.5) * 2, 4),
                "frame_scores":    [round(s, 4) for s in scores],
                "frame_images":    frame_images,
                "frames_analyzed": len(frames),
                "fps":             fps,
                "explanation":     self._explain(fake_score, scores),
            }

        except Exception as exc:
            return self._error_result(str(exc))
        finally:
            # Clean up downloaded files
            try:
                shutil.rmtree(job_dir, ignore_errors=True)
            except Exception:
                pass

    # ── Download ─────────────────────────────────────────────────────────────
    def _download(self, url: str, dest_dir: Path) -> Optional[Path]:
        """
        Download using yt-dlp (handles YouTube, Vimeo, direct MP4 links).
        Returns the path to the downloaded file, or None on failure.
        """
        try:
            import yt_dlp

            out_template = str(dest_dir / "%(id)s.%(ext)s")
            ydl_opts = {
                "outtmpl":   out_template,
                "format":    "bestvideo[height<=480][ext=mp4]/best[height<=480]/best",
                "quiet":     True,
                "no_warnings": True,
                "socket_timeout": AppConfig.VIDEO_DOWNLOAD_TIMEOUT,
                "max_filesize":   "100M",
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                filename = ydl.prepare_filename(info)

            # Find downloaded file
            candidates = list(dest_dir.glob("*"))
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_size)

        except ImportError:
            # yt-dlp not installed — try direct HTTP download
            return self._direct_download(url, dest_dir)
        except Exception:
            return self._direct_download(url, dest_dir)

        return None

    def _direct_download(self, url: str, dest_dir: Path) -> Optional[Path]:
        """Fallback: download raw URL content (works for direct MP4 links)."""
        try:
            import requests
            dest = dest_dir / "video.mp4"
            r = requests.get(url, stream=True,
                             timeout=AppConfig.VIDEO_DOWNLOAD_TIMEOUT)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            return dest
        except Exception:
            return None

    # ── Frame extraction ─────────────────────────────────────────────────────
    def _extract_frames(self, video_path: Path) -> tuple[list[Image.Image], float]:
        """
        Use OpenCV to extract evenly-spaced key frames.

        Returns
        -------
        (frames, fps) where frames is a list of PIL Images.
        """
        try:
            import cv2
        except ImportError:
            return [], 0.0

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [], 0.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration_sec = total_frames / fps

        # Respect max-duration limit
        if duration_sec > AppConfig.VIDEO_MAX_DURATION:
            cap.release()
            return [], fps

        # Compute frame indices to sample
        n_frames = min(AppConfig.VIDEO_MAX_FRAMES, max(1, total_frames // AppConfig.VIDEO_FRAME_INTERVAL))
        indices  = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image  = Image.fromarray(frame_rgb)
            # Resize to keep memory manageable
            pil_image  = pil_image.resize(AppConfig.IMAGE_SIZE, Image.LANCZOS)
            frames.append(pil_image)

        cap.release()
        return frames, fps

    # ── Helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _explain(overall: float, scores: list[float]) -> str:
        n     = len(scores)
        high  = sum(1 for s in scores if s >= AppConfig.THRESHOLD_FAKE)
        parts = [f"Analyzed {n} video frame(s)."]

        if high > 0:
            parts.append(
                f"{high}/{n} frame(s) showed signs of manipulation "
                f"(score ≥ {AppConfig.THRESHOLD_FAKE})."
            )
        if overall >= AppConfig.THRESHOLD_FAKE:
            parts.append("The video is likely manipulated or synthetic.")
        elif overall >= AppConfig.THRESHOLD_SUSPICIOUS:
            parts.append("Some frames appear suspicious; further review recommended.")
        else:
            parts.append("No significant manipulation detected in analyzed frames.")

        return " ".join(parts)

    @staticmethod
    def _error_result(msg: str) -> dict:
        return {
            "fake_score":      0.5,
            "real_score":      0.5,
            "label":           "UNCERTAIN",
            "confidence":      0.0,
            "frame_scores":    [],
            "frame_images":    [],
            "frames_analyzed": 0,
            "fps":             0.0,
            "explanation":     f"Analysis failed: {msg}",
            "error":           msg,
        }


# ── Helper ───────────────────────────────────────────────────────────────────
def _score_to_label(score: float) -> str:
    from config import AppConfig
    if score >= AppConfig.THRESHOLD_FAKE:
        return "FAKE"
    if score >= AppConfig.THRESHOLD_SUSPICIOUS:
        return "SUSPICIOUS"
    return "REAL"
