"""
utils/video_utils.py — Video Processing Utilities
===================================================
Provides all video I/O and preprocessing helpers for FakeShield AI:

  extract_frames(video_path, num_frames)  →  list[PIL.Image]
  extract_audio(video_path)               →  audio path or None
  get_video_metadata(video_path)          →  dict
  preprocess_video(video_path, ...)       →  list[np.ndarray]
  detect_scene_changes(video_path, ...)   →  list[int]  (frame indices)
  select_keyframes(video_path, n)         →  list[PIL.Image]  (scene-aware)

All functions return safe defaults on error; no exception is propagated
to the caller — failure reason is stored in the return value instead.
"""

from __future__ import annotations

import os
import sys
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AppConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Core helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_video_metadata(video_path: Union[str, Path]) -> dict:
    """
    Return basic metadata about a video file.

    Parameters
    ----------
    video_path : str | Path

    Returns
    -------
    dict  with keys: width, height, fps, total_frames, duration_sec,
                     codec, has_audio, file_size_mb, error (if any)
    """
    video_path = str(video_path)
    meta: dict = {
        "width":         0,
        "height":        0,
        "fps":           0.0,
        "total_frames":  0,
        "duration_sec":  0.0,
        "codec":         "unknown",
        "has_audio":     False,
        "file_size_mb":  0.0,
        "error":         None,
    }

    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            meta["error"] = f"Cannot open: {video_path}"
            return meta

        meta["width"]        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        meta["height"]       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        meta["fps"]          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        meta["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        meta["duration_sec"] = meta["total_frames"] / max(meta["fps"], 1)
        fourcc_int           = int(cap.get(cv2.CAP_PROP_FOURCC))
        meta["codec"]        = "".join(chr((fourcc_int >> (8 * i)) & 0xFF)
                                       for i in range(4)).strip()
        cap.release()

        if os.path.isfile(video_path):
            meta["file_size_mb"] = round(os.path.getsize(video_path) / 1e6, 2)

        # Detect audio stream with moviepy (optional)
        try:
            from moviepy.editor import VideoFileClip
            with VideoFileClip(video_path) as clip:
                meta["has_audio"] = clip.audio is not None
        except Exception:
            pass

    except Exception as exc:
        meta["error"] = str(exc)
        logger.warning("get_video_metadata failed: %s", exc)

    return meta


# ───────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: Union[str, Path],
    num_frames: int = 16,
    target_size: tuple[int, int] = AppConfig.IMAGE_SIZE,
) -> list[Image.Image]:
    """
    Extract *num_frames* evenly-spaced frames from a video.

    Parameters
    ----------
    video_path  : path to local video file
    num_frames  : how many frames to extract
    target_size : (width, height) to resize each frame

    Returns
    -------
    list of PIL Images (may be shorter than *num_frames* if video is short).
    Returns an empty list on failure.
    """
    video_path = str(video_path)
    frames: list[Image.Image] = []

    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("extract_frames: cannot open %s", video_path)
            return frames

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            logger.warning("extract_frames: no frames reported for %s", video_path)
            cap.release()
            return frames

        n        = min(num_frames, total)
        indices  = np.linspace(0, total - 1, n, dtype=int).tolist()

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if not ok:
                logger.debug("extract_frames: missed frame %d", idx)
                continue
            rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil   = Image.fromarray(rgb).resize(target_size, Image.LANCZOS)
            frames.append(pil)

        cap.release()

    except Exception as exc:
        logger.error("extract_frames error: %s", exc)

    return frames


# ───────────────────────────────────────────────────────────────────────────────

def extract_audio(
    video_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    audio_format: str = "wav",
) -> Optional[str]:
    """
    Extract the audio track from a video file.

    Parameters
    ----------
    video_path   : path to local video file
    output_dir   : directory to save the extracted audio
                   (uses system temp dir if None)
    audio_format : "wav" | "mp3" | "aac"

    Returns
    -------
    str path to the extracted audio file, or None on failure.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        logger.error("extract_audio: file not found: %s", video_path)
        return None

    if output_dir is None:
        output_dir = Path(tempfile.gettempdir())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem      = hashlib.md5(str(video_path).encode()).hexdigest()[:10]
    out_path  = output_dir / f"audio_{stem}.{audio_format}"

    try:
        from moviepy.editor import VideoFileClip

        with VideoFileClip(str(video_path)) as clip:
            if clip.audio is None:
                logger.warning("extract_audio: no audio stream in %s", video_path)
                return None
            clip.audio.write_audiofile(str(out_path), logger=None)

        return str(out_path)

    except ImportError:
        logger.warning("moviepy not installed; trying ffmpeg directly")
        return _extract_audio_ffmpeg(str(video_path), str(out_path), audio_format)
    except Exception as exc:
        logger.error("extract_audio error: %s", exc)
        return None


def _extract_audio_ffmpeg(src: str, dst: str, fmt: str) -> Optional[str]:
    """Fallback: use ffmpeg subprocess to extract audio."""
    import subprocess
    try:
        cmd = ["ffmpeg", "-y", "-i", src, "-vn",
               "-acodec", "pcm_s16le" if fmt == "wav" else "copy",
               dst]
        subprocess.run(cmd, check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return dst if os.path.isfile(dst) else None
    except Exception as exc:
        logger.error("_extract_audio_ffmpeg: %s", exc)
        return None


# ───────────────────────────────────────────────────────────────────────────────

def detect_scene_changes(
    video_path: Union[str, Path],
    threshold: float = 30.0,
    max_scenes: int = AppConfig.VIDEO_MAX_FRAMES,
) -> list[int]:
    """
    Detect shot boundaries using per-frame histogram difference.

    Parameters
    ----------
    video_path  : local video path
    threshold   : mean-absolute-difference threshold between consecutive frames
    max_scenes  : maximum number of scene-change frame indices to return

    Returns
    -------
    list of frame indices where a scene change was detected.
    Returns [0] as a single keyframe if detection fails.
    """
    video_path = str(video_path)
    scene_frames: list[int] = [0]        # always include frame 0

    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return scene_frames

        prev_hist = None
        frame_idx = 0

        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            gray     = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            hist     = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist     = hist.flatten() / (gray.size + 1e-6)

            if prev_hist is not None:
                diff = float(np.mean(np.abs(hist - prev_hist)) * 1000)
                if diff > threshold and frame_idx not in scene_frames:
                    scene_frames.append(frame_idx)

            prev_hist = hist
            frame_idx += 1

        cap.release()

    except Exception as exc:
        logger.error("detect_scene_changes error: %s", exc)

    # Cap and sort
    scene_frames = sorted(set(scene_frames))[:max_scenes]
    return scene_frames


# ───────────────────────────────────────────────────────────────────────────────

def select_keyframes(
    video_path: Union[str, Path],
    n: int = AppConfig.VIDEO_MAX_FRAMES,
    target_size: tuple[int, int] = AppConfig.IMAGE_SIZE,
    use_scene_detection: bool = True,
) -> list[Image.Image]:
    """
    Select the most representative keyframes, preferring scene-change
    boundaries for richer coverage.

    Parameters
    ----------
    video_path           : local video path
    n                    : number of keyframes to return
    target_size          : output frame size (W, H)
    use_scene_detection  : if False, falls back to uniform sampling

    Returns
    -------
    list[PIL.Image]
    """
    if use_scene_detection:
        try:
            indices = detect_scene_changes(video_path, max_scenes=n)
            if len(indices) >= 2:
                return _read_frames_at_indices(str(video_path), indices, target_size)
        except Exception:
            pass

    # Fallback: uniform sampling
    return extract_frames(video_path, num_frames=n, target_size=target_size)


def _read_frames_at_indices(
    video_path: str,
    indices: list[int],
    target_size: tuple[int, int],
) -> list[Image.Image]:
    """Read frames at specific indices from an open video."""
    frames: list[Image.Image] = []
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil   = Image.fromarray(rgb).resize(target_size, Image.LANCZOS)
            frames.append(pil)
        cap.release()
    except Exception as exc:
        logger.error("_read_frames_at_indices: %s", exc)
    return frames


# ───────────────────────────────────────────────────────────────────────────────

def preprocess_video(
    video_path: Union[str, Path],
    target_size: tuple[int, int] = AppConfig.IMAGE_SIZE,
    max_frames: int = AppConfig.VIDEO_MAX_FRAMES,
    normalise: bool = True,
) -> list[np.ndarray]:
    """
    Full preprocessing pipeline: extract keyframes → resize → normalise.

    Parameters
    ----------
    video_path   : local video path
    target_size  : (W, H) to resize each frame
    max_frames   : maximum frames to return
    normalise    : if True, scale pixel values to [0, 1] float32

    Returns
    -------
    list of np.ndarray, each shape (H, W, 3), dtype float32 or uint8.
    """
    frames = select_keyframes(video_path, n=max_frames, target_size=target_size)
    result: list[np.ndarray] = []

    for pil in frames:
        arr = np.array(pil.convert("RGB"), dtype=np.float32 if normalise else np.uint8)
        if normalise:
            arr = arr / 255.0
        result.append(arr)

    return result


# ───────────────────────────────────────────────────────────────────────────────

def download_video(
    url: str,
    dest_dir: Optional[Union[str, Path]] = None,
    max_filesize_mb: int = 200,
    max_duration_sec: int = AppConfig.VIDEO_MAX_DURATION,
) -> Optional[str]:
    """
    Download a video from a URL using yt-dlp (handles YouTube, Vimeo, etc.)
    with a direct HTTP fallback.

    Parameters
    ----------
    url              : YouTube URL, Vimeo, or direct .mp4 link
    dest_dir         : directory to save the file (uses temp dir if None)
    max_filesize_mb  : reject files larger than this
    max_duration_sec : reject videos longer than this (in seconds)

    Returns
    -------
    str path to the downloaded file, or None on failure.
    """
    if dest_dir is None:
        dest_dir = AppConfig.TEMP_DIR
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    stem     = hashlib.md5(url.encode()).hexdigest()[:10]
    template = str(dest_dir / f"video_{stem}.%(ext)s")

    # ── yt-dlp ────────────────────────────────────────────────────────────────
    try:
        import yt_dlp

        ydl_opts = {
            "outtmpl":    template,
            "format":     "bestvideo[height<=480][ext=mp4]/best[height<=480]/best",
            "quiet":      True,
            "no_warnings": True,
            "socket_timeout": AppConfig.VIDEO_DOWNLOAD_TIMEOUT,
            "max_filesize":   f"{max_filesize_mb}M",
            "match_filter":   _yt_dlp_duration_filter(max_duration_sec),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)

        if Path(path).is_file():
            return path

        # yt-dlp may change extension
        candidates = sorted(dest_dir.glob(f"video_{stem}.*"),
                            key=lambda p: p.stat().st_size, reverse=True)
        if candidates:
            return str(candidates[0])

    except ImportError:
        logger.warning("yt-dlp not installed; trying direct HTTP download")
    except Exception as exc:
        logger.warning("yt-dlp failed (%s); trying direct HTTP download", exc)

    # ── Direct HTTP fallback ──────────────────────────────────────────────────
    return _http_download(url, dest_dir / f"video_{stem}.mp4", max_filesize_mb)


def _yt_dlp_duration_filter(max_sec: int):
    """Return a yt-dlp match_filter function that rejects long videos."""
    def _filter(info_dict, **kwargs):
        dur = info_dict.get("duration", 0) or 0
        if dur > max_sec:
            return f"Video too long ({dur}s > {max_sec}s)"
        return None
    return _filter


def _http_download(url: str, dest: Path, max_mb: int) -> Optional[str]:
    """Fallback: stream-download a direct URL."""
    try:
        import requests
        r = requests.get(url, stream=True, timeout=AppConfig.VIDEO_DOWNLOAD_TIMEOUT)
        r.raise_for_status()
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):   # 1 MiB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded > max_mb * 1e6:
                    logger.warning("_http_download: size limit exceeded")
                    break
        return str(dest) if dest.is_file() and dest.stat().st_size > 0 else None
    except Exception as exc:
        logger.error("_http_download error: %s", exc)
        return None


# ───────────────────────────────────────────────────────────────────────────────

def is_supported_video_url(url: str) -> bool:
    """Return True if yt-dlp is likely to handle the URL."""
    supported_domains = (
        "youtube.com", "youtu.be", "vimeo.com", "dailymotion.com",
        "twitter.com", "x.com", "tiktok.com", "facebook.com",
        "instagram.com", "reddit.com",
    )
    lower = url.lower()
    return any(d in lower for d in supported_domains) or lower.endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".webm")
    )


def cleanup_temp_video(path: Union[str, Path]) -> bool:
    """Safely remove a downloaded temp video file."""
    try:
        p = Path(path)
        if p.is_file():
            p.unlink()
        return True
    except Exception as exc:
        logger.warning("cleanup_temp_video: %s", exc)
        return False
