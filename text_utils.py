"""
utils/text_utils.py — Text Preprocessing & URL Scraping
=========================================================
Helpers for cleaning article text, scraping articles from URLs,
detecting sensationalism, and extracting linguistic features.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AppConfig


# ── URL content fetching ─────────────────────────────────────────────────────
def fetch_url_content(url: str) -> tuple[str, str]:
    """
    Scrape article text from a news URL.

    Returns
    -------
    (title, body_text)  Both strings; empty on failure.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers,
                            timeout=AppConfig.URL_FETCH_TIMEOUT)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script / style / nav noise
        for tag in soup(["script", "style", "nav", "header",
                          "footer", "aside", "figure"]):
            tag.decompose()

        title = soup.find("h1")
        title = title.get_text(strip=True) if title else ""

        # Try article tag first, then all paragraphs
        article = soup.find("article")
        if article:
            paragraphs = article.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        body = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        body = clean_text(body)

        return title, body

    except Exception:
        return "", ""


# ── Text cleaning ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Basic normalisation for raw article text."""
    # Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    # Remove repeated punctuation
    text = re.sub(r"[!?]{2,}", "!", text)
    return text.strip()


def truncate_text(text: str, max_words: int = 300) -> str:
    """Truncate to *max_words* preserving sentence boundaries."""
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    # Back up to last sentence end
    for punct in (".", "!", "?"):
        idx = truncated.rfind(punct)
        if idx > len(truncated) * 0.8:
            return truncated[:idx + 1] + " …"
    return truncated + " …"


# ── Feature extraction ───────────────────────────────────────────────────────
def detect_sensationalism(text: str) -> dict:
    """
    Measure linguistic features associated with misinformation.

    Returns
    -------
    dict with:
        score        float 0–1
        caps_ratio   float  proportion of ALL-CAPS words
        exclamation  int    count of exclamation marks
        question     int    count of question marks
        clickbait    list[str]  matched phrases
    """
    CLICKBAIT = [
        "you won't believe", "shocking", "bombshell", "explosive",
        "breaking", "exclusive", "they don't want you", "secret",
        "exposed", "banned", "censored", "wake up", "share before",
        "mainstream media", "miracle", "one weird trick",
        "doctors hate", "urgent", "must read",
    ]

    words     = text.split()
    caps_cnt  = sum(1 for w in words if w.isupper() and len(w) > 2)
    caps_ratio = caps_cnt / max(len(words), 1)
    exclaim   = text.count("!")
    questions = text.count("?")
    lower     = text.lower()
    hits      = [cb for cb in CLICKBAIT if cb in lower]

    raw_score = (
        caps_ratio * 0.3
        + min(exclaim / 10, 1.0) * 0.25
        + min(len(hits) / 5, 1.0) * 0.45
    )

    return {
        "score":      round(min(raw_score, 1.0), 4),
        "caps_ratio": round(caps_ratio, 4),
        "exclamation": exclaim,
        "question":    questions,
        "clickbait":   hits,
    }


def extract_entities(text: str) -> dict:
    """
    Extract named entities using simple regex patterns.
    (spaCy is an optional enhancement.)
    """
    # Capitalised proper nouns (rough heuristic)
    proper_nouns = re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text)
    urls         = re.findall(r"https?://\S+", text)
    numbers      = re.findall(r"\b\d[\d,\.]*\s*(?:%|million|billion|thousand)?\b", text)

    return {
        "proper_nouns": list(dict.fromkeys(proper_nouns))[:10],
        "urls":         urls[:5],
        "statistics":   list(dict.fromkeys(numbers))[:10],
    }


def is_valid_url(url: str) -> bool:
    """Return True if the string looks like a valid HTTP URL."""
    try:
        result = urlparse(url)
        return result.scheme in ("http", "https") and bool(result.netloc)
    except Exception:
        return False


def word_count(text: str) -> int:
    return len(text.split())


def reading_time_minutes(text: str) -> float:
    """Estimate reading time (avg 200 wpm)."""
    return round(word_count(text) / 200, 1)
