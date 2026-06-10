"""
config.py — Application Configuration
======================================
Central configuration for FakeShield AI.
Modify settings here to customize model behavior, UI, and thresholds.
"""

from pathlib import Path


class AppConfig:
    # ── App Info ────────────────────────────────────────────────────────────
    APP_NAME        = "FakeShield AI"
    APP_VERSION     = "1.0.0"
    APP_TAGLINE     = "Multimodal Fake News Detection System"
    APP_AUTHOR      = "Final Year Project — AI & Machine Learning"
    APP_ICON        = "🛡️"

    # ── Directory Paths ─────────────────────────────────────────────────────
    BASE_DIR    = Path(__file__).parent
    MODELS_DIR  = BASE_DIR / "models" / "weights"
    REPORTS_DIR = BASE_DIR / "reports"
    DATA_DIR    = BASE_DIR / "data"
    ASSETS_DIR  = BASE_DIR / "assets"
    TEMP_DIR    = BASE_DIR / "temp"

    # ── Database ─────────────────────────────────────────────────────────────
    DB_PATH = DATA_DIR / "analysis_history.db"

    # ── HuggingFace Models ───────────────────────────────────────────────────
    # Primary text fake-news classifier (BERT fine-tuned on LIAR dataset)
    TEXT_MODEL_PRIMARY  = "mrm8488/bert-tiny-finetuned-fake-news-detection"
    # RoBERTa fallback
    TEXT_MODEL_FALLBACK = "hamzab/roberta-fake-news-classification"
    # Zero-shot last resort
    TEXT_MODEL_ZSC      = "facebook/bart-large-mnli"

    # ── Image Model ──────────────────────────────────────────────────────────
    # EfficientNet-B0 from torchvision (pretrained on ImageNet)
    IMAGE_MODEL_ARCH    = "efficientnet_b0"
    IMAGE_SIZE          = (224, 224)          # resize input images to this
    ELA_QUALITY         = 90                  # JPEG quality for ELA pass
    ELA_SCALE           = 15                  # amplification for ELA display

    # ── Video Analysis ───────────────────────────────────────────────────────
    VIDEO_MAX_DURATION  = 300                 # seconds — reject longer videos
    VIDEO_FRAME_INTERVAL = 60                 # extract every Nth frame
    VIDEO_MAX_FRAMES    = 8                   # cap to avoid memory issues
    VIDEO_DOWNLOAD_TIMEOUT = 60              # seconds

    # ── Score Fusion Weights ─────────────────────────────────────────────────
    # Must sum to 1.0 when all modalities present
    WEIGHT_TEXT  = 0.45
    WEIGHT_IMAGE = 0.35
    WEIGHT_VIDEO = 0.20

    # ── Classification Thresholds ─────────────────────────────────────────────
    THRESHOLD_FAKE       = 0.60   # fake_score >= this → FAKE
    THRESHOLD_SUSPICIOUS = 0.40   # fake_score >= this → SUSPICIOUS (else REAL)

    # ── Text Limits ──────────────────────────────────────────────────────────
    MAX_TEXT_LENGTH  = 512        # tokens fed to the model
    MIN_TEXT_WORDS   = 10         # reject very short inputs
    URL_FETCH_TIMEOUT = 10        # seconds for article scraping

    # ── UI / Branding ────────────────────────────────────────────────────────
    COLOR_PRIMARY    = "#4F8BF9"
    COLOR_FAKE       = "#FF4444"
    COLOR_REAL       = "#00C851"
    COLOR_SUSPICIOUS = "#FFBB33"
    COLOR_NEUTRAL    = "#9E9E9E"
    COLOR_BG_DARK    = "#0E1117"
    COLOR_BG_CARD    = "#1E2130"
    COLOR_TEXT       = "#FAFAFA"

    # Gradient used for the header hero section
    HEADER_GRADIENT  = "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)"

    # ── Ensure runtime directories exist ────────────────────────────────────
    @classmethod
    def ensure_dirs(cls):
        for d in [cls.MODELS_DIR, cls.REPORTS_DIR, cls.DATA_DIR,
                  cls.ASSETS_DIR, cls.TEMP_DIR]:
            d.mkdir(parents=True, exist_ok=True)
