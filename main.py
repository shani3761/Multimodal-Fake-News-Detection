"""
main.py — FakeShield AI Production Entry Point
================================================
Supports three modes:

  1. CLI analysis (default)
     python main.py --text "article…"
     python main.py --text "…" --image photo.jpg --video https://youtu.be/xxx

  2. Streamlit web UI
     python main.py --web
     python main.py --web --mode streamlit --port 8501

  3. FastAPI REST server
     python main.py --web --mode api --port 8000

All models load lazily; the same ``FakeNewsInference`` singleton is used
regardless of the entry-point chosen.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# ── Ensure project root is importable ───────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import AppConfig

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("main")


# ════════════════════════════════════════════════════════════════════════════
# Argument parser
# ════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "🛡️  FakeShield AI — Multimodal Fake News Detection System\n"
            "University Final Year Project\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
──────────────────────────────────────────────────────
EXAMPLES
──────────────────────────────────────────────────────
  # CLI — text only
  python main.py --text "Doctors reveal shocking cure for everything!"

  # CLI — text + image
  python main.py --text "Viral photo proves…" --image evidence.jpg

  # CLI — full multimodal
  python main.py --text "Breaking news…" --image img.jpg --video https://youtu.be/xxx

  # CLI — feed a URL directly (article is scraped automatically)
  python main.py --text https://example.com/news/article

  # CLI — batch (JSON array of inputs)
  python main.py --batch inputs.json --output results.json

  # Web — Streamlit UI  (default)
  python main.py --web
  python main.py --web --port 8501

  # Web — FastAPI REST server
  python main.py --web --mode api --port 8000

  # Pre-load all models then exit (useful in Docker health checks)
  python main.py --warmup
──────────────────────────────────────────────────────
""",
    )

    # ── Web mode ─────────────────────────────────────────────────────────────
    p.add_argument(
        "--web", action="store_true",
        help="Launch web interface instead of CLI mode",
    )
    p.add_argument(
        "--mode", choices=["streamlit", "api"], default="streamlit",
        help="Web backend: 'streamlit' (default) or 'api' (FastAPI + uvicorn)",
    )
    p.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address for the web server (default: 0.0.0.0)",
    )
    p.add_argument(
        "--port", type=int, default=None,
        help="Port (default: 8501 for Streamlit, 8000 for API)",
    )
    p.add_argument(
        "--reload", action="store_true",
        help="Enable hot-reload (FastAPI/uvicorn only)",
    )

    # ── CLI analysis inputs ───────────────────────────────────────────────────
    cli = p.add_argument_group("CLI analysis")
    cli.add_argument("--text",  type=str, metavar="TEXT_OR_URL",
                     help="Article text or URL to analyse")
    cli.add_argument("--image", type=str, metavar="PATH",
                     help="Path to an image file")
    cli.add_argument("--video", type=str, metavar="URL",
                     help="Video URL (YouTube / direct .mp4)")

    # ── Batch mode ────────────────────────────────────────────────────────────
    cli.add_argument("--batch",  type=str, metavar="JSON",
                     help="Path to a JSON file with a list of input dicts")
    cli.add_argument("--output", type=str, metavar="JSON",
                     help="Write JSON results to this file (default: stdout)")

    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--no-db",  action="store_true",
                   help="Skip persisting results to SQLite")
    p.add_argument("--warmup", action="store_true",
                   help="Pre-load all models, print summary, and exit")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG logging")

    return p


# ════════════════════════════════════════════════════════════════════════════
# Web launchers
# ════════════════════════════════════════════════════════════════════════════

def launch_streamlit(host: str, port: int):
    """Start the Streamlit frontend."""
    app_path = str(ROOT / "app.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.address", host,
        "--server.port",    str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    logger.info("Starting Streamlit on http://%s:%d …", host, port)
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Streamlit server stopped.")
    except FileNotFoundError:
        logger.error("streamlit not found. Install it: pip install streamlit")
        sys.exit(1)


def launch_api(host: str, port: int, reload: bool):
    """Start the FastAPI server via uvicorn."""
    try:
        import uvicorn
    except ImportError:
        logger.error("uvicorn not found. Install: pip install uvicorn[standard]")
        sys.exit(1)

    logger.info("Starting FastAPI on http://%s:%d …", host, port)
    uvicorn.run(
        "api:app",
        host    = host,
        port    = port,
        reload  = reload,
        workers = 1 if reload else None,
        log_level="info",
    )


# ════════════════════════════════════════════════════════════════════════════
# CLI analysis runner
# ════════════════════════════════════════════════════════════════════════════

def run_cli(args: argparse.Namespace):
    """Execute CLI analysis and print / save results."""
    import json as _json
    from inference import FakeNewsInference

    engine = FakeNewsInference.get_instance()

    if args.batch:
        _run_batch(engine, args)
        return

    if not any([args.text, args.image, args.video]):
        logger.error("No inputs provided. Pass --text, --image, --video or --batch.")
        sys.exit(1)

    _validate_inputs(args)

    logger.info("Running analysis …")
    result = engine.predict(
        text      = args.text,
        image     = args.image,
        video_url = args.video,
        save_to_db= not args.no_db,
    )

    _print_rich_summary(result)

    # Serialise (strip PIL images and other non-JSON objects)
    safe   = _serialise(result)
    output = _json.dumps(safe, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output)
        logger.info("Results written → %s", args.output)
    else:
        print(output)


def _run_batch(engine, args: argparse.Namespace):
    import json as _json

    logger.info("Batch mode — loading %s …", args.batch)
    with open(args.batch) as f:
        items = _json.load(f)

    if not isinstance(items, list):
        logger.error("Batch JSON must be a list of dicts.")
        sys.exit(1)

    results = engine.predict_batch(items, save_to_db=not args.no_db)
    output  = _json.dumps([_serialise(r) for r in results], indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output)
        logger.info("Batch results written → %s (%d items)", args.output, len(results))
    else:
        print(output)


def _validate_inputs(args: argparse.Namespace):
    if args.image and not Path(args.image).is_file():
        logger.error("Image file not found: %s", args.image)
        sys.exit(1)


def _print_rich_summary(result: dict):
    v     = result.get("verdict", "?")
    fs    = result.get("final_score", 0.5)
    conf  = result.get("confidence", 0.0)
    emoji = {"FAKE": "🔴", "REAL": "🟢", "SUSPICIOUS": "🟡"}.get(v, "⚪")

    bar_len = 40
    filled  = int(fs * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)

    print("\n" + "═" * 56)
    print(f"  🛡️  FakeShield AI — Analysis Result")
    print("─" * 56)
    print(f"  {emoji}  Verdict   :  {v}")
    print(f"  📊  Fake score:  {fs:.1%}  [{bar}]")
    print(f"  🎯  Confidence:  {conf:.1%}")
    print("─" * 56)

    for mod, ind in result.get("individual", {}).items():
        ind_bar = "█" * int(ind["fake_score"] * 20) + "░" * (20 - int(ind["fake_score"] * 20))
        print(f"  {mod.capitalize():<8}  {ind['fake_score']:.1%}  [{ind_bar}]  ({ind['label']})")

    elapsed = result.get("meta", {}).get("elapsed_sec", 0)
    ts      = result.get("meta", {}).get("timestamp", "")
    print("─" * 56)
    print(f"  ⏱  Elapsed: {elapsed:.2f}s  ·  {ts}")
    print("═" * 56 + "\n")


def _serialise(result: dict) -> dict:
    """Return a JSON-safe version of the result dict."""
    safe = {}
    for k, v in result.items():
        if k == "_raw":
            continue
        if isinstance(v, dict):
            safe[k] = {kk: vv for kk, vv in v.items()
                       if not hasattr(vv, "save") and not hasattr(vv, "read")}
        else:
            safe[k] = v
    return safe


# ════════════════════════════════════════════════════════════════════════════
# Warm-up
# ════════════════════════════════════════════════════════════════════════════

def run_warmup():
    from inference import FakeNewsInference
    logger.info("Pre-loading all models …")
    engine = FakeNewsInference.get_instance()
    engine.warm_up(("text", "image", "video"))
    print("\n✅ All models loaded successfully.")
    print(f"   Load times: {engine._load_times}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    AppConfig.ensure_dirs()
    parser = build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    _print_banner()

    # ── Warm-up mode ─────────────────────────────────────────────────────────
    if args.warmup:
        run_warmup()
        return

    # ── Web mode ─────────────────────────────────────────────────────────────
    if args.web:
        if args.mode == "streamlit":
            port = args.port or 8501
            launch_streamlit(args.host, port)
        else:
            port = args.port or 8000
            launch_api(args.host, port, args.reload)
        return

    # ── CLI mode ──────────────────────────────────────────────────────────────
    run_cli(args)


def _print_banner():
    print("""
 ███████╗ █████╗ ██╗  ██╗███████╗███████╗██╗  ██╗██╗███████╗██╗     ██████╗      █████╗ ██╗
 ██╔════╝██╔══██╗██║ ██╔╝██╔════╝██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗    ██╔══██╗██║
 █████╗  ███████║█████╔╝ █████╗  ███████╗███████║██║█████╗  ██║     ██║  ██║    ███████║██║
 ██╔══╝  ██╔══██║██╔═██╗ ██╔══╝  ╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║    ██╔══██║██║
 ██║     ██║  ██║██║  ██╗███████╗███████║██║  ██║██║███████╗███████╗██████╔╝    ██║  ██║██║
 ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝     ╚═╝  ╚═╝╚═╝

  Multimodal Fake News Detection System  ·  v{ver}
  University Final Year Project
""".format(ver=AppConfig.APP_VERSION))


if __name__ == "__main__":
    main()
