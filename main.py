"""
main.py — FakeShield AI · Streamlit Cloud + CLI Unified Entry Point
====================================================================
Works in THREE modes automatically:

  1. Streamlit Cloud / `streamlit run main.py`
     → Full web UI (detected via Streamlit runtime context)

  2. CLI  →  `python main.py --text "article..." --image photo.jpg`
     → Command-line analysis, JSON output

  3. Web launcher  →  `python main.py --web [--mode api] [--port N]`
     → Spawns Streamlit or FastAPI server

The Streamlit-vs-CLI split happens at IMPORT TIME (line ~40) so
no CLI code ever executes on Streamlit Cloud.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# ── 1. Fix import paths — MUST be first ─────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 2. Detect Streamlit runtime ──────────────────────────────────────────────
def _running_in_streamlit() -> bool:
    """Return True when this file is being executed by the Streamlit runner."""
    # Primary check: Streamlit sets a script-run context
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        pass
    # Fallback: Streamlit Cloud sets this env var
    if os.environ.get("STREAMLIT_SERVER_PORT"):
        return True
    # Fallback 2: check argv for streamlit runner
    return any("streamlit" in arg for arg in sys.argv[:2])

_IN_STREAMLIT = _running_in_streamlit()


# ════════════════════════════════════════════════════════════════════════════
#  MODE A — STREAMLIT WEB UI
#  Executed when deployed on share.streamlit.io OR `streamlit run main.py`
# ════════════════════════════════════════════════════════════════════════════

if _IN_STREAMLIT:
    import io
    import warnings
    warnings.filterwarnings("ignore")

    import streamlit as st
    from PIL import Image

    # ── Page config — MUST be the very first Streamlit call ─────────────────
    st.set_page_config(
        page_title="FakeShield AI",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Project imports (all guarded inside this block) ──────────────────────
    from config import AppConfig
    from database import AnalysisDatabase
    from components.ui_components import UIComponents
    from components.results_display import ResultsDisplay
    from components.report_generator import ReportGenerator
    from utils.text_utils import fetch_url_content, is_valid_url, clean_text
    from utils.score_fusion import ScoreFusion

    AppConfig.ensure_dirs()

    # ── Lazy model loaders (cached once per session) ─────────────────────────
    @st.cache_resource(show_spinner="🧠 Loading BERT text model…")
    def _text_analyzer():
        from models.text_model import TextAnalyzer
        a = TextAnalyzer()
        a.load_model()
        return a

    @st.cache_resource(show_spinner="🖼️ Loading image analysis model…")
    def _image_analyzer():
        from models.image_model import ImageAnalyzer
        a = ImageAnalyzer()
        a.load_model()
        return a

    @st.cache_resource(show_spinner=False)
    def _video_analyzer():
        from models.video_model import VideoAnalyzer
        return VideoAnalyzer()

    @st.cache_resource(show_spinner=False)
    def _db():
        return AnalysisDatabase()

    # ── UI singletons ─────────────────────────────────────────────────────────
    _ui       = UIComponents()
    _display  = ResultsDisplay()
    _reporter = ReportGenerator()

    # ── Session state defaults ────────────────────────────────────────────────
    _SS_DEFAULTS = {"results": None, "report_html": None,
                    "report_pdf": None, "_fetched_text": ""}
    for _k, _v in _SS_DEFAULTS.items():
        st.session_state.setdefault(_k, _v)

    # ────────────────────────────────────────────────────────────────────────
    # Helper: run analysis across requested modalities
    # ────────────────────────────────────────────────────────────────────────
    def run_analysis(text=None, image=None, video_url=None) -> dict:
        results: dict = {}

        if text and text.strip():
            with st.spinner("🧠 Analysing text with BERT…"):
                try:
                    results["text"] = _text_analyzer().analyze(text.strip())
                except Exception as exc:
                    st.warning(f"Text analysis warning: {exc}")

        if image is not None:
            with st.spinner("🔬 Analysing image (ELA + Grad-CAM)…"):
                try:
                    results["image"] = _image_analyzer().analyze(image)
                except Exception as exc:
                    st.warning(f"Image analysis warning: {exc}")

        if video_url and video_url.strip():
            with st.spinner("🎬 Downloading & analysing video frames…"):
                try:
                    results["video"] = _video_analyzer().analyze(video_url.strip())
                except Exception as exc:
                    st.warning(f"Video analysis warning: {exc}")

        if results:
            try:
                results["overall"] = ScoreFusion().fuse(results)
            except Exception as exc:
                st.warning(f"Score fusion warning: {exc}")

        return results

    # ────────────────────────────────────────────────────────────────────────
    # Helper: persist result to SQLite
    # ────────────────────────────────────────────────────────────────────────
    def _save(results: dict, atype: str,
              text: str = "", video_url: str = "") -> None:
        try:
            overall = results.get("overall", {})
            _db().save_analysis({
                "analysis_type":    atype,
                "input_summary":    (text or "")[:300],
                "input_video_url":  video_url or "",
                "overall_score":    overall.get("overall_score", 0.5),
                "overall_label":    overall.get("overall_label", "UNCERTAIN"),
                "text_score":       results.get("text",  {}).get("fake_score"),
                "image_score":      results.get("image", {}).get("fake_score"),
                "video_score":      results.get("video", {}).get("fake_score"),
                "confidence":       overall.get("confidence", 0.0),
                "explanation_text": overall.get("explanation", ""),
                "details": {
                    k: {kk: vv for kk, vv in v.items()
                        if not hasattr(vv, "save")}
                    for k, v in results.items() if isinstance(v, dict)
                },
            })
        except Exception:
            pass   # DB errors must never crash the UI

    # ────────────────────────────────────────────────────────────────────────
    # Helper: trigger analysis, store in session, rerun
    # ────────────────────────────────────────────────────────────────────────
    def _dispatch(text=None, image=None, video_url=None,
                  atype="text") -> None:
        r = run_analysis(text=text, image=image, video_url=video_url)
        if r:
            _save(r, atype, text or "", video_url or "")
            st.session_state.results    = r
            st.session_state.report_html = None
            st.session_state.report_pdf  = None
            st.rerun()
        else:
            _ui.warning_panel("Analysis returned no results — check your input.")

    # ════════════════════════════════════════════════════════════════════════
    #  PAGE LAYOUT
    # ════════════════════════════════════════════════════════════════════════

    _ui.apply_custom_css()
    _ui.render_header()
    _ui.render_sidebar(_db())

    # ── Tabs ──────────────────────────────────────────────────────────────────
    t_text, t_img, t_vid, t_comb, t_hist = st.tabs([
        "📝 Text Analysis",
        "🖼️  Image Analysis",
        "🎬 Video Analysis",
        "🔗 Combined",
        "📊 History",
    ])

    # ── TAB 1: TEXT ──────────────────────────────────────────────────────────
    with t_text:
        st.markdown("### Analyse a news article or text snippet")
        _ui.info_panel(
            "Paste article text <b>or</b> a URL — the article will be "
            "automatically scraped from the page."
        )
        mode = st.radio(
            "Input mode", ["✏️ Paste text", "🔗 Enter URL"],
            horizontal=True, label_visibility="collapsed",
        )
        text_val = ""

        if "✏️" in mode:
            text_val = st.text_area(
                "Article text", height=220,
                placeholder="Paste the full article or news excerpt here…",
                label_visibility="collapsed",
            )
        else:
            raw_url = st.text_input(
                "Article URL",
                placeholder="https://example.com/article",
                label_visibility="collapsed",
            )
            if raw_url:
                if not is_valid_url(raw_url):
                    _ui.warning_panel("Please enter a valid HTTP(S) URL.")
                elif st.button("🌐 Fetch Article", key="fetch_btn"):
                    with st.spinner("Scraping article…"):
                        _, body = fetch_url_content(raw_url)
                        st.session_state._fetched_text = clean_text(body) if body else ""
            if st.session_state._fetched_text:
                text_val = st.session_state._fetched_text
                st.text_area(
                    "Fetched preview", text_val[:500] + "…",
                    height=120, disabled=True,
                )

        if st.button("🔍 Analyze Text", key="btn_text", type="primary"):
            words = len(text_val.strip().split())
            if words < AppConfig.MIN_TEXT_WORDS:
                _ui.warning_panel(
                    f"Please provide at least {AppConfig.MIN_TEXT_WORDS} words "
                    f"(you have {words})."
                )
            else:
                _dispatch(text=text_val, atype="text")

    # ── TAB 2: IMAGE ─────────────────────────────────────────────────────────
    with t_img:
        st.markdown("### Analyse an image for manipulation or deepfakes")
        _ui.info_panel(
            "Supported formats: JPEG, PNG, WebP. "
            "The system applies Error Level Analysis, noise mapping, and Grad-CAM."
        )
        uploaded = st.file_uploader(
            "Upload image", type=["jpg","jpeg","png","webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            pil_img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(pil_img, caption=uploaded.name, use_column_width=True)
            with c2:
                st.markdown(
                    f"**File:** `{uploaded.name}`  \n"
                    f"**Dimensions:** {pil_img.width} × {pil_img.height} px  \n"
                    f"**Mode:** {pil_img.mode}"
                )
            if st.button("🔍 Analyze Image", key="btn_img", type="primary"):
                _dispatch(image=pil_img, atype="image")

    # ── TAB 3: VIDEO ─────────────────────────────────────────────────────────
    with t_vid:
        st.markdown("### Analyse video frames for manipulation")
        _ui.info_panel(
            f"Supports YouTube, Vimeo, and direct MP4 links. "
            f"Up to {AppConfig.VIDEO_MAX_FRAMES} frames are extracted and analysed."
        )
        vid_url = st.text_input(
            "Video URL",
            placeholder="https://www.youtube.com/watch?v=…  or  https://example.com/video.mp4",
            label_visibility="collapsed",
        )
        if st.button("🔍 Analyze Video", key="btn_vid", type="primary"):
            if not vid_url.strip():
                _ui.warning_panel("Please enter a video URL.")
            elif not is_valid_url(vid_url.strip()):
                _ui.warning_panel("Invalid URL — must start with http:// or https://")
            else:
                _dispatch(video_url=vid_url.strip(), atype="video")

    # ── TAB 4: COMBINED ──────────────────────────────────────────────────────
    with t_comb:
        st.markdown("### Full multimodal analysis — text + image + video")
        _ui.info_panel(
            "Provide any combination of inputs. "
            "Scores are fused using weighted averages for maximum accuracy."
        )
        cl, cr = st.columns(2)
        with cl:
            comb_text = st.text_area(
                "📝 Article text (optional)", height=150,
                placeholder="Paste article text…",
                label_visibility="collapsed",
            )
        with cr:
            comb_img_file = st.file_uploader(
                "🖼️ Image (optional)", type=["jpg","jpeg","png","webp"],
                key="comb_img", label_visibility="collapsed",
            )
        comb_vid = st.text_input(
            "🎬 Video URL (optional)",
            placeholder="YouTube or direct MP4 URL (optional)…",
            label_visibility="collapsed",
        )

        has_any = bool(comb_text.strip() or comb_img_file or comb_vid.strip())
        if st.button("🔍 Run Multimodal Analysis", key="btn_comb",
                     type="primary", disabled=not has_any):
            comb_img = None
            if comb_img_file:
                comb_img = Image.open(
                    io.BytesIO(comb_img_file.read())
                ).convert("RGB")
            _dispatch(
                text=comb_text.strip() or None,
                image=comb_img,
                video_url=comb_vid.strip() or None,
                atype="combined",
            )

    # ── TAB 5: HISTORY ───────────────────────────────────────────────────────
    with t_hist:
        st.markdown("### 📊 Analysis History")
        try:
            hist  = _db().get_history(30)
            stats = _db().get_statistics()
            hc1, hc2, hc3, hc4 = st.columns(4)
            hc1.metric("Total Analyses", stats["total"])
            hc2.metric("🔴 Fake",         stats["fake"])
            hc3.metric("🟢 Real",         stats["real"])
            hc4.metric("🟡 Suspicious",   stats["suspicious"])

            if hist:
                import pandas as pd
                df = pd.DataFrame(hist)[[
                    "id","timestamp","analysis_type","overall_label","overall_score"
                ]]
                df["overall_score"] = df["overall_score"].apply(
                    lambda x: f"{x:.1%}" if isinstance(x, float) else "–"
                )
                df.columns = ["ID","Timestamp","Type","Verdict","Score"]
                st.dataframe(df, use_container_width=True, hide_index=True)
                if st.button("🗑️ Clear History"):
                    _db().clear_all()
                    st.rerun()
            else:
                _ui.info_panel("No analyses yet — run your first analysis above!")
        except Exception as _hist_err:
            st.error(f"History error: {_hist_err}")

    # ── RESULTS PANEL (persists below tabs) ──────────────────────────────────
    if st.session_state.results:
        st.divider()
        st.markdown("## 📊 Analysis Results")
        _display.render_all(st.session_state.results, _ui)

        st.divider()
        _ui.section_title("📥 Download Report")
        rc1, rc2, rc3 = st.columns([1, 1, 2])

        with rc1:
            if st.button("📄 Generate HTML"):
                st.session_state.report_html = _reporter.generate_html(
                    st.session_state.results
                )
            if st.session_state.report_html:
                html_bytes = st.session_state.report_html
                if isinstance(html_bytes, str):
                    html_bytes = html_bytes.encode()
                st.download_button(
                    "⬇️ Download HTML",
                    data=html_bytes,
                    file_name="fakeshield_report.html",
                    mime="text/html",
                )

        with rc2:
            if st.button("📑 Generate PDF"):
                pdf = _reporter.generate_pdf(st.session_state.results)
                if pdf:
                    st.session_state.report_pdf = pdf
                else:
                    _ui.warning_panel("PDF generation requires reportlab.")
            if st.session_state.get("report_pdf"):
                st.download_button(
                    "⬇️ Download PDF",
                    data=st.session_state.report_pdf,
                    file_name="fakeshield_report.pdf",
                    mime="application/pdf",
                )

        with rc3:
            if st.button("🔄 New Analysis"):
                for _k in ("results","report_html","report_pdf","_fetched_text"):
                    st.session_state[_k] = _SS_DEFAULTS.get(_k)
                st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  MODE B — CLI / LAUNCHER  (only when NOT in Streamlit)
# ════════════════════════════════════════════════════════════════════════════

else:
    import argparse
    import json
    import logging
    import subprocess

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    _log = logging.getLogger("main")

    # ── Argument parser ───────────────────────────────────────────────────────
    def _build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            prog="main.py",
            description="🛡️  FakeShield AI — Multimodal Fake News Detection",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples
--------
  # Text only
  python main.py --text "BREAKING: Shocking revelation exposes deep state!"

  # Text + image
  python main.py --text "Viral photo proves..." --image evidence.jpg

  # Full multimodal
  python main.py --text "Breaking news" --image img.jpg --video https://youtu.be/xxx

  # Analyse article from URL (auto-scraped)
  python main.py --text https://example.com/article

  # Batch mode
  python main.py --batch inputs.json --output results.json

  # Launch Streamlit UI
  python main.py --web

  # Launch FastAPI server
  python main.py --web --mode api --port 8000

  # Pre-warm all models
  python main.py --warmup
""",
        )
        # Web launcher
        p.add_argument("--web", action="store_true",
                       help="Launch web interface (Streamlit or FastAPI)")
        p.add_argument("--mode", choices=["streamlit","api"], default="streamlit",
                       help="Web backend (default: streamlit)")
        p.add_argument("--host", default="0.0.0.0")
        p.add_argument("--port", type=int, default=None)
        p.add_argument("--reload", action="store_true",
                       help="Enable hot-reload (FastAPI only)")
        # CLI inputs
        g = p.add_argument_group("analysis inputs")
        g.add_argument("--text",  metavar="TEXT_OR_URL",
                       help="Article text or URL")
        g.add_argument("--image", metavar="PATH",
                       help="Path to image file")
        g.add_argument("--video", metavar="URL",
                       help="Video URL")
        g.add_argument("--batch",  metavar="JSON",
                       help="Path to JSON file with list of input dicts")
        g.add_argument("--output", metavar="JSON",
                       help="Write results to this JSON file (default: stdout)")
        # Misc
        p.add_argument("--no-db",   action="store_true",
                       help="Skip saving result to SQLite")
        p.add_argument("--warmup",  action="store_true",
                       help="Pre-load all models and exit")
        p.add_argument("--verbose", "-v", action="store_true")
        return p

    # ── Web launchers ─────────────────────────────────────────────────────────
    def _launch_streamlit(host: str, port: int) -> None:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(ROOT / "main.py"),
            "--server.address", host,
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ]
        _log.info("Starting Streamlit → http://%s:%d", host, port)
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            _log.info("Stopped.")
        except FileNotFoundError:
            _log.error("streamlit not installed — pip install streamlit")
            sys.exit(1)

    def _launch_api(host: str, port: int, reload: bool) -> None:
        try:
            import uvicorn
        except ImportError:
            _log.error("uvicorn not installed — pip install uvicorn[standard]")
            sys.exit(1)
        _log.info("Starting FastAPI → http://%s:%d", host, port)
        uvicorn.run("api:app", host=host, port=port,
                    reload=reload, log_level="info")

    # ── CLI analysis ──────────────────────────────────────────────────────────
    def _run_cli(args: argparse.Namespace) -> None:
        from inference import FakeNewsInference

        engine = FakeNewsInference.get_instance()

        if args.batch:
            with open(args.batch) as fh:
                items = json.load(fh)
            if not isinstance(items, list):
                _log.error("Batch JSON must be a list of dicts")
                sys.exit(1)
            results = engine.predict_batch(items, save_to_db=not args.no_db)
            _write_output([_safe_json(r) for r in results], args.output)
            return

        if not any([args.text, args.image, args.video]):
            _log.error("No inputs provided — use --text, --image, or --video")
            sys.exit(1)

        if args.image and not Path(args.image).is_file():
            _log.error("Image file not found: %s", args.image)
            sys.exit(1)

        _log.info("Running analysis…")
        result = engine.predict(
            text=args.text,
            image=args.image,
            video_url=args.video,
            save_to_db=not args.no_db,
        )
        _print_summary(result)
        _write_output(_safe_json(result), args.output)

    def _print_summary(r: dict) -> None:
        v  = r.get("verdict", "?")
        fs = r.get("final_score", 0.5)
        c  = r.get("confidence", 0.0)
        em = {"FAKE":"🔴","REAL":"🟢","SUSPICIOUS":"🟡"}.get(v,"⚪")
        bar = "█" * int(fs * 40) + "░" * (40 - int(fs * 40))
        print(f"\n{'═'*56}")
        print(f"  {em}  Verdict:     {v}")
        print(f"     Fake score:  {fs:.1%}  [{bar}]")
        print(f"     Confidence:  {c:.1%}")
        for mod, ind in r.get("individual", {}).items():
            s = ind.get("fake_score", 0)
            print(f"     {mod.capitalize():<8}: {s:.1%}  ({ind.get('label','?')})")
        print(f"  ⏱  {r.get('meta',{}).get('elapsed_sec',0):.2f}s")
        print(f"{'═'*56}\n")

    def _safe_json(r: dict) -> dict:
        return {k: v for k, v in r.items()
                if k != "_raw" and not hasattr(v, "save")}

    def _write_output(data, path: str | None) -> None:
        out = json.dumps(data, indent=2, default=str)
        if path:
            Path(path).write_text(out)
            _log.info("Results written → %s", path)
        else:
            print(out)

    # ── Warm-up ───────────────────────────────────────────────────────────────
    def _warmup() -> None:
        from inference import FakeNewsInference
        _log.info("Pre-loading all models…")
        e = FakeNewsInference.get_instance()
        e.warm_up(("text", "image"))
        print("\n✅ Models loaded successfully.")
        print(f"   Load times: {e._load_times}")

    # ── Main entry point ──────────────────────────────────────────────────────
    def main() -> None:
        from config import AppConfig
        AppConfig.ensure_dirs()

        parser = _build_parser()
        args   = parser.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        if args.warmup:
            _warmup()
            return

        if args.web:
            port = args.port or (8000 if args.mode == "api" else 8501)
            if args.mode == "api":
                _launch_api(args.host, port, args.reload)
            else:
                _launch_streamlit(args.host, port)
            return

        _run_cli(args)

    if __name__ == "__main__":
        main()
