cat > /home/claude/fake_news_detector/main.py << 'ENDOFFILE'
"""
main.py ─ FakeShield AI · Streamlit Cloud & CLI Unified Entry Point
=====================================================================
Deploy on share.streamlit.io  →  set "Main file path" to  main.py
Run locally                   →  streamlit run main.py
CLI                           →  python main.py --text "..."
API server                    →  python main.py --web --mode api

Architecture
────────────
  • ALL project imports live INSIDE functions / @st.cache_resource
    so a broken optional dependency (reportlab, cv2, torch …) never
    crashes the whole app at import-time.
  • __init__.py files are intentionally EMPTY to prevent eager-loading.
  • Path setup happens on line 1 before anything else.
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# 0.  ABSOLUTE FIRST — sys.path + working directory
#     Must run before ANY other import so every "from x import y" resolves.
# ══════════════════════════════════════════════════════════════════════════════
import os, sys
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Ensure sub-packages can also find the project root via os.getcwd()
os.chdir(_ROOT)

# Propagate to child processes (uvicorn, subprocess launchers)
os.environ["PYTHONPATH"] = (
    str(_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Detect Streamlit runtime
# ══════════════════════════════════════════════════════════════════════════════
def _in_streamlit() -> bool:
    """Return True when Streamlit is executing this file."""
    # Method A: official runtime API (works for both local & Cloud)
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            return True
    except Exception:
        pass
    # Method B: Streamlit Cloud sets this env variable
    if os.environ.get("STREAMLIT_SERVER_PORT"):
        return True
    # Method C: launched as  streamlit run main.py
    return len(sys.argv) >= 2 and "streamlit" in sys.argv[0].lower()


_STREAMLIT = _in_streamlit()


# ══════════════════════════════════════════════════════════════════════════════
# 2A.  STREAMLIT MODE  ─ full web UI
# ══════════════════════════════════════════════════════════════════════════════
if _STREAMLIT:

    import io
    import warnings
    warnings.filterwarnings("ignore")

    import streamlit as st                  # always available on Streamlit Cloud

    # ── Page config ── MUST be the VERY FIRST Streamlit call ─────────────────
    st.set_page_config(
        page_title = "FakeShield AI",
        page_icon  = "🛡️",
        layout     = "wide",
        initial_sidebar_state = "expanded",
    )

    # ── Lazy project imports (inside cached functions = safe) ─────────────────

    @st.cache_resource(show_spinner=False)
    def _cfg():
        """Load and return AppConfig."""
        from config import AppConfig          # noqa: PLC0415
        AppConfig.ensure_dirs()
        return AppConfig

    @st.cache_resource(show_spinner=False)
    def _db():
        """Return singleton AnalysisDatabase."""
        from database import AnalysisDatabase  # noqa: PLC0415
        return AnalysisDatabase()

    @st.cache_resource(show_spinner="🧠  Loading BERT text model…")
    def _text_model():
        from models.text_model import TextAnalyzer  # noqa: PLC0415
        a = TextAnalyzer()
        a.load_model()
        return a

    @st.cache_resource(show_spinner="🖼️  Loading image analysis model…")
    def _image_model():
        from models.image_model import ImageAnalyzer  # noqa: PLC0415
        a = ImageAnalyzer()
        a.load_model()
        return a

    @st.cache_resource(show_spinner=False)
    def _video_model():
        from models.video_model import VideoAnalyzer  # noqa: PLC0415
        return VideoAnalyzer()

    @st.cache_resource(show_spinner=False)
    def _ui():
        from components.ui_components import UIComponents  # noqa: PLC0415
        return UIComponents()

    @st.cache_resource(show_spinner=False)
    def _display():
        from components.results_display import ResultsDisplay  # noqa: PLC0415
        return ResultsDisplay()

    @st.cache_resource(show_spinner=False)
    def _reporter():
        from components.report_generator import ReportGenerator  # noqa: PLC0415
        return ReportGenerator()

    # ── Session-state defaults ────────────────────────────────────────────────
    _DEFAULTS = {
        "results":      None,
        "report_html":  None,
        "report_pdf":   None,
        "_fetched":     "",
    }
    for _k, _v in _DEFAULTS.items():
        st.session_state.setdefault(_k, _v)

    # ── Helper: run multi-modal analysis ─────────────────────────────────────
    def _run(text=None, image=None, video_url=None) -> dict:
        results: dict = {}

        if text and text.strip():
            with st.spinner("🧠 Analysing text with BERT…"):
                try:
                    results["text"] = _text_model().analyze(text.strip())
                except Exception as e:
                    st.warning(f"Text analysis error: {e}")

        if image is not None:
            with st.spinner("🔬 Analysing image (ELA + Grad-CAM)…"):
                try:
                    results["image"] = _image_model().analyze(image)
                except Exception as e:
                    st.warning(f"Image analysis error: {e}")

        if video_url and video_url.strip():
            with st.spinner("🎬 Downloading & analysing video frames…"):
                try:
                    results["video"] = _video_model().analyze(video_url.strip())
                except Exception as e:
                    st.warning(f"Video analysis error: {e}")

        if results:
            try:
                from utils.score_fusion import ScoreFusion  # noqa: PLC0415
                results["overall"] = ScoreFusion().fuse(results)
            except Exception as e:
                st.warning(f"Score fusion error: {e}")

        return results

    # ── Helper: save to SQLite (silent failure — DB must not crash the UI) ───
    def _save(results: dict, atype: str,
              text: str = "", video_url: str = "") -> None:
        try:
            cfg     = _cfg()
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
            pass

    # ── Helper: run + persist + rerun ────────────────────────────────────────
    def _dispatch(text=None, image=None, video_url=None, atype="text") -> None:
        r = _run(text=text, image=image, video_url=video_url)
        if r:
            _save(r, atype, text or "", video_url or "")
            st.session_state.results    = r
            st.session_state.report_html = None
            st.session_state.report_pdf  = None
            st.rerun()
        else:
            _ui().warning_panel("Analysis returned no results. Check your input.")

    # ── Fetch text from URL ───────────────────────────────────────────────────
    def _fetch_url(url: str) -> str:
        try:
            from utils.text_utils import fetch_url_content, clean_text  # noqa
            _, body = fetch_url_content(url)
            return clean_text(body) if body else ""
        except Exception:
            return ""

    def _valid_url(url: str) -> bool:
        try:
            from utils.text_utils import is_valid_url  # noqa
            return is_valid_url(url)
        except Exception:
            return url.startswith("http://") or url.startswith("https://")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE RENDER
    # ══════════════════════════════════════════════════════════════════════════

    _ui().apply_custom_css()
    _ui().render_header()

    # ── Sidebar (DB errors must not crash the page) ───────────────────────────
    try:
        _ui().render_sidebar(_db())
    except Exception as _e:
        with st.sidebar:
            st.warning(f"Sidebar error: {_e}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    _t1, _t2, _t3, _t4, _t5 = st.tabs([
        "📝 Text Analysis",
        "🖼️  Image Analysis",
        "🎬 Video Analysis",
        "🔗 Combined",
        "📊 History",
    ])

    # ───────────────────────────── TAB 1 · TEXT ──────────────────────────────
    with _t1:
        st.markdown("### Analyse a news article or text snippet")
        _ui().info_panel(
            "Paste article text <b>or</b> a URL — "
            "the article will be automatically scraped."
        )

        _mode = st.radio(
            "Input mode", ["✏️ Paste text", "🔗 Enter URL"],
            horizontal=True, label_visibility="collapsed",
        )
        _text_val = ""

        if "✏️" in _mode:
            _text_val = st.text_area(
                "Article text", height=220, label_visibility="collapsed",
                placeholder="Paste the full article or news excerpt here…",
            )
        else:
            _url_in = st.text_input(
                "Article URL", label_visibility="collapsed",
                placeholder="https://example.com/article",
            )
            if _url_in:
                if not _valid_url(_url_in):
                    _ui().warning_panel("Please enter a valid HTTP(S) URL.")
                elif st.button("🌐 Fetch Article", key="fetch_btn"):
                    with st.spinner("Scraping article…"):
                        st.session_state["_fetched"] = _fetch_url(_url_in)

            if st.session_state.get("_fetched"):
                _text_val = st.session_state["_fetched"]
                st.text_area(
                    "Fetched preview",
                    _text_val[:500] + ("…" if len(_text_val) > 500 else ""),
                    height=110, disabled=True,
                )

        if st.button("🔍 Analyze Text", key="btn_text", type="primary"):
            _wc = len(_text_val.strip().split())
            _min = _cfg().MIN_TEXT_WORDS
            if _wc < _min:
                _ui().warning_panel(f"Provide at least {_min} words (you have {_wc}).")
            else:
                _dispatch(text=_text_val, atype="text")

    # ───────────────────────────── TAB 2 · IMAGE ─────────────────────────────
    with _t2:
        st.markdown("### Analyse an image for manipulation or deepfakes")
        _ui().info_panel(
            "Supported: JPEG, PNG, WebP. "
            "Applies Error Level Analysis, noise mapping, and Grad-CAM."
        )

        _uploaded = st.file_uploader(
            "Upload image", type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
        if _uploaded:
            try:
                from PIL import Image as _PILImage  # noqa
                _pil = _PILImage.open(io.BytesIO(_uploaded.read())).convert("RGB")
            except Exception as _e:
                st.error(f"Could not open image: {_e}")
                _pil = None

            if _pil:
                _c1, _c2 = st.columns([1, 2])
                with _c1:
                    st.image(_pil, caption=_uploaded.name, use_column_width=True)
                with _c2:
                    st.markdown(
                        f"**File:** `{_uploaded.name}`  \n"
                        f"**Size:** {_pil.width} × {_pil.height} px  \n"
                        f"**Mode:** `{_pil.mode}`"
                    )
                if st.button("🔍 Analyze Image", key="btn_img", type="primary"):
                    _dispatch(image=_pil, atype="image")

    # ───────────────────────────── TAB 3 · VIDEO ─────────────────────────────
    with _t3:
        st.markdown("### Analyse video frames for manipulation")
        _max_f = _cfg().VIDEO_MAX_FRAMES
        _ui().info_panel(
            f"Supports YouTube, Vimeo, and direct MP4 links. "
            f"Up to {_max_f} key frames are extracted and analysed independently."
        )

        _vid_url = st.text_input(
            "Video URL", label_visibility="collapsed",
            placeholder="https://www.youtube.com/watch?v=…  or  direct .mp4 link",
        )
        if st.button("🔍 Analyze Video", key="btn_vid", type="primary"):
            if not _vid_url.strip():
                _ui().warning_panel("Please enter a video URL.")
            elif not _valid_url(_vid_url.strip()):
                _ui().warning_panel("Invalid URL — must start with http:// or https://")
            else:
                _dispatch(video_url=_vid_url.strip(), atype="video")

    # ───────────────────────────── TAB 4 · COMBINED ──────────────────────────
    with _t4:
        st.markdown("### Full multimodal analysis — text + image + video")
        _ui().info_panel(
            "Provide any combination of inputs. "
            "All active modalities are fused via weighted averaging."
        )

        _cl, _cr = st.columns(2)
        with _cl:
            _ct = st.text_area(
                "📝 Article text (optional)", height=150,
                placeholder="Paste article text here…",
                label_visibility="collapsed",
            )
        with _cr:
            _ci_file = st.file_uploader(
                "🖼️ Image (optional)", type=["jpg", "jpeg", "png", "webp"],
                key="comb_img", label_visibility="collapsed",
            )

        _cv = st.text_input(
            "🎬 Video URL (optional)",
            placeholder="YouTube or direct MP4 URL (optional)…",
            label_visibility="collapsed",
        )

        _has_input = bool(_ct.strip() or _ci_file or _cv.strip())
        if st.button("🔍 Run Multimodal Analysis", key="btn_comb",
                     type="primary", disabled=not _has_input):
            _ci_pil = None
            if _ci_file:
                try:
                    from PIL import Image as _PILI  # noqa
                    _ci_pil = _PILI.open(io.BytesIO(_ci_file.read())).convert("RGB")
                except Exception:
                    pass
            _dispatch(
                text      = _ct.strip() or None,
                image     = _ci_pil,
                video_url = _cv.strip() or None,
                atype     = "combined",
            )

    # ───────────────────────────── TAB 5 · HISTORY ───────────────────────────
    with _t5:
        st.markdown("### 📊 Analysis History")
        try:
            _hist  = _db().get_history(30)
            _stats = _db().get_statistics()

            _hc1, _hc2, _hc3, _hc4 = st.columns(4)
            _hc1.metric("Total",        _stats["total"])
            _hc2.metric("🔴 Fake",      _stats["fake"])
            _hc3.metric("🟢 Real",      _stats["real"])
            _hc4.metric("🟡 Suspicious",_stats["suspicious"])

            if _hist:
                import pandas as _pd  # noqa
                _df = _pd.DataFrame(_hist)[[
                    "id", "timestamp", "analysis_type",
                    "overall_label", "overall_score"
                ]]
                _df["overall_score"] = _df["overall_score"].apply(
                    lambda x: f"{x:.1%}" if isinstance(x, (int, float)) else "–"
                )
                _df.columns = ["ID", "Timestamp", "Type", "Verdict", "Score"]
                st.dataframe(_df, use_container_width=True, hide_index=True)

                if st.button("🗑️ Clear All History", key="clr_hist"):
                    _db().clear_all()
                    st.success("History cleared.")
                    st.rerun()
            else:
                _ui().info_panel("No analyses yet — run your first analysis above!")

        except Exception as _he:
            st.error(f"History error: {_he}")

    # ── Results panel (persists across tab switches) ──────────────────────────
    if st.session_state.results:
        st.divider()
        st.markdown("## 📊 Analysis Results")

        try:
            _display().render_all(st.session_state.results, _ui())
        except Exception as _re:
            st.error(f"Results display error: {_re}")
            st.json({
                k: v for k, v in st.session_state.results.items()
                if k not in ("_raw",) and not hasattr(v, "save")
            })

        st.divider()
        _ui().section_title("📥 Download Report")
        _rc1, _rc2, _rc3 = st.columns([1, 1, 2])

        with _rc1:
            if st.button("📄 Generate HTML Report", key="gen_html"):
                try:
                    st.session_state.report_html = _reporter().generate_html(
                        st.session_state.results
                    )
                except Exception as _e:
                    st.error(f"HTML report error: {_e}")

            if st.session_state.report_html:
                _html = st.session_state.report_html
                st.download_button(
                    "⬇️ Download HTML",
                    data   = _html.encode() if isinstance(_html, str) else _html,
                    file_name = "fakeshield_report.html",
                    mime   = "text/html",
                )

        with _rc2:
            if st.button("📑 Generate PDF Report", key="gen_pdf"):
                try:
                    _pdf = _reporter().generate_pdf(st.session_state.results)
                    if _pdf:
                        st.session_state.report_pdf = _pdf
                    else:
                        _ui().warning_panel("PDF needs `reportlab` — pip install reportlab")
                except Exception as _e:
                    st.error(f"PDF error: {_e}")

            if st.session_state.get("report_pdf"):
                st.download_button(
                    "⬇️ Download PDF",
                    data      = st.session_state.report_pdf,
                    file_name = "fakeshield_report.pdf",
                    mime      = "application/pdf",
                )

        with _rc3:
            if st.button("🔄 Start New Analysis", key="btn_reset"):
                for _k in ("results", "report_html", "report_pdf", "_fetched"):
                    st.session_state[_k] = _DEFAULTS.get(_k)
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 2B.  CLI / LAUNCHER MODE  ─ only reached when NOT running inside Streamlit
# ══════════════════════════════════════════════════════════════════════════════
else:

    import argparse
    import json
    import logging
    import subprocess

    logging.basicConfig(
        format  = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
        level   = logging.INFO,
    )
    _log = logging.getLogger("main")

    # ── Argument parser ───────────────────────────────────────────────────────
    def _build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            prog        = "main.py",
            description = "🛡️  FakeShield AI — Multimodal Fake News Detection",
            formatter_class = argparse.RawDescriptionHelpFormatter,
            epilog = """
EXAMPLES
────────
  python main.py --text "BREAKING: Shocking deep-state revelation!!!"
  python main.py --text "Viral photo proves..." --image evidence.jpg
  python main.py --text article.txt --image img.jpg --video https://youtu.be/xxx
  python main.py --text https://example.com/article         # auto-scraped
  python main.py --batch inputs.json --output results.json
  python main.py --web                                      # Streamlit UI
  python main.py --web --mode api --port 8000               # FastAPI
  python main.py --warmup                                   # pre-load models
""",
        )
        p.add_argument("--web",    action="store_true",
                       help="Launch web interface (Streamlit or FastAPI)")
        p.add_argument("--mode",   choices=["streamlit","api"], default="streamlit")
        p.add_argument("--host",   default="0.0.0.0")
        p.add_argument("--port",   type=int, default=None)
        p.add_argument("--reload", action="store_true",
                       help="Hot-reload (FastAPI only)")

        g = p.add_argument_group("analysis")
        g.add_argument("--text",   metavar="TEXT_OR_URL")
        g.add_argument("--image",  metavar="PATH")
        g.add_argument("--video",  metavar="URL")
        g.add_argument("--batch",  metavar="JSON",
                       help="JSON file with list of input dicts")
        g.add_argument("--output", metavar="JSON",
                       help="Write JSON results here (default: stdout)")

        p.add_argument("--no-db",   action="store_true")
        p.add_argument("--warmup",  action="store_true")
        p.add_argument("--verbose", "-v", action="store_true")
        return p

    # ── Launchers ─────────────────────────────────────────────────────────────
    def _launch_streamlit(host: str, port: int) -> None:
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(_ROOT / "main.py"),
            "--server.address",  host,
            "--server.port",     str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ]
        _log.info("Streamlit → http://%s:%d", host, port)
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            pass
        except FileNotFoundError:
            _log.error("streamlit not installed — pip install streamlit")
            sys.exit(1)

    def _launch_api(host: str, port: int, reload: bool) -> None:
        try:
            import uvicorn  # noqa
        except ImportError:
            _log.error("uvicorn not installed — pip install uvicorn[standard]")
            sys.exit(1)
        _log.info("FastAPI → http://%s:%d", host, port)
        import uvicorn  # noqa
        uvicorn.run("api:app", host=host, port=port,
                    reload=reload, log_level="info")

    # ── CLI analysis ──────────────────────────────────────────────────────────
    def _run_cli(args: argparse.Namespace) -> None:
        from inference import FakeNewsInference  # noqa: PLC0415
        engine = FakeNewsInference.get_instance()

        if args.batch:
            with open(args.batch) as fh:
                items = json.load(fh)
            if not isinstance(items, list):
                _log.error("Batch JSON must be a list")
                sys.exit(1)
            results = engine.predict_batch(items, save_to_db=not args.no_db)
            _write(_clean(results), args.output)
            return

        if not any([args.text, args.image, args.video]):
            _log.error("No inputs — use --text, --image, or --video")
            sys.exit(1)

        if args.image and not Path(args.image).is_file():
            _log.error("Image file not found: %s", args.image)
            sys.exit(1)

        result = engine.predict(
            text      = args.text,
            image     = args.image,
            video_url = args.video,
            save_to_db= not args.no_db,
        )
        _print_result(result)
        _write(_clean(result), args.output)

    def _print_result(r: dict) -> None:
        v  = r.get("verdict",     "?")
        fs = r.get("final_score", 0.5)
        c  = r.get("confidence",  0.0)
        em = {"FAKE":"🔴","REAL":"🟢","SUSPICIOUS":"🟡"}.get(v, "⚪")
        bar= "█" * int(fs * 40) + "░" * (40 - int(fs * 40))
        print(f"\n{'═'*58}")
        print(f"  {em}  Verdict:    {v}")
        print(f"     Fake score: {fs:.1%}  [{bar}]")
        print(f"     Confidence: {c:.1%}")
        for mod, ind in r.get("individual", {}).items():
            s = ind.get("fake_score", 0)
            print(f"     {mod.capitalize():<8}: {s:.1%}  ({ind.get('label','?')})")
        print(f"  ⏱  {r.get('meta',{}).get('elapsed_sec',0):.2f}s")
        print(f"{'═'*58}\n")

    def _clean(obj):
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()
                    if k != "_raw" and not hasattr(v, "save")}
        return obj

    def _write(data, path: str | None) -> None:
        out = json.dumps(data, indent=2, default=str)
        if path:
            Path(path).write_text(out)
            _log.info("Results → %s", path)
        else:
            print(out)

    # ── Warm-up ───────────────────────────────────────────────────────────────
    def _warmup() -> None:
        from inference import FakeNewsInference  # noqa
        _log.info("Pre-loading models…")
        e = FakeNewsInference.get_instance()
        e.warm_up(("text", "image"))
        print("\n✅ Models loaded.")
        print(f"   Load times: {e._load_times}")

    # ── Entry point ───────────────────────────────────────────────────────────
    def main() -> None:
        from config import AppConfig  # noqa
        AppConfig.ensure_dirs()

        args = _build_parser().parse_args()
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
ENDOFFILE

echo "✅ main.py written — $(wc -l < main.py) lines"
