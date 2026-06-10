"""
components/ui_components.py — Custom Streamlit UI Components
=============================================================
Provides:
  • Global CSS injection (dark-mode, professional theme)
  • Header / hero section
  • Sidebar navigation & stats
  • Metric cards, badge chips, info panels
  • Loading animations
"""

from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AppConfig


class UIComponents:
    """All reusable front-end building blocks for FakeShield AI."""

    # ── Global CSS ────────────────────────────────────────────────────────────
    def apply_custom_css(self):
        """Inject the global dark-theme stylesheet."""
        st.markdown(f"""
        <style>
        /* ── Global reset ────────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background: {AppConfig.COLOR_BG_DARK};
            color: {AppConfig.COLOR_TEXT};
        }}

        /* ── Sidebar ─────────────────────────────────────────────────── */
        section[data-testid="stSidebar"] {{
            background: #161b27;
            border-right: 1px solid #2d3548;
        }}

        /* ── Tabs ────────────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: #161b27;
            border-radius: 12px;
            padding: 4px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            color: #9ca3af;
            font-weight: 500;
            padding: 8px 18px;
        }}
        .stTabs [aria-selected="true"] {{
            background: {AppConfig.COLOR_PRIMARY} !important;
            color: #fff !important;
        }}

        /* ── Buttons ─────────────────────────────────────────────────── */
        .stButton > button {{
            background: linear-gradient(135deg, {AppConfig.COLOR_PRIMARY}, #6366f1);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1rem;
            padding: 12px 28px;
            transition: all .25s ease;
            width: 100%;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(79,139,249,.45);
        }}

        /* ── Text inputs & text areas ────────────────────────────────── */
        .stTextInput input, .stTextArea textarea {{
            background: #1e2130 !important;
            border: 1.5px solid #2d3548 !important;
            border-radius: 10px !important;
            color: #f0f0f0 !important;
            font-size: .95rem !important;
        }}
        .stTextInput input:focus, .stTextArea textarea:focus {{
            border-color: {AppConfig.COLOR_PRIMARY} !important;
            box-shadow: 0 0 0 3px rgba(79,139,249,.2) !important;
        }}

        /* ── File uploader ───────────────────────────────────────────── */
        .stFileUploader {{
            background: #1e2130;
            border: 2px dashed #2d3548;
            border-radius: 12px;
            padding: 16px;
        }}

        /* ── Cards ───────────────────────────────────────────────────── */
        .metric-card {{
            background: {AppConfig.COLOR_BG_CARD};
            border: 1px solid #2d3548;
            border-radius: 14px;
            padding: 20px 24px;
            text-align: center;
            transition: transform .2s;
        }}
        .metric-card:hover {{ transform: translateY(-3px); }}
        .metric-value {{
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1;
        }}
        .metric-label {{
            font-size: .8rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: .06em;
            margin-top: 4px;
        }}
        .metric-sub {{
            font-size: .85rem;
            color: #6b7280;
            margin-top: 6px;
        }}

        /* ── Credibility badge ───────────────────────────────────────── */
        .badge-fake        {{ background:#ff4444;color:#fff;border-radius:99px;padding:6px 20px;font-weight:700;font-size:1.1rem;letter-spacing:.05em; }}
        .badge-suspicious  {{ background:#ff8800;color:#fff;border-radius:99px;padding:6px 20px;font-weight:700;font-size:1.1rem;letter-spacing:.05em; }}
        .badge-real        {{ background:#00c851;color:#fff;border-radius:99px;padding:6px 20px;font-weight:700;font-size:1.1rem;letter-spacing:.05em; }}
        .badge-uncertain   {{ background:#9e9e9e;color:#fff;border-radius:99px;padding:6px 20px;font-weight:700;font-size:1.1rem;letter-spacing:.05em; }}

        /* ── Section header ──────────────────────────────────────────── */
        .section-title {{
            font-size: 1.35rem;
            font-weight: 700;
            color: #f0f0f0;
            border-left: 4px solid {AppConfig.COLOR_PRIMARY};
            padding-left: 12px;
            margin: 24px 0 16px;
        }}

        /* ── Info / warning panels ───────────────────────────────────── */
        .info-panel {{
            background: #1e2a3a;
            border: 1px solid #2d4a6a;
            border-left: 4px solid {AppConfig.COLOR_PRIMARY};
            border-radius: 8px;
            padding: 14px 18px;
            font-size: .9rem;
            color: #c8d8f0;
        }}
        .warning-panel {{
            background: #2a1e10;
            border: 1px solid #4a3010;
            border-left: 4px solid #ff8800;
            border-radius: 8px;
            padding: 14px 18px;
            font-size: .9rem;
            color: #f0d0a0;
        }}
        .success-panel {{
            background: #10231a;
            border: 1px solid #104a28;
            border-left: 4px solid #00c851;
            border-radius: 8px;
            padding: 14px 18px;
            font-size: .9rem;
            color: #a0f0c0;
        }}

        /* ── Word importance highlight ───────────────────────────────── */
        .word-fake  {{ background: rgba(255,68,68,.35); border-radius:3px;
                       padding:1px 3px; font-weight:600; color:#ff6666; }}
        .word-real  {{ background: rgba(0,200,81,.25);  border-radius:3px;
                       padding:1px 3px; font-weight:600; color:#66cc88; }}

        /* ── Hero ────────────────────────────────────────────────────── */
        .hero {{
            background: {AppConfig.HEADER_GRADIENT};
            border-radius: 18px;
            padding: 42px 36px;
            text-align: center;
            margin-bottom: 28px;
            border: 1px solid #2d3548;
        }}
        .hero h1 {{
            font-size: 2.8rem;
            font-weight: 800;
            margin: 0;
            background: linear-gradient(90deg,#4f8bf9,#a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .hero p {{
            font-size: 1.05rem;
            color:#9ca3af;
            margin: 10px 0 0;
        }}

        /* ── Divider ─────────────────────────────────────────────────── */
        .divider {{
            height:1px;
            background:linear-gradient(90deg,transparent,#2d3548,transparent);
            margin:24px 0;
        }}

        /* ── Scrollbar ───────────────────────────────────────────────── */
        ::-webkit-scrollbar {{ width:6px; }}
        ::-webkit-scrollbar-track {{ background:#161b27; }}
        ::-webkit-scrollbar-thumb {{ background:#2d3548; border-radius:3px; }}
        </style>
        """, unsafe_allow_html=True)

    # ── Hero / Header ────────────────────────────────────────────────────────
    def render_header(self):
        st.markdown("""
        <div class="hero">
            <h1>🛡️ FakeShield AI</h1>
            <p>Multimodal Fake News Detection &amp; Credibility Analysis System</p>
            <p style="font-size:.8rem;color:#555;margin-top:8px;">
                Powered by BERT · EfficientNet · Grad-CAM · Error Level Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def render_sidebar(self, db):
        with st.sidebar:
            st.markdown(f"## {AppConfig.APP_ICON} {AppConfig.APP_NAME}")
            st.markdown(f"*v{AppConfig.APP_VERSION}*")
            st.divider()

            # Stats from DB
            try:
                stats = db.get_statistics()
                st.markdown("### 📊 Session Stats")
                col1, col2 = st.columns(2)
                col1.metric("Total Analyses", stats["total"])
                col2.metric("Fake Detected", stats["fake"])
                col1.metric("Real Content",   stats["real"])
                col2.metric("Suspicious",      stats["suspicious"])
            except Exception:
                pass

            st.divider()
            st.markdown("### ⚙️ Settings")
            st.session_state.setdefault("show_ela",     True)
            st.session_state.setdefault("show_gradcam", True)
            st.session_state.setdefault("show_noise",   False)
            st.session_state["show_ela"]     = st.checkbox("Show ELA map",    value=st.session_state["show_ela"])
            st.session_state["show_gradcam"] = st.checkbox("Show Grad-CAM",   value=st.session_state["show_gradcam"])
            st.session_state["show_noise"]   = st.checkbox("Show noise map",  value=st.session_state["show_noise"])

            st.divider()
            st.markdown("### ℹ️ About")
            st.markdown("""
            **FakeShield AI** is a Final Year Project
            implementing multimodal fake-news detection
            using state-of-the-art deep learning.

            **Techniques Used:**
            - 🤗 BERT / RoBERTa (text)
            - 🖼️ EfficientNet + ELA (image)
            - 🎬 yt-dlp + OpenCV (video)
            - 🔍 Grad-CAM explainability
            - 📄 Automated PDF reports
            """)

    # ── Metric card ───────────────────────────────────────────────────────────
    def metric_card(self, value: str, label: str,
                    sub: str = "", color: str = AppConfig.COLOR_PRIMARY):
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{color}">{value}</div>
            <div class="metric-label">{label}</div>
            {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
        </div>
        """, unsafe_allow_html=True)

    # ── Credibility badge ─────────────────────────────────────────────────────
    def credibility_badge(self, label: str):
        cls = {
            "FAKE":       "badge-fake",
            "SUSPICIOUS": "badge-suspicious",
            "REAL":       "badge-real",
        }.get(label.upper(), "badge-uncertain")
        emoji = {"FAKE": "🔴", "SUSPICIOUS": "🟡",
                 "REAL": "🟢"}.get(label.upper(), "⚪")
        st.markdown(
            f'<div style="text-align:center;margin:12px 0;">'
            f'<span class="{cls}">{emoji} {label}</span></div>',
            unsafe_allow_html=True,
        )

    # ── Section title ─────────────────────────────────────────────────────────
    def section_title(self, title: str):
        st.markdown(f'<div class="section-title">{title}</div>',
                    unsafe_allow_html=True)

    # ── Info panels ───────────────────────────────────────────────────────────
    def info_panel(self, msg: str):
        st.markdown(f'<div class="info-panel">ℹ️ {msg}</div>',
                    unsafe_allow_html=True)

    def warning_panel(self, msg: str):
        st.markdown(f'<div class="warning-panel">⚠️ {msg}</div>',
                    unsafe_allow_html=True)

    def success_panel(self, msg: str):
        st.markdown(f'<div class="success-panel">✅ {msg}</div>',
                    unsafe_allow_html=True)

    # ── Divider ───────────────────────────────────────────────────────────────
    def divider(self):
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Word highlighting ─────────────────────────────────────────────────────
    def highlighted_text(self, text: str,
                         word_importance: list[tuple]) -> str:
        """
        Return HTML string with fake-leaning words in red, credibility
        words in green.
        """
        if not word_importance:
            return f"<p>{text}</p>"

        imp_dict = {w: s for w, s in word_importance}
        words    = text.split()
        parts    = []

        for w in words:
            clean = w.lower().strip(".,!?;:\"'()")
            score = imp_dict.get(clean)
            if score and score > 0.3:
                parts.append(f'<span class="word-fake">{w}</span>')
            elif score and score < -0.3:
                parts.append(f'<span class="word-real">{w}</span>')
            else:
                parts.append(w)

        return "<p style='line-height:1.9;font-size:.95rem'>" + " ".join(parts) + "</p>"
