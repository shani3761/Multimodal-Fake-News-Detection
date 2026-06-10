"""
components/results_display.py — Analysis Results Renderer
==========================================================
Renders all result sections:
  • Overall credibility gauge (Plotly)
  • Per-modality score cards
  • Text word-importance chart
  • Image ELA / Grad-CAM panels
  • Video frame timeline
  • Explanation accordion
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AppConfig
from utils.score_fusion import get_label_color, get_label_emoji


class ResultsDisplay:
    """Renders all result panels for FakeShield AI."""

    # ── Master results renderer ──────────────────────────────────────────────
    def render_all(self, results: dict, ui):
        """
        Render the complete results section.

        Parameters
        ----------
        results : dict   Full result dict (text, image, video, overall keys)
        ui      : UIComponents instance (for styled helpers)
        """
        overall = results.get("overall", {})
        if not overall:
            ui.warning_panel("No analysis results to display.")
            return

        # ── Hero banner ──────────────────────────────────────────────────────
        self._render_overall_banner(overall, ui)
        ui.divider()

        # ── Score breakdown ──────────────────────────────────────────────────
        ui.section_title("📊 Modality Scores")
        self._render_score_cards(results, ui)
        ui.divider()

        # ── Gauge + confidence ring ──────────────────────────────────────────
        g_col, e_col = st.columns([1, 1])
        with g_col:
            ui.section_title("🎯 Credibility Gauge")
            self._render_gauge(overall)
        with e_col:
            ui.section_title("📝 Analysis Summary")
            if overall.get("explanation"):
                st.markdown(
                    f"<div class='info-panel'>{overall['explanation'].replace(chr(10), '<br>')}</div>",
                    unsafe_allow_html=True,
                )
        ui.divider()

        # ── Per-modality detail sections ─────────────────────────────────────
        if "text" in results:
            with st.expander("📝 Text Analysis Detail", expanded=True):
                self._render_text_detail(results["text"], ui)

        if "image" in results:
            with st.expander("🖼️ Image Analysis Detail", expanded=True):
                self._render_image_detail(results["image"])

        if "video" in results:
            with st.expander("🎬 Video Analysis Detail", expanded=True):
                self._render_video_detail(results["video"], ui)

    # ── Overall banner ───────────────────────────────────────────────────────
    def _render_overall_banner(self, overall: dict, ui):
        label      = overall.get("overall_label", "UNCERTAIN")
        score      = overall.get("overall_score", 0.5)
        confidence = overall.get("confidence", 0.0)
        color      = get_label_color(label)
        emoji      = get_label_emoji(label)

        st.markdown(f"""
        <div style="background:{AppConfig.COLOR_BG_CARD};border:1px solid {color};
                    border-radius:18px;padding:32px;text-align:center;margin:16px 0;
                    box-shadow:0 4px 24px rgba(0,0,0,.4)">
            <div style="font-size:3.5rem;margin-bottom:8px">{emoji}</div>
            <div style="font-size:2rem;font-weight:800;color:{color};
                        letter-spacing:.04em">{label}</div>
            <div style="font-size:1rem;color:#9ca3af;margin-top:6px">
                Fake probability: <b style="color:{color}">{score:.1%}</b>
                &nbsp;·&nbsp;
                Confidence: <b style="color:#ccc">{confidence:.1%}</b>
            </div>
            <div style="font-size:.8rem;color:#555;margin-top:8px">
                Modalities analysed: {", ".join(overall.get("modalities", [])) or "N/A"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Score cards ──────────────────────────────────────────────────────────
    def _render_score_cards(self, results: dict, ui):
        cols = st.columns(max(len([k for k in ("text","image","video","overall")
                                   if k in results]), 1))
        pairs = [
            ("text",    "📝 Text",   "text_model"),
            ("image",   "🖼️ Image",  "image_model"),
            ("video",   "🎬 Video",  "video_model"),
            ("overall", "🔗 Overall","fusion"),
        ]
        idx = 0
        for key, name, _ in pairs:
            if key not in results:
                continue
            r      = results[key]
            score  = r.get("overall_score" if key == "overall" else "fake_score", 0.5)
            label  = r.get("overall_label" if key == "overall" else "label", "UNCERTAIN")
            color  = get_label_color(label)
            with cols[idx]:
                ui.metric_card(
                    value=f"{score:.1%}",
                    label=f"{name} Fake Score",
                    sub=label,
                    color=color,
                )
            idx += 1

    # ── Plotly gauge ─────────────────────────────────────────────────────────
    def _render_gauge(self, overall: dict):
        try:
            import plotly.graph_objects as go

            score = overall.get("overall_score", 0.5) * 100
            label = overall.get("overall_label", "UNCERTAIN")
            color = get_label_color(label)

            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = score,
                number= {"suffix": "%", "font": {"size": 36, "color": color}},
                title = {"text": "Fake Probability",
                         "font": {"size": 16, "color": "#9ca3af"}},
                gauge = {
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1,
                        "tickcolor": "#444",
                        "tickfont": {"color": "#aaa"},
                    },
                    "bar":   {"color": color, "thickness": .25},
                    "bgcolor": "#1e2130",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  40], "color": "#0a2318"},
                        {"range": [40, 60], "color": "#2a1e10"},
                        {"range": [60, 100],"color": "#2a0e0e"},
                    ],
                    "threshold": {
                        "line":      {"color": color, "width": 4},
                        "thickness": .8,
                        "value":     score,
                    },
                },
            ))
            fig.update_layout(
                paper_bgcolor="#161b27",
                plot_bgcolor ="#161b27",
                height=280,
                margin={"t": 30, "b": 10, "l": 20, "r": 20},
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            score = overall.get("overall_score", 0.5)
            st.progress(score)
            st.write(f"Fake probability: **{score:.1%}**")

    # ── Text detail ──────────────────────────────────────────────────────────
    def _render_text_detail(self, text_result: dict, ui):
        st.markdown(f"**Model:** `{text_result.get('model_used','N/A')}`")

        # Score bar chart
        try:
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x     = ["Fake Probability", "Real Probability"],
                y     = [text_result.get("fake_score", 0),
                         text_result.get("real_score", 0)],
                marker_color = [AppConfig.COLOR_FAKE, AppConfig.COLOR_REAL],
                text  = [f"{text_result.get('fake_score',0):.1%}",
                         f"{text_result.get('real_score',0):.1%}"],
                textposition = "auto",
            ))
            fig.update_layout(
                title     = "Text Classification Scores",
                paper_bgcolor="#161b27", plot_bgcolor="#161b27",
                font={"color":"#ccc"}, height=250,
                margin={"t":40,"b":10,"l":0,"r":0},
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            c1, c2 = st.columns(2)
            c1.metric("Fake", f"{text_result.get('fake_score',0):.1%}")
            c2.metric("Real", f"{text_result.get('real_score',0):.1%}")

        # Word importance
        wi = text_result.get("word_importance", [])
        if wi:
            st.markdown("**Top Influential Words**")
            try:
                import plotly.graph_objects as go
                words, scores = zip(*wi[:15])
                colours = [AppConfig.COLOR_FAKE if s > 0 else AppConfig.COLOR_REAL
                           for s in scores]
                fig2 = go.Figure(go.Bar(
                    x=list(scores), y=list(words),
                    orientation="h",
                    marker_color=colours,
                ))
                fig2.update_layout(
                    paper_bgcolor="#161b27", plot_bgcolor="#161b27",
                    font={"color":"#ccc"}, height=350,
                    xaxis_title="Importance (+ = fake, – = credible)",
                    margin={"t":10,"b":30,"l":0,"r":0},
                )
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass

        # Highlighted text
        highlighted = ui.highlighted_text(
            text_result.get("analyzed_text", ""),
            wi,
        )
        with st.expander("View highlighted text"):
            st.markdown(highlighted, unsafe_allow_html=True)

        # Sensationalism
        sens = text_result.get("sensationalism", 0)
        if sens:
            st.markdown(f"**Sensationalism score:** `{sens:.1%}`")

        # Credibility cues
        cues = text_result.get("credibility_cues", [])
        if cues:
            st.markdown("**Credibility cues found:** " +
                        ", ".join(f"`{c}`" for c in cues))

        # Explanation
        if text_result.get("explanation"):
            ui.info_panel(text_result["explanation"])

    # ── Image detail ─────────────────────────────────────────────────────────
    def _render_image_detail(self, img_result: dict):
        scores = img_result.get("scores_detail", {})
        if scores:
            try:
                import plotly.graph_objects as go
                labels = ["ELA", "Noise", "Clone Detect", "Deep Features"]
                vals   = [scores.get("ela",0), scores.get("noise",0),
                          scores.get("clone",0), scores.get("deep_feat",0)]
                colors = [
                    AppConfig.COLOR_FAKE if v >= AppConfig.THRESHOLD_FAKE
                    else (AppConfig.COLOR_SUSPICIOUS
                          if v >= AppConfig.THRESHOLD_SUSPICIOUS
                          else AppConfig.COLOR_REAL)
                    for v in vals
                ]
                fig = go.Figure(go.Bar(
                    x=labels, y=vals, marker_color=colors,
                    text=[f"{v:.1%}" for v in vals], textposition="auto",
                ))
                fig.update_layout(
                    title="Per-Technique Manipulation Scores",
                    paper_bgcolor="#161b27", plot_bgcolor="#161b27",
                    font={"color":"#ccc"}, height=280,
                    yaxis={"range":[0,1]},
                    margin={"t":40,"b":10,"l":0,"r":0},
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.write(scores)

        # Image panels
        col1, col2, col3 = st.columns(3)
        show_ela     = st.session_state.get("show_ela",     True)
        show_gradcam = st.session_state.get("show_gradcam", True)
        show_noise   = st.session_state.get("show_noise",   False)

        if show_ela and img_result.get("ela_image"):
            with col1:
                st.markdown("**ELA Map**")
                st.image(img_result["ela_image"],
                         caption="Bright = high error level (possible edit)",
                         use_column_width=True)
        if show_gradcam and img_result.get("gradcam_image"):
            with col2:
                st.markdown("**Grad-CAM Saliency**")
                st.image(img_result["gradcam_image"],
                         caption="Red = model attention / suspicious region",
                         use_column_width=True)
        if show_noise and img_result.get("noise_image"):
            with col3:
                st.markdown("**Noise Map**")
                st.image(img_result["noise_image"],
                         caption="High-frequency residual noise pattern",
                         use_column_width=True)

        if img_result.get("explanation"):
            st.markdown(f"> {img_result['explanation']}")

    # ── Video detail ─────────────────────────────────────────────────────────
    def _render_video_detail(self, vid_result: dict, ui):
        frame_scores = vid_result.get("frame_scores", [])
        frames       = vid_result.get("frame_images", [])

        st.markdown(
            f"**Frames analysed:** {vid_result.get('frames_analyzed', 0)} | "
            f"**FPS:** {vid_result.get('fps', 0):.1f}"
        )

        if frame_scores:
            try:
                import plotly.graph_objects as go
                colors = [
                    AppConfig.COLOR_FAKE if s >= AppConfig.THRESHOLD_FAKE
                    else (AppConfig.COLOR_SUSPICIOUS
                          if s >= AppConfig.THRESHOLD_SUSPICIOUS
                          else AppConfig.COLOR_REAL)
                    for s in frame_scores
                ]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[f"Frame {i+1}" for i in range(len(frame_scores))],
                    y=frame_scores,
                    marker_color=colors,
                    text=[f"{s:.1%}" for s in frame_scores],
                    textposition="auto",
                ))
                fig.add_hline(y=AppConfig.THRESHOLD_FAKE,
                              line_dash="dot", line_color=AppConfig.COLOR_FAKE,
                              annotation_text="FAKE threshold")
                fig.add_hline(y=AppConfig.THRESHOLD_SUSPICIOUS,
                              line_dash="dot", line_color=AppConfig.COLOR_SUSPICIOUS,
                              annotation_text="SUSPICIOUS threshold")
                fig.update_layout(
                    title="Per-Frame Manipulation Score",
                    paper_bgcolor="#161b27", plot_bgcolor="#161b27",
                    font={"color":"#ccc"}, height=280,
                    yaxis={"range":[0,1]},
                    margin={"t":40,"b":10,"l":0,"r":0},
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

        if frames:
            st.markdown("**Extracted Key Frames**")
            n_cols = min(len(frames), 4)
            cols   = st.columns(n_cols)
            for i, (frame, score) in enumerate(zip(frames, frame_scores)):
                color = (AppConfig.COLOR_FAKE   if score >= AppConfig.THRESHOLD_FAKE
                         else AppConfig.COLOR_SUSPICIOUS
                         if score >= AppConfig.THRESHOLD_SUSPICIOUS
                         else AppConfig.COLOR_REAL)
                with cols[i % n_cols]:
                    st.image(frame, use_column_width=True)
                    st.markdown(
                        f"<small style='color:{color}'>Frame {i+1}: {score:.1%}</small>",
                        unsafe_allow_html=True,
                    )

        if vid_result.get("explanation"):
            ui.info_panel(vid_result["explanation"])
