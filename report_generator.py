"""
components/report_generator.py — Analysis Report Generator
============================================================
Creates downloadable HTML and PDF reports from analysis results.

Usage:
    gen = ReportGenerator()
    html_bytes = gen.generate_html(results, timestamp)
    pdf_bytes  = gen.generate_pdf(results, timestamp)   # requires reportlab
"""

from __future__ import annotations

import base64
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AppConfig
from utils.score_fusion import get_label_color, get_label_emoji


class ReportGenerator:
    """Generates HTML and PDF analysis reports."""

    # ── HTML report ──────────────────────────────────────────────────────────
    def generate_html(self, results: dict, timestamp: Optional[str] = None) -> bytes:
        """
        Build a self-contained HTML report from analysis results.

        Returns
        -------
        bytes  UTF-8 encoded HTML.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        overall = results.get("overall", {})
        label   = overall.get("overall_label", "UNCERTAIN")
        score   = overall.get("overall_score", 0.5)
        color   = get_label_color(label)
        emoji   = get_label_emoji(label)
        conf    = overall.get("confidence", 0.0)

        # Individual scores block
        individual_rows = ""
        for mod in ("text", "image", "video"):
            if mod not in results:
                continue
            r      = results[mod]
            ms     = r.get("fake_score", 0)
            ml     = r.get("label", "N/A")
            mc     = get_label_color(ml)
            individual_rows += f"""
            <tr>
              <td><b>{mod.capitalize()}</b></td>
              <td style="color:{mc};font-weight:700">{ml}</td>
              <td>{ms:.1%}</td>
              <td>{r.get("confidence", 0):.1%}</td>
            </tr>"""

        # Explanation text
        text_explanation = ""
        if "text" in results:
            tr = results["text"]
            text_explanation = f"""
            <div class="section">
              <h3>📝 Text Analysis</h3>
              <p><b>Model:</b> {tr.get("model_used","N/A")}</p>
              <p><b>Fake score:</b> {tr.get("fake_score",0):.1%} &nbsp;|&nbsp;
                 <b>Real score:</b> {tr.get("real_score",0):.1%}</p>
              <p><b>Sensationalism index:</b> {tr.get("sensationalism",0):.1%}</p>
              <p>{tr.get("explanation","")}</p>
              {"<p><b>Credibility cues:</b> " + ", ".join(tr.get("credibility_cues",[]))+"</p>" if tr.get("credibility_cues") else ""}
            </div>"""

        image_explanation = ""
        if "image" in results:
            ir = results["image"]
            sd = ir.get("scores_detail", {})
            image_explanation = f"""
            <div class="section">
              <h3>🖼️ Image Analysis</h3>
              <p><b>ELA score:</b> {sd.get("ela",0):.1%} &nbsp;|&nbsp;
                 <b>Noise score:</b> {sd.get("noise",0):.1%} &nbsp;|&nbsp;
                 <b>Clone score:</b> {sd.get("clone",0):.1%} &nbsp;|&nbsp;
                 <b>Deep feature score:</b> {sd.get("deep_feat",0):.1%}</p>
              <p>{ir.get("explanation","")}</p>
            </div>"""

        video_explanation = ""
        if "video" in results:
            vr = results["video"]
            video_explanation = f"""
            <div class="section">
              <h3>🎬 Video Analysis</h3>
              <p><b>Frames analysed:</b> {vr.get("frames_analyzed",0)} &nbsp;|&nbsp;
                 <b>FPS:</b> {vr.get("fps",0):.1f}</p>
              <p>{vr.get("explanation","")}</p>
            </div>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FakeShield AI — Analysis Report</title>
<style>
  body {{
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #0e1117;
    color: #e0e0e0;
    margin: 0; padding: 0;
  }}
  .container {{ max-width: 900px; margin: 0 auto; padding: 32px 24px; }}
  .header {{
    background: linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
    border-radius: 16px;
    padding: 36px;
    text-align: center;
    margin-bottom: 28px;
    border: 1px solid #2d3548;
  }}
  .header h1 {{
    font-size: 2.2rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg,#4f8bf9,#a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .header p {{ color: #9ca3af; margin: 8px 0 0; }}
  .verdict-card {{
    background: #1e2130;
    border: 2px solid {color};
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin-bottom: 24px;
  }}
  .verdict-emoji {{ font-size: 3.5rem; }}
  .verdict-label {{
    font-size: 2.2rem; font-weight: 800;
    color: {color}; letter-spacing: .06em;
  }}
  .verdict-sub {{ color: #9ca3af; margin-top: 8px; font-size: .95rem; }}
  .score-bar-wrap {{
    background: #161b27; border-radius: 99px;
    height: 12px; margin: 12px 0 4px; overflow: hidden;
  }}
  .score-bar {{
    height: 100%; border-radius: 99px;
    background: {color};
    width: {score*100:.1f}%;
    transition: width 1s ease;
  }}
  .section {{
    background: #1e2130;
    border: 1px solid #2d3548;
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 18px;
  }}
  .section h3 {{ margin: 0 0 14px; color: #e0e0e0; font-size: 1.15rem; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #161b27; color: #9ca3af; text-align: left;
        padding: 10px 14px; font-size: .8rem; text-transform: uppercase;
        letter-spacing: .06em; }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #2d3548;
        font-size: .95rem; }}
  .footer {{
    text-align: center; color: #555; font-size: .8rem; margin-top: 36px;
    border-top: 1px solid #2d3548; padding-top: 16px;
  }}
  .badge {{
    display: inline-block; padding: 4px 14px; border-radius: 99px;
    font-size: .8rem; font-weight: 700; background: {color};
    color: #fff; margin-left: 8px;
  }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>🛡️ FakeShield AI</h1>
    <p>Multimodal Fake News Detection Report</p>
    <p style="font-size:.8rem;color:#555">{timestamp}</p>
  </div>

  <!-- Verdict -->
  <div class="verdict-card">
    <div class="verdict-emoji">{emoji}</div>
    <div class="verdict-label">{label}</div>
    <div class="verdict-sub">
      Fake probability: <b style="color:{color}">{score:.1%}</b> &nbsp;·&nbsp;
      Confidence: <b>{conf:.1%}</b>
    </div>
    <div class="score-bar-wrap"><div class="score-bar"></div></div>
    <small style="color:#555">{score:.1%} fake probability</small>
  </div>

  <!-- Modality scores -->
  <div class="section">
    <h3>📊 Modality Breakdown</h3>
    <table>
      <tr>
        <th>Modality</th><th>Verdict</th>
        <th>Fake Score</th><th>Confidence</th>
      </tr>
      {individual_rows}
      <tr style="font-weight:700">
        <td>🔗 Overall (Fusion)</td>
        <td style="color:{color}">{label}</td>
        <td>{score:.1%}</td>
        <td>{conf:.1%}</td>
      </tr>
    </table>
  </div>

  <!-- Weights used -->
  {"" if not overall.get("weights_used") else _weights_section(overall["weights_used"])}

  <!-- Per-modality explanations -->
  {text_explanation}
  {image_explanation}
  {video_explanation}

  <!-- Overall explanation -->
  <div class="section">
    <h3>🔍 Fusion Explanation</h3>
    <p style="white-space:pre-line">{overall.get("explanation","N/A")}</p>
  </div>

  <!-- Methodology -->
  <div class="section">
    <h3>🧪 Methodology</h3>
    <p>
      <b>Text:</b> BERT/RoBERTa fine-tuned on LIAR dataset, combined with
      sensationalism keyword analysis.<br>
      <b>Image:</b> Error Level Analysis (ELA), noise-pattern analysis,
      copy-move detection heuristic, and EfficientNet-B0 deep-feature anomaly
      scoring with Grad-CAM explainability.<br>
      <b>Video:</b> yt-dlp download, scene-aware keyframe extraction via
      OpenCV, per-frame image analysis.<br>
      <b>Fusion:</b> Weighted average — Text {AppConfig.WEIGHT_TEXT:.0%},
      Image {AppConfig.WEIGHT_IMAGE:.0%}, Video {AppConfig.WEIGHT_VIDEO:.0%}
      (re-normalised for absent modalities).
    </p>
  </div>

  <!-- Disclaimer -->
  <div class="section" style="border-color:#4a3010">
    <h3>⚠️ Disclaimer</h3>
    <p style="color:#aaa;font-size:.88rem">
      This report is generated automatically by an AI system and is intended
      for research and educational purposes only.  Results should not be used
      as the sole basis for editorial or legal decisions.  Always verify
      information with multiple credible sources.
    </p>
  </div>

  <div class="footer">
    Generated by <b>FakeShield AI v{AppConfig.APP_VERSION}</b>
    &nbsp;·&nbsp; {AppConfig.APP_AUTHOR}
    &nbsp;·&nbsp; {timestamp}
  </div>

</div>
</body>
</html>"""
        return html.encode("utf-8")

    # ── PDF report ───────────────────────────────────────────────────────────
    def generate_pdf(self, results: dict, timestamp: Optional[str] = None) -> Optional[bytes]:
        """
        Generate a PDF report using ReportLab.

        Returns
        -------
        bytes of PDF content, or None if ReportLab is not installed.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table,
                TableStyle, HRFlowable,
            )
            from reportlab.lib.enums import TA_CENTER, TA_LEFT

        except ImportError:
            return None

        buf    = io.BytesIO()
        doc    = SimpleDocTemplate(buf, pagesize=A4,
                                   leftMargin=2*cm, rightMargin=2*cm,
                                   topMargin=2*cm,  bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # ── Custom styles ────────────────────────────────────────────────────
        title_style = ParagraphStyle(
            "Title2", parent=styles["Title"],
            fontSize=24, spaceAfter=6, textColor=colors.HexColor("#4F8BF9"),
        )
        h2_style = ParagraphStyle(
            "H2", parent=styles["Heading2"],
            fontSize=14, textColor=colors.HexColor("#E0E0E0"),
            spaceAfter=6, spaceBefore=14,
        )
        body_style = ParagraphStyle(
            "Body2", parent=styles["Normal"],
            fontSize=10, textColor=colors.HexColor("#C0C0C0"),
            spaceAfter=4,
        )
        center_style = ParagraphStyle(
            "Center", parent=body_style, alignment=TA_CENTER,
        )

        overall = results.get("overall", {})
        label   = overall.get("overall_label", "UNCERTAIN")
        score   = overall.get("overall_score", 0.5)
        conf    = overall.get("confidence", 0.0)
        color   = colors.HexColor(get_label_color(label))
        emoji   = get_label_emoji(label)

        # ── Title ────────────────────────────────────────────────────────────
        story.append(Paragraph("🛡️ FakeShield AI — Analysis Report", title_style))
        story.append(Paragraph(
            f"<font color='#9ca3af'>{AppConfig.APP_TAGLINE}</font>", center_style
        ))
        story.append(Paragraph(
            f"<font color='#555555' size='9'>{timestamp}</font>", center_style
        ))
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(color=colors.HexColor("#2d3548"), thickness=1))

        # ── Verdict ──────────────────────────────────────────────────────────
        story.append(Spacer(1, 0.3*cm))
        verdict_style = ParagraphStyle(
            "Verdict", parent=styles["Normal"],
            fontSize=20, alignment=TA_CENTER, textColor=color, fontName="Helvetica-Bold",
        )
        story.append(Paragraph(f"{emoji}  {label}", verdict_style))
        story.append(Paragraph(
            f"Fake probability: <b>{score:.1%}</b> &nbsp;|&nbsp; Confidence: <b>{conf:.1%}</b>",
            center_style,
        ))
        story.append(Spacer(1, 0.4*cm))

        # ── Score table ──────────────────────────────────────────────────────
        story.append(Paragraph("Modality Breakdown", h2_style))
        tbl_data = [["Modality", "Verdict", "Fake Score", "Confidence"]]
        for mod in ("text", "image", "video"):
            if mod not in results:
                continue
            r = results[mod]
            tbl_data.append([
                mod.capitalize(),
                r.get("label", "N/A"),
                f"{r.get('fake_score',0):.1%}",
                f"{r.get('confidence',0):.1%}",
            ])
        tbl_data.append(["Overall (Fusion)", label,
                          f"{score:.1%}", f"{conf:.1%}"])

        tbl = Table(tbl_data, colWidths=[4*cm, 3.5*cm, 3.5*cm, 3.5*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#161b27")),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.HexColor("#9ca3af")),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0),  9),
            ("BACKGROUND",  (0, 1), (-1, -2), colors.HexColor("#1e2130")),
            ("BACKGROUND",  (0, -1),(-1, -1), colors.HexColor("#0f3460")),
            ("TEXTCOLOR",   (0, 1), (-1, -1), colors.HexColor("#e0e0e0")),
            ("FONTSIZE",    (0, 1), (-1, -1), 10),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#2d3548")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2),
             [colors.HexColor("#1e2130"), colors.HexColor("#181d2b")]),
            ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",  (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING",(0,0),(-1,-1),   7),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.4*cm))

        # ── Explanations ─────────────────────────────────────────────────────
        for mod in ("text", "image", "video", "overall"):
            if mod not in results:
                continue
            r = results[mod]
            expl = r.get("explanation") or r.get("explanation_text", "")
            if not expl:
                continue
            story.append(Paragraph(f"{mod.capitalize()} Explanation", h2_style))
            story.append(Paragraph(expl.replace("\n","<br/>"), body_style))

        story.append(Spacer(1, 0.4*cm))
        story.append(HRFlowable(color=colors.HexColor("#2d3548"), thickness=1))

        # ── Disclaimer ───────────────────────────────────────────────────────
        disclaimer = ParagraphStyle(
            "Disc", parent=styles["Normal"],
            fontSize=8, textColor=colors.HexColor("#888888"), spaceAfter=4,
        )
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(
            "DISCLAIMER: This report is generated by an automated AI system for "
            "research and educational purposes only. Results should be verified "
            "against multiple credible sources before editorial or legal use.",
            disclaimer,
        ))
        story.append(Paragraph(
            f"Generated by FakeShield AI v{AppConfig.APP_VERSION} · "
            f"{AppConfig.APP_AUTHOR} · {timestamp}",
            disclaimer,
        ))

        doc.build(story)
        return buf.getvalue()

    # ── Streamlit download helper ─────────────────────────────────────────────
    def streamlit_download_buttons(self, results: dict):
        """Render HTML and PDF download buttons in a Streamlit context."""
        import streamlit as st

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fname     = datetime.now().strftime("fakeshield_report_%Y%m%d_%H%M%S")

        col1, col2 = st.columns(2)

        with col1:
            html_bytes = self.generate_html(results, timestamp)
            st.download_button(
                label="📄 Download HTML Report",
                data=html_bytes,
                file_name=f"{fname}.html",
                mime="text/html",
                use_container_width=True,
            )

        with col2:
            pdf_bytes = self.generate_pdf(results, timestamp)
            if pdf_bytes:
                st.download_button(
                    label="📑 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{fname}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.button("📑 PDF (install reportlab)",
                          disabled=True, use_container_width=True)


# ── Module helpers ────────────────────────────────────────────────────────────
def _weights_section(weights: dict) -> str:
    rows = "".join(
        f"<tr><td>{k.capitalize()}</td><td>{v:.1%}</td></tr>"
        for k, v in weights.items()
    )
    return f"""
    <div class="section">
      <h3>⚖️ Fusion Weights Used</h3>
      <table>
        <tr><th>Modality</th><th>Weight</th></tr>
        {rows}
      </table>
    </div>"""
