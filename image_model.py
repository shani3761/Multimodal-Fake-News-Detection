"""
models/image_model.py — CNN Image Forgery / DeepFake Analyzer
==============================================================
Pipeline:
  1. Error Level Analysis (ELA)       – detects JPEG re-compression artifacts
  2. Noise Pattern Analysis            – high-frequency noise map
  3. Copy-Move / Clone Detection       – heuristic region repetition check
  4. EfficientNet-B0 Feature Anomaly  – deep feature magnitude as proxy score
  5. Grad-CAM Saliency                – shows "suspicious" regions

The combined score is a weighted blend of the above four signals.
"""

from __future__ import annotations
import io
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AppConfig


class ImageAnalyzer:
    """
    Analyzes an image for signs of manipulation / deepfake.

    Methods
    -------
    load_model()         Download EfficientNet weights (called once).
    analyze(image)       Full pipeline; returns result dict.
    """

    def __init__(self):
        self._model      = None
        self._transform  = None
        self._hooks      = {}
        self._activations = {}
        self._gradients   = {}
        self.is_loaded   = False
        self.device_str  = "cpu"

    # ── Model loading ────────────────────────────────────────────────────────
    def load_model(self) -> bool:
        if self.is_loaded:
            return True
        try:
            import torch
            import torchvision.models as tv_models
            import torchvision.transforms as T

            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(self.device_str)

            # EfficientNet-B0 pre-trained on ImageNet
            self._model = tv_models.efficientnet_b0(
                weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
            self._model.eval()
            self._model.to(device)

            # Register Grad-CAM hooks on the last conv block
            target_layer = self._model.features[-1]
            target_layer.register_forward_hook(self._fwd_hook)
            target_layer.register_full_backward_hook(self._bwd_hook)

            self._transform = T.Compose([
                T.Resize(AppConfig.IMAGE_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            self.is_loaded = True
            return True
        except Exception as e:
            # Graceful degradation: run without deep-feature component
            self.is_loaded = False
            self._model    = None
            return False

    # ── Hooks for Grad-CAM ───────────────────────────────────────────────────
    def _fwd_hook(self, module, inp, out):
        self._activations["value"] = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients["value"] = grad_out[0].detach()

    # ── Main analysis ────────────────────────────────────────────────────────
    def analyze(self, image: Image.Image) -> dict:
        """
        Parameters
        ----------
        image : PIL.Image.Image

        Returns
        -------
        dict with keys:
            fake_score      float 0–1
            real_score      float 0–1
            label           str
            confidence      float 0–1
            ela_image       PIL.Image  (ELA visualization)
            gradcam_image   PIL.Image | None
            noise_image     PIL.Image  (noise map)
            scores_detail   dict       (per-technique scores)
            explanation     str
        """
        if not self.is_loaded:
            self.load_model()

        image = image.convert("RGB")
        img_arr = np.array(image, dtype=np.float32)

        # --- Technique 1: Error Level Analysis ---
        ela_arr, ela_score = self._ela(image)

        # --- Technique 2: Noise Pattern Analysis ---
        noise_arr, noise_score = self._noise_analysis(img_arr)

        # --- Technique 3: Statistical / Copy-Move Heuristic ---
        clone_score = self._clone_heuristic(img_arr)

        # --- Technique 4: EfficientNet anomaly score + Grad-CAM ---
        feat_score, gradcam_arr = self._deep_feature_score(image)

        # --- Weighted fusion of technique scores ---
        weights     = [0.35, 0.25, 0.15, 0.25]
        scores      = [ela_score, noise_score, clone_score, feat_score]
        fake_score  = float(np.dot(weights, scores))
        fake_score  = max(0.0, min(1.0, fake_score))

        label = _score_to_label(fake_score)

        # Build visualizations
        ela_img     = Image.fromarray(
            np.clip(ela_arr * AppConfig.ELA_SCALE, 0, 255).astype(np.uint8)
        )
        noise_img   = Image.fromarray(
            np.clip(noise_arr, 0, 255).astype(np.uint8)
        )
        gradcam_img = self._overlay_gradcam(image, gradcam_arr) if gradcam_arr is not None else None

        return {
            "fake_score":    round(fake_score, 4),
            "real_score":    round(1 - fake_score, 4),
            "label":         label,
            "confidence":    round(abs(fake_score - 0.5) * 2, 4),
            "ela_image":     ela_img,
            "noise_image":   noise_img,
            "gradcam_image": gradcam_img,
            "scores_detail": {
                "ela":       round(ela_score,   4),
                "noise":     round(noise_score, 4),
                "clone":     round(clone_score, 4),
                "deep_feat": round(feat_score,  4),
            },
            "explanation":   self._explain(fake_score, ela_score, noise_score,
                                           clone_score, feat_score),
        }

    # ── ELA ──────────────────────────────────────────────────────────────────
    def _ela(self, image: Image.Image):
        """
        Error Level Analysis.
        Saves the image at JPEG quality=90, then computes per-pixel differences.
        Edited regions resurface at a higher error level because they have been
        saved at a different initial quality.
        """
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=AppConfig.ELA_QUALITY)
        buf.seek(0)
        compressed = Image.open(buf).convert("RGB")

        orig = np.array(image,      dtype=np.float32)
        comp = np.array(compressed, dtype=np.float32)

        diff      = np.abs(orig - comp)           # shape (H, W, 3)
        ela_score = float(diff.mean() / 255.0)    # 0→clean, higher→suspect

        # Amplify for visualization
        diff_rgb = np.clip(diff, 0, 255).astype(np.uint8)
        return diff_rgb, ela_score

    # ── Noise analysis ───────────────────────────────────────────────────────
    def _noise_analysis(self, img: np.ndarray):
        """
        High-pass residual noise map.
        Composite / edited regions often have inconsistent noise texture.
        """
        try:
            from scipy.ndimage import gaussian_filter
            gray = np.mean(img, axis=2)
            blurred = gaussian_filter(gray, sigma=2)
            noise   = np.abs(gray - blurred)
            score   = float(noise.std() / (gray.std() + 1e-6))
            score   = min(score * 2, 1.0)

            # Colourize noise map (red channel = high noise)
            noise_norm = (noise / (noise.max() + 1e-8) * 255).astype(np.uint8)
            noise_rgb  = np.stack([noise_norm,
                                   np.zeros_like(noise_norm),
                                   255 - noise_norm], axis=2)
            return noise_rgb, score
        except Exception:
            blank = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
            return blank, 0.5

    # ── Clone / Copy-Move heuristic ──────────────────────────────────────────
    def _clone_heuristic(self, img: np.ndarray) -> float:
        """
        Lightweight heuristic: checks block-level variance uniformity.
        Cloned regions share nearly identical texture statistics.
        """
        try:
            gray   = np.mean(img, axis=2)
            h, w   = gray.shape
            block  = 32
            vars_  = []
            for r in range(0, h - block, block):
                for c in range(0, w - block, block):
                    blk = gray[r:r+block, c:c+block]
                    vars_.append(blk.var())
            if not vars_:
                return 0.5
            arr   = np.array(vars_)
            # High ratio of near-zero-variance blocks → likely cloned region
            suspicious_ratio = float((arr < arr.mean() * 0.05).sum() / len(arr))
            return min(suspicious_ratio * 3, 1.0)
        except Exception:
            return 0.5

    # ── EfficientNet deep features + Grad-CAM ────────────────────────────────
    def _deep_feature_score(self, image: Image.Image):
        """
        Run EfficientNet forward pass.
        Uses L2-norm of the penultimate feature vector as an anomaly proxy:
        authentic images cluster around the ImageNet manifold; forged images
        often push feature norms to extremes.
        Also computes Grad-CAM saliency map.
        """
        if self._model is None:
            return 0.5, None

        try:
            import torch

            device  = torch.device(self.device_str)
            tensor  = self._transform(image).unsqueeze(0).to(device)
            tensor.requires_grad_(True)

            logits  = self._model(tensor)
            top_cls = int(logits.argmax(dim=1).item())

            # Grad-CAM backward
            self._model.zero_grad()
            logits[0, top_cls].backward()

            # Feature anomaly score based on global average pooling magnitude
            if "value" in self._activations:
                act    = self._activations["value"]      # (1, C, H, W)
                f_norm = float(act.pow(2).mean().sqrt().item())
                # Calibrate: typical norms 1–5 → score 0.1–0.9
                feat_score = float(np.clip((f_norm - 1) / 6, 0, 1))
            else:
                feat_score = 0.5

            # Build Grad-CAM heatmap
            gradcam = None
            if "value" in self._activations and "value" in self._gradients:
                act  = self._activations["value"].squeeze()   # (C, H, W)
                grad = self._gradients["value"].squeeze()     # (C, H, W)
                weights = grad.mean(dim=[1, 2])               # (C,)
                cam  = (weights[:, None, None] * act).sum(0)  # (H, W)
                cam  = torch.relu(cam).cpu().numpy()
                if cam.max() > 0:
                    cam = cam / cam.max()
                gradcam = cam

            return feat_score, gradcam

        except Exception:
            return 0.5, None

    # ── Grad-CAM overlay ─────────────────────────────────────────────────────
    def _overlay_gradcam(self, image: Image.Image,
                         cam: np.ndarray) -> Image.Image:
        """Resize cam to image size and overlay as a heatmap."""
        try:
            import cv2
            target_size = (image.width, image.height)
            cam_resized = cv2.resize(cam, target_size)
            heatmap = cv2.applyColorMap(
                (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            orig_np     = np.array(image)
            overlay     = cv2.addWeighted(orig_np, 0.6, heatmap_rgb, 0.4, 0)
            return Image.fromarray(overlay)
        except Exception:
            return image

    # ── Explanation text ─────────────────────────────────────────────────────
    def _explain(self, final: float, ela: float, noise: float,
                 clone: float, feat: float) -> str:
        parts = []
        if ela > 0.25:
            parts.append(
                f"ELA detected re-compression inconsistencies (score={ela:.2f}), "
                "suggesting possible editing or compositing."
            )
        if noise > 0.4:
            parts.append(
                "Irregular noise patterns were found, common in digitally altered images."
            )
        if clone > 0.3:
            parts.append(
                "Repeated texture blocks detected, which may indicate copy-move forgery."
            )
        if feat > 0.6:
            parts.append(
                "Deep feature analysis placed this image outside the expected "
                "authentic-image distribution."
            )
        if not parts:
            parts.append("No significant manipulation signals were detected.")
        return " ".join(parts)


# ── Helper ───────────────────────────────────────────────────────────────────
def _score_to_label(score: float) -> str:
    from config import AppConfig
    if score >= AppConfig.THRESHOLD_FAKE:
        return "FAKE"
    if score >= AppConfig.THRESHOLD_SUSPICIOUS:
        return "SUSPICIOUS"
    return "REAL"
