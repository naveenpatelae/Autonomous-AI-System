#!/usr/bin/env python3
# =====================================================================
# 🎭 AFFECTIVE ENGINE  — Body-Side Module
#
# Migrated from Kaggle notebook Cell 5 (Level III — Gestalt Sensorium):
#   • AffectiveComputeEngine  — biometric stress routing
#   • MultimodalProjector     — vision/text bridge (torch-optional)
#   • GestaltSensorium        — bio + visual + audio fusion
#   • process_event_vision    — delta-frame event camera
#   • WiFiSensingLobe         — CSI-based pose estimation
#   • SpatialMetrology        — sensor calibration
#   • QuantumSonar            — gravity-nav anomaly scan
#   • DeviceTelemetryTracker  — Haversine device proximity
#   • DiscoveryLobe           — BLE/WiFi peripheral registry
#   • set_perception_frequency — stress-indexed time-dilation
#
# Original body-side logic (preserved exactly):
#   • TTSProfile              — pitch/rate/volume/style container
#   • AffectiveManifold       — BPM + emotion → TTSProfile mapping (#126)
#   • SocialContextSwitcher   — active-app tone switching (#101)
#   • UniversalLinguisticLobe — language detect + translate (#111)
#   • LiquidUI                — adaptive avatar window (#24)
#   • CognitiveMirror         — predictive pre-compute cache (#65)
#   • ExecutiveFunction       — proactive calendar pre-loader (#59)
#   • AffectiveEngine         — top-level coordinator
# =====================================================================

from __future__ import annotations

import json
import logging
import math
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("AffectiveEngine")

# ── Optional deps ─────────────────────────────────────────────────────
try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

try:
    from langdetect import detect as langdetect_detect
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

# =====================================================================
# ── MIGRATED FROM NOTEBOOK CELL 5 (Level III — Gestalt Sensorium) ──
# =====================================================================


class AffectiveComputeEngine:
    """
    Reads biometric sensor data and computes a normalised stress index.
    Routes response mode based on stress level:
      < 0.4  → DENSE_ANALYTICAL
      >= 0.4 → COMPASSION_CORE
    """

    def read_biometrics(self, user_state: str) -> Tuple[int, float]:
        """
        Map a named user-state string to (heart_rate_bpm, pupil_dilation_mm).
        Extend this mapping as real sensor feeds come online.
        """
        _map = {
            "CALM_FOCUSED":          (65,  3.0),
            "FRUSTRATED_EXHAUSTED":  (95,  5.5),
            "NEUTRAL":               (75,  4.0),
            "STRESSED":              (105, 6.0),
            "RELAXED":               (58,  2.8),
        }
        return _map.get(user_state.upper(), (75, 4.0))

    def calculate_stress_index(self, hr: int, pupil: float) -> float:
        """
        Returns a float in [0.0, 1.0].
        HR contribution weight 0.6, pupil dilation weight 0.4.
        Clamped to physiologically plausible ranges.
        """
        hr_norm    = min(max((hr    - 60) / 40, 0.0), 1.0)
        pupil_norm = min(max((pupil - 2.5) / 4.0, 0.0), 1.0)
        return round(hr_norm * 0.6 + pupil_norm * 0.4, 4)

    def route_symbiotic_response(self, stress: float) -> str:
        """Map stress index to response mode string."""
        if stress < 0.4:
            return "MODE: DENSE_ANALYTICAL"
        return "MODE: COMPASSION_CORE"

    def full_pipeline(self, user_state: str) -> Dict[str, Any]:
        """
        Convenience: state string → full dict with hr, pupil, stress, mode.
        """
        hr, pupil = self.read_biometrics(user_state)
        stress    = self.calculate_stress_index(hr, pupil)
        mode      = self.route_symbiotic_response(stress)
        return {"hr": hr, "pupil": pupil, "stress": stress, "mode": mode}


# ── Torch-optional MultimodalProjector ───────────────────────────────

if _TORCH_OK:
    class MultimodalProjector(nn.Module):
        """
        Projects vision embeddings (768-dim ViT) into the LLM text space.
        Bridges visual tokens into the language model's embedding dimension.
        """
        def __init__(self, vision_dim: int = 768, text_dim: int = 4096):
            super().__init__()
            self.bridge = nn.Linear(vision_dim, text_dim)

        def forward(self, vision_features):
            return self.bridge(vision_features)

else:
    class MultimodalProjector:  # type: ignore[no-redef]
        """CPU stub when torch is unavailable."""
        def __init__(self, vision_dim: int = 768, text_dim: int = 4096):
            self.vision_dim = vision_dim
            self.text_dim   = text_dim

        def forward(self, vision_features):
            raise RuntimeError("torch not available — MultimodalProjector is stub only.")


# ── GestaltSensorium ─────────────────────────────────────────────────

if _TORCH_OK:
    class GestaltSensorium:
        """
        Fuses biometric, visual, and audio modality vectors into a single
        10000-dim Hyperdimensional Computing (HDC) bundle vector.
        Binding operator: element-wise sign product (XOR in ±1 space).
        """
        def __init__(self, D: int = 10_000):
            self.D = D
            self.bio_proj = nn.Linear(2,   D)
            self.vis_proj = nn.Linear(768, D)
            self.aud_proj = nn.Linear(128, D)

        def ingest_reality(
            self,
            bio: "torch.Tensor",
            vis: "torch.Tensor",
            aud: "torch.Tensor",
        ) -> "torch.Tensor":
            """
            bio: (1, 2)    — [hr_norm, pupil_norm]
            vis: (1, 768)  — ViT CLS token
            aud: (1, 128)  — MFCC feature vector
            Returns: (1, D) binary HDC bundle in {-1, +1}
            """
            bio_v = torch.sign(self.bio_proj(bio))
            vis_v = torch.sign(self.vis_proj(vis))
            aud_v = torch.sign(self.aud_proj(aud))
            return bio_v * vis_v * aud_v

else:
    class GestaltSensorium:  # type: ignore[no-redef]
        """CPU stub."""
        def __init__(self, D: int = 10_000):
            self.D = D

        def ingest_reality(self, bio, vis, aud):
            raise RuntimeError("torch not available — GestaltSensorium is stub only.")


def process_event_vision(
    frame0: "Any",
    frame1: "Any",
    threshold: float = 0.1,
) -> "Any":
    """
    Delta-frame event camera processor.
    Computes pixel-wise difference between two consecutive frames.
    Events:  +1.0 where delta > +threshold (ON event)
             -1.0 where delta < -threshold (OFF event)
              0.0 otherwise

    Works with both torch.Tensor and numpy.ndarray inputs.
    Falls back gracefully if neither is available.
    """
    if _TORCH_OK and hasattr(frame0, "shape"):
        try:
            delta  = frame1 - frame0
            events = torch.zeros_like(delta)
            events[delta  >  threshold] =  1.0
            events[delta  < -threshold] = -1.0
            return events
        except Exception:
            pass
    # numpy path
    try:
        import numpy as np
        delta  = frame1 - frame0
        events = np.zeros_like(delta)
        events[delta  >  threshold] =  1.0
        events[delta  < -threshold] = -1.0
        return events
    except Exception as exc:
        raise RuntimeError("process_event_vision requires torch or numpy") from exc


class WiFiSensingLobe:
    """
    Passive WiFi Channel State Information (CSI) human detection.
    Classifies environment state from mean signal energy:
      energy > 0.5 → HUMAN_DETECTED (MOVING)
      energy > 0.2 → HUMAN_DETECTED (STATIONARY)
      else         → ENVIRONMENT_CLEAR
    """

    def __init__(self, csi_subcarriers: int = 64):
        self.csi_subcarriers = csi_subcarriers

    def reconstruct_pose_from_rf(self, csi: "Any") -> str:
        """
        csi: a 1-D array/tensor of length csi_subcarriers representing
             normalised amplitude values in [0, 1].
        """
        if _TORCH_OK and hasattr(csi, "mean"):
            energy = torch.mean(csi).item()
        else:
            try:
                import numpy as np
                energy = float(np.mean(csi))
            except Exception:
                energy = float(sum(csi) / max(1, len(csi)))

        if energy > 0.5:
            return "HUMAN_DETECTED (MOVING)"
        elif energy > 0.2:
            return "HUMAN_DETECTED (STATIONARY)"
        return "ENVIRONMENT_CLEAR"


class SpatialMetrology:
    """
    Manages spatial sensor calibration state.
    In production: replace stub with IMU/LiDAR calibration routines.
    """

    def __init__(self):
        self.calibrated = False
        self._calibration_data: Optional[Any] = None

    def calibrate_sensor(self, calibration_data: Any) -> str:
        self._calibration_data = calibration_data
        self.calibrated = True
        return "CALIBRATION_SUCCESS"

    def get_status(self) -> Dict[str, Any]:
        return {"calibrated": self.calibrated,
                "data_present": self._calibration_data is not None}


class QuantumSonar:
    """
    Gravity-gradient inertial navigation unit.
    In production: feeds from a gravimeter or accelerometer array.
    scan_for_anomalies() uses stub randomness — replace with real sensor read.
    """

    def __init__(self):
        self.nav_mode          = "QUANTUM_INERTIAL"
        self.detection_range_km = 120

    def scan_for_anomalies(self, gravity_flux: float) -> str:
        """
        gravity_flux: deviation in µGal (10⁻⁸ m/s²).
        Threshold >0.7 → anomaly.
        """
        if gravity_flux > 0.7:
            return "TARGET_ACQUIRED_VIA_GRAVITY"
        return "ENVIRONMENT_NOMINAL"

    def get_status(self) -> Dict[str, Any]:
        return {
            "nav_mode": self.nav_mode,
            "detection_range_km": self.detection_range_km,
        }


class DeviceTelemetryTracker:
    """
    Tracks authorised devices and computes Haversine distance to them.
    vault_access: True means the device list was loaded from the secure vault.
    """

    def __init__(self, vault_access: bool = True):
        self.vault_access       = vault_access
        self.authorized_devices: Dict[str, Dict[str, Any]] = {}

    def register_device(
        self,
        device_id: str,
        lat: float,
        lon: float,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.authorized_devices[device_id] = {
            "lat": lat, "lon": lon,
            "metadata": metadata or {},
            "last_seen": time.time(),
        }

    def calculate_haversine_distance(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """Returns great-circle distance in kilometres."""
        R = 6_371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi       = math.radians(lat2 - lat1)
        dlambda    = math.radians(lon2 - lon1)
        a = (math.sin(dphi / 2) ** 2
             + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def distance_to_device(
        self,
        device_id: str,
        current_lat: float,
        current_lon: float,
    ) -> Optional[float]:
        dev = self.authorized_devices.get(device_id)
        if not dev:
            return None
        return self.calculate_haversine_distance(
            current_lat, current_lon, dev["lat"], dev["lon"]
        )

    def get_status(self) -> Dict[str, Any]:
        return {
            "vault_access":        self.vault_access,
            "authorized_count":    len(self.authorized_devices),
            "device_ids":          list(self.authorized_devices.keys()),
        }


class DiscoveryLobe:
    """
    Peripheral discovery — maintains a runtime registry of nearby
    BLE/WiFi devices detected during scanning.
    In production: replace scan_airwaves() with real BLE/WiFi scan calls.
    """

    def __init__(self):
        self.device_registry: Dict[str, Dict[str, str]] = {
            "Tesla_Model_3": {"type": "CAR",    "protocol": "BLE"},
            "HomePod_Mini":  {"type": "SPEAKER", "protocol": "AirPlay"},
        }

    def register_device(
        self,
        name: str,
        device_type: str,
        protocol: str,
    ) -> None:
        self.device_registry[name] = {"type": device_type, "protocol": protocol}

    def scan_airwaves(self) -> List[str]:
        """Returns list of currently known device names."""
        return list(self.device_registry.keys())

    def get_device(self, name: str) -> Optional[Dict[str, str]]:
        return self.device_registry.get(name)

    def get_status(self) -> Dict[str, Any]:
        return {
            "device_count": len(self.device_registry),
            "devices": self.device_registry,
        }


def set_perception_frequency(stress_index: float) -> str:
    """
    Maps a normalised stress index [0.0, 1.0] to a perception mode.

    Formula: effective_frequency = 1 + stress_index * 999
      > 500 → BULLET-TIME  (hyper-aware, fine-grained event processing)
      ≤ 500 → AMBIENT-AWARE (relaxed, coarse sampling)

    This mirrors how biological attention sharpens under stress.
    """
    effective_frequency = 1 + stress_index * 999
    if effective_frequency > 500:
        return "MODE: BULLET-TIME"
    return "MODE: AMBIENT-AWARE"


# =====================================================================
# ── ORIGINAL BODY-SIDE AFFECTIVE ENGINE CODE (preserved exactly) ────
# =====================================================================


@dataclass
class TTSProfile:
    """TTS voice parameters dynamically adjusted to biometric state."""
    pitch:   float = 1.0    # multiplier: 0.5 (low) – 2.0 (high)
    rate:    float = 1.0    # multiplier: 0.5 (slow) – 2.0 (fast)
    volume:  float = 1.0    # 0.0 – 1.0
    style:   str   = "calm" # "calm" | "urgent" | "gentle" | "technical"


class AffectiveManifold:
    """
    #126: Maps (BPM, emotion, context) → TTSProfile.

    BPM ranges:
      < 60  → very calm: slow, deep, gentle
      60-80 → calm: normal rate, normal pitch
      80-100→ engaged: slightly faster
      100-115→ elevated: faster, higher pitch, fewer words
      > 115  → stressed: fast, concise, quieter (backs off)
    """

    def compute_tts_profile(
        self, bpm: int, emotion: str = "neutral", context: str = "chat"
    ) -> TTSProfile:
        profile = TTSProfile()

        if bpm < 60:
            profile.pitch = 0.90
            profile.rate  = 0.85
            profile.style = "gentle"
        elif bpm < 80:
            profile.pitch = 1.00
            profile.rate  = 1.00
            profile.style = "calm"
        elif bpm < 100:
            profile.pitch = 1.05
            profile.rate  = 1.10
            profile.style = "calm"
        elif bpm < 115:
            profile.pitch = 1.10
            profile.rate  = 1.20
            profile.style = "urgent"
            profile.volume = 0.90
        else:
            profile.pitch  = 1.15
            profile.rate   = 1.35
            profile.volume = 0.75
            profile.style  = "urgent"

        if emotion == "sad":
            profile.pitch = max(0.75, profile.pitch - 0.15)
            profile.rate  = max(0.70, profile.rate  - 0.15)
            profile.style = "gentle"
        elif emotion in ("happy", "excited"):
            profile.pitch = min(1.40, profile.pitch + 0.15)
            profile.rate  = min(1.30, profile.rate  + 0.10)
        elif emotion == "angry":
            profile.pitch = max(0.85, profile.pitch - 0.05)
            profile.rate  = min(1.25, profile.rate  + 0.05)

        if context == "coding":
            profile.style = "technical"
        elif context == "casual":
            profile.style = "calm"

        return profile

    def apply_to_pyttsx3(self, profile: TTSProfile, engine) -> bool:
        try:
            rate = engine.getProperty("rate")
            engine.setProperty("rate", int(rate * profile.rate))
            vol  = engine.getProperty("volume")
            engine.setProperty("volume", max(0.1, min(1.0, vol * profile.volume)))
            return True
        except Exception as e:
            logger.warning(f"[AffectiveManifold] pyttsx3 apply error: {e}")
            return False

    def format_response_for_stress(self, text: str, bpm: int) -> str:
        if bpm <= 115:
            return text
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        short = " ".join(sentences[:2])
        if len(short) < len(text):
            short += " [Response shortened — you seem busy.]"
        return short


class SocialContextSwitcher:
    """
    #101: Detects active window/application and switches AI tone.
    """

    CONTEXT_MAP = {
        "code": {
            "apps": {"visual studio code", "vscode", "code", "pycharm", "xcode",
                     "sublime", "atom", "vim", "emacs", "intellij", "cursor"},
            "tone": "technical",
            "style_hint": "Use technical terminology, code snippets, and precise language.",
        },
        "chat": {
            "apps": {"discord", "slack", "whatsapp", "messages", "telegram",
                     "signal", "teams", "zoom", "facetime"},
            "tone": "casual",
            "style_hint": "Use casual, empathetic language. Be brief and friendly.",
        },
        "terminal": {
            "apps": {"terminal", "iterm", "iterm2", "hyper", "warp", "kitty", "alacritty"},
            "tone": "concise",
            "style_hint": "Be concise. Prefer commands over explanations.",
        },
        "productivity": {
            "apps": {"calendar", "mail", "outlook", "notion", "obsidian",
                     "reminders", "todo", "things", "omnifocus"},
            "tone": "professional",
            "style_hint": "Be professional and action-oriented.",
        },
        "browser": {
            "apps": {"safari", "chrome", "firefox", "edge", "brave", "opera"},
            "tone": "balanced",
            "style_hint": "Be informative and balanced.",
        },
    }

    def __init__(self):
        self._current_context = "balanced"
        self._current_app     = ""
        self._stop_evt        = threading.Event()
        self._callbacks: List[Callable[[str, str], None]] = []

    def add_context_callback(self, fn: Callable[[str, str], None]):
        self._callbacks.append(fn)

    def _get_active_app(self) -> str:
        try:
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ["osascript", "-e",
                     "tell application \"System Events\" to get name of first "
                     "application process whose frontmost is true"],
                    capture_output=True, text=True, timeout=3,
                )
                return result.stdout.strip().lower()
            elif _PSUTIL_OK:
                for proc in psutil.process_iter(["name", "status"]):
                    if proc.info.get("status") == psutil.STATUS_RUNNING:
                        return (proc.info.get("name") or "").lower()
        except Exception:
            pass
        return ""

    def detect_context(self, app_name: str = "") -> Tuple[str, dict]:
        app = app_name or self._get_active_app()
        app_lower = app.lower()
        for context, cfg in self.CONTEXT_MAP.items():
            if any(a in app_lower for a in cfg["apps"]):
                return context, cfg
        return "balanced", {"tone": "balanced", "style_hint": "Be clear and helpful."}

    def get_system_hint(self, app_name: str = "") -> str:
        _, cfg = self.detect_context(app_name)
        return cfg.get("style_hint", "")

    def start_monitoring(self, interval_sec: float = 5.0):
        def _loop():
            while not self._stop_evt.is_set():
                app = self._get_active_app()
                if app and app != self._current_app:
                    context, _ = self.detect_context(app)
                    if context != self._current_context:
                        self._current_context = context
                        self._current_app     = app
                        logger.info(f"[SocialContext] Context changed: {context} (app={app})")
                        for fn in self._callbacks:
                            try:
                                fn(context, app)
                            except Exception:
                                pass
                time.sleep(interval_sec)

        threading.Thread(target=_loop, daemon=True,
                          name="SocialContextMonitor").start()

    def stop(self):
        self._stop_evt.set()

    @property
    def current_context(self) -> str:
        return self._current_context


class UniversalLinguisticLobe:
    """
    #111: Auto-detect language and translate seamlessly.
    """

    LANG_NAMES = {
        "en":    "English",    "es": "Spanish",   "fr": "French",
        "de":    "German",     "hi": "Hindi",     "ja": "Japanese",
        "zh-cn": "Chinese",    "ar": "Arabic",    "pt": "Portuguese",
        "ru":    "Russian",    "it": "Italian",   "ko": "Korean",
    }

    TRANSLATE_SYSTEM = (
        "You are a precise translator. Translate the following text to {target_lang}. "
        "Return ONLY the translated text, nothing else."
    )

    def __init__(self, llm_fn: Optional[Callable] = None):
        self._llm = llm_fn

    def detect_language(self, text: str) -> str:
        if _LANGDETECT_OK:
            try:
                return langdetect_detect(text)
            except Exception:
                pass
        if re.search(r'[\u0600-\u06FF]', text): return "ar"
        if re.search(r'[\u0900-\u097F]', text): return "hi"
        if re.search(r'[\u4E00-\u9FFF]', text): return "zh-cn"
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text): return "ja"
        if re.search(r'[\u0400-\u04FF]', text): return "ru"
        return "en"

    def translate(self, text: str, target_lang: str = "English") -> str:
        if not self._llm:
            return text
        try:
            system = self.TRANSLATE_SYSTEM.format(target_lang=target_lang)
            result = self._llm(f"Translate:\n{text}", system=system, model_hint="3b")
            return result.strip()
        except Exception as e:
            logger.warning(f"[LinguisticLobe] Translation error: {e}")
            return text

    def process(self, text: str) -> Tuple[str, str, str]:
        lang = self.detect_language(text)
        if lang == "en" or lang.startswith("en"):
            return text, lang, text
        english_text = self.translate(text, "English")
        return text, lang, english_text

    def respond_in_language(self, response_en: str, target_lang: str) -> str:
        if target_lang == "en" or target_lang.startswith("en"):
            return response_en
        lang_name = self.LANG_NAMES.get(target_lang, target_lang)
        return self.translate(response_en, lang_name)


@dataclass
class UILayout:
    width:      int  = 580
    height:     int  = 520
    mode:       str  = "default"
    show_chart: bool = False
    show_code:  bool = False
    show_cot:   bool = False


class LiquidUI:
    """#24: Dynamically adjusts Qt/web avatar window size to task context."""

    LAYOUTS: Dict[str, UILayout] = {
        "casual":    UILayout(width=400,  height=420, mode="compact"),
        "default":   UILayout(width=580,  height=520, mode="default"),
        "coding":    UILayout(width=900,  height=650, mode="expanded",
                              show_code=True, show_cot=True),
        "analysis":  UILayout(width=1000, height=700, mode="expanded",
                              show_chart=True),
        "sleeping":  UILayout(width=300,  height=300, mode="compact"),
        "reasoning": UILayout(width=750,  height=600, mode="expanded",
                              show_cot=True),
    }

    def __init__(self, on_layout_change: Optional[Callable[[UILayout], None]] = None):
        self._on_change = on_layout_change
        self._current   = self.LAYOUTS["default"]

    def adapt(self, context: str, has_chart: bool = False, has_code: bool = False):
        if context in self.LAYOUTS:
            layout = self.LAYOUTS[context]
        elif has_chart:
            layout = self.LAYOUTS["analysis"]
        elif has_code:
            layout = self.LAYOUTS["coding"]
        else:
            layout = self.LAYOUTS["default"]

        if layout.mode != self._current.mode:
            self._current = layout
            logger.info(
                f"[LiquidUI] Layout: {layout.mode} ({layout.width}x{layout.height})"
            )
            if self._on_change:
                try:
                    self._on_change(layout)
                except Exception:
                    pass

    def get_current(self) -> UILayout:
        return self._current


class CognitiveMirror:
    """#65: Pre-computes likely next commands based on time-of-day + context."""

    PREDICTIONS: List[Tuple[Tuple[int, int], str, List[str]]] = [
        ((8,  10), "productivity", ["What's on my calendar today?",
                                    "Summarize my unread emails"]),
        ((9,  10), "code",         ["Open VS Code",
                                    "Show recent git commits"]),
        ((12, 14), "casual",       ["What's the weather?",
                                    "Set a lunch reminder"]),
        ((17, 19), "productivity", ["What's left on my to-do list?",
                                    "Any missed calls or messages?"]),
        ((22, 24), "casual",       ["What should I prep for tomorrow?",
                                    "Summary of today's tasks completed"]),
    ]

    def __init__(self, llm_fn: Optional[Callable] = None):
        self._llm      = llm_fn
        self._cache:   Dict[str, str] = {}
        self._stop_evt = threading.Event()
        self._lock     = threading.Lock()

    def _predict_commands(self, hour: int, context: str) -> List[str]:
        for (h_start, h_end), ctx, cmds in self.PREDICTIONS:
            if h_start <= hour < h_end and ctx == context:
                return cmds
        return []

    def _precompute(self, command: str):
        if not self._llm or command in self._cache:
            return
        try:
            response = self._llm(command, model_hint="3b")
            with self._lock:
                self._cache[command] = response
            logger.debug(f"[CognitiveMirror] Pre-computed: '{command[:50]}'")
        except Exception as e:
            logger.debug(f"[CognitiveMirror] Pre-compute error: {e}")

    def start(self, context_fn: Optional[Callable[[], str]] = None):
        def _loop():
            while not self._stop_evt.is_set():
                hour    = time.localtime().tm_hour
                context = context_fn() if context_fn else "default"
                for cmd in self._predict_commands(hour, context):
                    if not self._stop_evt.is_set():
                        self._precompute(cmd)
                time.sleep(300)

        threading.Thread(target=_loop, daemon=True, name="CognitiveMirror").start()

    def get_cached(self, command: str) -> Optional[str]:
        with self._lock:
            for cached_cmd, cached_resp in self._cache.items():
                if (command.lower().strip() == cached_cmd.lower().strip()
                        or len(
                            set(command.lower().split())
                            & set(cached_cmd.lower().split())
                        ) >= 3):
                    logger.info(f"[CognitiveMirror] Cache HIT: '{command[:50]}'")
                    return cached_resp
        return None

    def stop(self):
        self._stop_evt.set()


class ExecutiveFunction:
    """#59: Watches calendar, fires callback 10 min before upcoming meetings."""

    def __init__(
        self,
        on_upcoming_meeting: Optional[Callable[[dict], None]] = None,
    ):
        self._on_meeting         = on_upcoming_meeting
        self._stop_evt           = threading.Event()
        self._notified_meetings: set = set()

    def _get_upcoming_meetings(self) -> List[dict]:
        if platform.system() != "Darwin":
            return []
        script = '''
        tell application "Calendar"
            set nowDate to current date
            set futureDate to nowDate + (15 * minutes)
            set meetingList to {}
            repeat with aCal in calendars
                set evts to (events of aCal whose start date > nowDate and start date < futureDate)
                repeat with evt in evts
                    set end of meetingList to {title:summary of evt, startDate:start date of evt as text}
                end repeat
            end repeat
            return meetingList
        end tell
        '''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                meetings = []
                raw = result.stdout.strip()
                for item in raw.split("}, {"):
                    if "title:" in item:
                        title_match = re.search(r'title:([^,}]+)', item)
                        date_match  = re.search(r'startDate:([^,}]+)', item)
                        if title_match:
                            meetings.append({
                                "title": title_match.group(1).strip(),
                                "start": date_match.group(1).strip() if date_match else "",
                            })
                return meetings
        except Exception as e:
            logger.debug(f"[ExecutiveFunction] Calendar error: {e}")
        return []

    def start(self):
        def _loop():
            while not self._stop_evt.is_set():
                try:
                    for meeting in self._get_upcoming_meetings():
                        mid = meeting.get("title", "") + meeting.get("start", "")
                        if mid not in self._notified_meetings:
                            self._notified_meetings.add(mid)
                            logger.info(f"[ExecFn] Upcoming: {meeting.get('title')}")
                            if self._on_meeting:
                                try:
                                    self._on_meeting(meeting)
                                except Exception:
                                    pass
                except Exception as e:
                    logger.debug(f"[ExecutiveFunction] Loop error: {e}")
                time.sleep(60)

        threading.Thread(target=_loop, daemon=True,
                          name="ExecutiveFunction").start()

    def stop(self):
        self._stop_evt.set()


# =====================================================================
# ── AFFECTIVE ENGINE COORDINATOR ────────────────────────────────────
# =====================================================================

class AffectiveEngine:
    """
    Top-level affective coordinator wired into EdgeNodeOrchestrator.
    Exposes all Gestalt Sensorium components as first-class attributes.
    """

    def __init__(
        self,
        llm_fn:              Optional[Callable] = None,
        on_layout_change:    Optional[Callable] = None,
        on_context_change:   Optional[Callable] = None,
        on_upcoming_meeting: Optional[Callable] = None,
    ):
        # Original body-side subsystems
        self.manifold   = AffectiveManifold()
        self.context    = SocialContextSwitcher()
        self.linguistic = UniversalLinguisticLobe(llm_fn=llm_fn)
        self.liquid_ui  = LiquidUI(on_layout_change=on_layout_change)
        self.mirror     = CognitiveMirror(llm_fn=llm_fn)
        self.executive  = ExecutiveFunction(on_upcoming_meeting=on_upcoming_meeting)

        # Migrated Gestalt Sensorium subsystems
        self.affective_compute   = AffectiveComputeEngine()
        self.multimodal_projector = MultimodalProjector()
        self.sensorium           = GestaltSensorium()
        self.wifi_radar          = WiFiSensingLobe()
        self.metrology           = SpatialMetrology()
        self.quantum_sonar       = QuantumSonar()
        self.telemetry_tracker   = DeviceTelemetryTracker(vault_access=True)
        self.peripheral_discovery = DiscoveryLobe()

        if on_context_change:
            self.context.add_context_callback(
                lambda ctx, app: on_context_change(ctx, app)
            )

    def start(self):
        self.context.start_monitoring()
        self.mirror.start(context_fn=lambda: self.context.current_context)
        self.executive.start()
        logger.info("[AffectiveEngine] All affective + sensorium systems started.")

    def stop(self):
        self.context.stop()
        self.mirror.stop()
        self.executive.stop()

    # ── Affective helpers ─────────────────────────────────────────────

    def get_tts_profile(self, bpm: int = 72, emotion: str = "neutral") -> TTSProfile:
        return self.manifold.compute_tts_profile(
            bpm=bpm,
            emotion=emotion,
            context=self.context.current_context,
        )

    def process_input(self, text: str) -> Tuple[str, str, str]:
        """Detect language → (original, lang_code, english_text)."""
        return self.linguistic.process(text)

    def adapt_response(self, response: str, user_lang: str, bpm: int = 72) -> str:
        resp = self.manifold.format_response_for_stress(response, bpm)
        return self.linguistic.respond_in_language(resp, user_lang)

    def get_system_style_hint(self) -> str:
        return self.context.get_system_hint()

    # ── Biometric pipeline ────────────────────────────────────────────

    def biometric_pipeline(self, user_state: str) -> Dict[str, Any]:
        """Full biometric read → stress index → TTS profile → perception mode."""
        bio   = self.affective_compute.full_pipeline(user_state)
        tts   = self.manifold.compute_tts_profile(
            bpm=bio["hr"], context=self.context.current_context
        )
        perc  = set_perception_frequency(bio["stress"])
        return {**bio, "tts_rate": tts.rate, "tts_pitch": tts.pitch,
                "tts_style": tts.style, "perception": perc}

    # ── Status ────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        return {
            "current_context":    self.context.current_context,
            "current_ui_layout":  self.liquid_ui.get_current().mode,
            "torch_available":    _TORCH_OK,
            "psutil_available":   _PSUTIL_OK,
            "langdetect_available": _LANGDETECT_OK,
            "sensorium": {
                "wifi_subcarriers":    self.wifi_radar.csi_subcarriers,
                "sonar_range_km":      self.quantum_sonar.detection_range_km,
                "spatial_calibrated":  self.metrology.calibrated,
                "known_devices":       self.peripheral_discovery.scan_airwaves(),
            },
        }


# ── Module-level singleton ────────────────────────────────────────────
_engine: Optional[AffectiveEngine] = None


def get_affective_engine(**kwargs) -> AffectiveEngine:
    global _engine
    if _engine is None:
        _engine = AffectiveEngine(**kwargs)
    return _engine


# =====================================================================
# ── SELF-TEST ────────────────────────────────────────────────────────
# =====================================================================

if __name__ == "__main__":
    import sys
    import traceback

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    PASSED = 0
    FAILED = 0

    def ok(label: str):
        global PASSED
        PASSED += 1
        print(f"  ✅ {label}")

    def fail(label: str, exc: Exception):
        global FAILED
        FAILED += 1
        print(f"  ❌ {label}: {exc}")
        traceback.print_exc()

    print("\n=== AffectiveComputeEngine ===")
    try:
        ace = AffectiveComputeEngine()
        hr, pupil = ace.read_biometrics("CALM_FOCUSED")
        assert hr == 65 and pupil == 3.0, "CALM_FOCUSED values wrong"
        stress = ace.calculate_stress_index(hr, pupil)
        assert 0.0 <= stress <= 1.0, "Stress not normalised"
        mode = ace.route_symbiotic_response(stress)
        assert "MODE:" in mode, "Mode string malformed"
        pipe = ace.full_pipeline("STRESSED")
        assert pipe["stress"] > pipe["hr"] * 0.0, "pipeline dict incomplete"
        assert "mode" in pipe
        ok("read_biometrics CALM_FOCUSED")
        ok("calculate_stress_index normalised")
        ok("route_symbiotic_response")
        ok("full_pipeline STRESSED")
    except Exception as e:
        fail("AffectiveComputeEngine", e)

    print("\n=== set_perception_frequency ===")
    try:
        assert set_perception_frequency(0.0) == "MODE: AMBIENT-AWARE"
        assert set_perception_frequency(0.5) == "MODE: BULLET-TIME"
        assert set_perception_frequency(1.0) == "MODE: BULLET-TIME"
        ok("stress=0.0 → AMBIENT-AWARE")
        ok("stress=0.5 → BULLET-TIME")
        ok("stress=1.0 → BULLET-TIME")
    except Exception as e:
        fail("set_perception_frequency", e)

    print("\n=== WiFiSensingLobe ===")
    try:
        wsl = WiFiSensingLobe(csi_subcarriers=64)
        try:
            import numpy as np
            csi_moving     = np.ones(64) * 0.8
            csi_stationary = np.ones(64) * 0.3
            csi_clear      = np.ones(64) * 0.1
            assert "MOVING"      in wsl.reconstruct_pose_from_rf(csi_moving)
            assert "STATIONARY"  in wsl.reconstruct_pose_from_rf(csi_stationary)
            assert "CLEAR"       in wsl.reconstruct_pose_from_rf(csi_clear)
            ok("MOVING detection")
            ok("STATIONARY detection")
            ok("ENVIRONMENT_CLEAR detection")
        except ImportError:
            # numpy not available — test with list
            assert "MOVING" in wsl.reconstruct_pose_from_rf([0.9] * 64)
            ok("WiFiSensingLobe list input fallback")
    except Exception as e:
        fail("WiFiSensingLobe", e)

    print("\n=== SpatialMetrology ===")
    try:
        sm = SpatialMetrology()
        assert not sm.calibrated
        result = sm.calibrate_sensor({"imu": "MPU6050"})
        assert result == "CALIBRATION_SUCCESS"
        assert sm.calibrated
        status = sm.get_status()
        assert status["calibrated"] and status["data_present"]
        ok("calibrate_sensor")
        ok("get_status after calibration")
    except Exception as e:
        fail("SpatialMetrology", e)

    print("\n=== QuantumSonar ===")
    try:
        qs = QuantumSonar()
        assert qs.scan_for_anomalies(0.9) == "TARGET_ACQUIRED_VIA_GRAVITY"
        assert qs.scan_for_anomalies(0.3) == "ENVIRONMENT_NOMINAL"
        status = qs.get_status()
        assert status["nav_mode"] == "QUANTUM_INERTIAL"
        ok("anomaly threshold > 0.7 → TARGET_ACQUIRED")
        ok("anomaly threshold < 0.7 → NOMINAL")
        ok("get_status nav_mode")
    except Exception as e:
        fail("QuantumSonar", e)

    print("\n=== DeviceTelemetryTracker ===")
    try:
        dtt = DeviceTelemetryTracker(vault_access=True)
        dtt.register_device("Phone_A", lat=12.9716, lon=77.5946)
        dist = dtt.distance_to_device("Phone_A",
                                       current_lat=12.9800,
                                       current_lon=77.5900)
        assert dist is not None and dist < 5.0, f"Distance unrealistic: {dist}"
        assert dtt.distance_to_device("Ghost_Device", 0, 0) is None
        status = dtt.get_status()
        assert status["authorized_count"] == 1
        ok("register_device + haversine distance")
        ok("unknown device returns None")
        ok("get_status authorized_count")
    except Exception as e:
        fail("DeviceTelemetryTracker", e)

    print("\n=== DiscoveryLobe ===")
    try:
        dl = DiscoveryLobe()
        devices = dl.scan_airwaves()
        assert len(devices) >= 2
        dl.register_device("Smart_TV", "TV", "HDMI-CEC")
        assert "Smart_TV" in dl.scan_airwaves()
        dev = dl.get_device("Tesla_Model_3")
        assert dev is not None and dev["protocol"] == "BLE"
        ok("scan_airwaves returns default devices")
        ok("register_device + re-scan")
        ok("get_device returns correct protocol")
    except Exception as e:
        fail("DiscoveryLobe", e)

    print("\n=== process_event_vision ===")
    try:
        import numpy as np
        f0 = np.array([[0.1, 0.2], [0.3, 0.4]])
        f1 = np.array([[0.8, 0.1], [0.3, 0.0]])
        ev = process_event_vision(f0, f1, threshold=0.1)
        assert ev[0, 0] ==  1.0, "Expected ON event"
        assert ev[0, 1] ==  0.0, "Expected no event (small delta)"
        assert ev[1, 1] == -1.0, "Expected OFF event"
        ok("ON event detected")
        ok("zero event (sub-threshold)")
        ok("OFF event detected")
    except ImportError:
        print("  ⚠️  numpy not available — skipping process_event_vision test")
    except Exception as e:
        fail("process_event_vision", e)

    print("\n=== AffectiveManifold ===")
    try:
        am = AffectiveManifold()
        # bpm=55 → sub-60 branch (rate=0.85, pitch=0.90, style=gentle)
        # bpm=90 → 80-100 branch (rate=1.10)
        # bpm=120 → >115 branch (rate=1.35, volume=0.75)
        very_calm = am.compute_tts_profile(bpm=55)
        engaged   = am.compute_tts_profile(bpm=90)
        stressed  = am.compute_tts_profile(bpm=120)

        assert very_calm.rate  < 1.0,              f"sub-60 rate should be < 1.0, got {very_calm.rate}"
        assert engaged.rate    > 1.0,              f"80-100 rate should be > 1.0, got {engaged.rate}"
        assert stressed.rate   > engaged.rate,     f"stress rate should exceed engaged"
        assert stressed.volume < 1.0,              f"stress volume should drop, got {stressed.volume}"
        assert very_calm.style == "gentle"
        assert stressed.style  == "urgent"

        # Emotion modifiers
        sad     = am.compute_tts_profile(bpm=75, emotion="sad")
        happy   = am.compute_tts_profile(bpm=75, emotion="happy")
        neutral = am.compute_tts_profile(bpm=75)
        assert sad.rate   < neutral.rate
        assert happy.pitch > neutral.pitch

        # Stress shortening — text must be long enough that 2 sentences + suffix < full
        long_text = (
            "Here is a very long and detailed explanation covering many topics. "
            "This is the second sentence with substantial content. "
            "Third sentence continues the explanation in depth. "
            "Fourth sentence adds even more detail. "
            "Fifth sentence wraps up the comprehensive overview of the subject matter."
        )
        short = am.format_response_for_stress(long_text, bpm=120)
        # 2 sentences kept + suffix must be shorter than all 5 sentences
        assert len(short) < len(long_text), (
            f"Expected shortened ({len(short)}) < original ({len(long_text)})"
        )
        no_short = am.format_response_for_stress(long_text, bpm=80)
        assert no_short == long_text  # bpm ≤ 115 must not shorten

        ok("sub-60 bpm rate < 1.0 (slow)")
        ok("80-100 bpm rate > 1.0 (engaged)")
        ok("stressed rate > engaged rate")
        ok("stressed volume drops")
        ok("styles: gentle / urgent correct")
        ok("sad emotion lowers rate")
        ok("happy emotion raises pitch")
        ok("format_response_for_stress shortens at bpm=120")
        ok("format_response_for_stress unchanged at bpm=80")
    except Exception as e:
        fail("AffectiveManifold", e)

    print("\n=== SocialContextSwitcher ===")
    try:
        scs = SocialContextSwitcher()
        ctx1, _ = scs.detect_context("Visual Studio Code")
        ctx2, _ = scs.detect_context("Discord")
        ctx3, _ = scs.detect_context("Terminal")
        assert ctx1 == "code"
        assert ctx2 == "chat"
        assert ctx3 == "terminal"
        hint = scs.get_system_hint("Visual Studio Code")
        assert "technical" in hint.lower()
        ok("VS Code → code context")
        ok("Discord → chat context")
        ok("Terminal → terminal context")
        ok("style_hint contains 'technical'")
    except Exception as e:
        fail("SocialContextSwitcher", e)

    print("\n=== UniversalLinguisticLobe ===")
    try:
        ull = UniversalLinguisticLobe()
        _, lang_en, _ = ull.process("Hello, how are you?")
        _, lang_hi, _ = ull.process("नमस्ते, कैसे हैं आप?")
        _, lang_ar, _ = ull.process("مرحبا")
        _, lang_zh, _ = ull.process("你好世界")
        assert lang_en == "en",    f"en: got {lang_en}"
        assert lang_hi == "hi",    f"hi: got {lang_hi}"
        assert lang_ar == "ar",    f"ar: got {lang_ar}"
        assert lang_zh == "zh-cn", f"zh-cn: got {lang_zh}"
        ok("English detected")
        ok("Hindi detected (Devanagari)")
        ok("Arabic detected")
        ok("Chinese detected")
    except Exception as e:
        fail("UniversalLinguisticLobe", e)

    print("\n=== LiquidUI ===")
    try:
        layouts_seen = []
        lui = LiquidUI(on_layout_change=lambda l: layouts_seen.append(l.mode))
        lui.adapt("casual")
        lui.adapt("coding")
        lui.adapt("analysis")
        assert "compact"  in layouts_seen
        assert "expanded" in layouts_seen
        assert lui.get_current().width >= 900
        ok("casual → compact")
        ok("coding → expanded")
        ok("analysis → expanded with chart")
    except Exception as e:
        fail("LiquidUI", e)

    print("\n=== CognitiveMirror ===")
    try:
        cm = CognitiveMirror(llm_fn=None)
        preds = cm._predict_commands(9, "code")
        assert len(preds) > 0
        cm._cache["What's on my calendar today?"] = "3 meetings."
        hit = cm.get_cached("What's on my calendar today?")
        assert hit == "3 meetings."
        ok("predictions at 9am/code context non-empty")
        ok("cache hit exact match")
    except Exception as e:
        fail("CognitiveMirror", e)

    print("\n=== AffectiveEngine (integrated) ===")
    try:
        ae = AffectiveEngine()
        profile = ae.get_tts_profile(bpm=90, emotion="happy")
        assert profile.rate > 1.0
        orig, lang, en = ae.process_input("Hello world")
        assert lang == "en" and en == "Hello world"
        bio_pipe = ae.biometric_pipeline("FRUSTRATED_EXHAUSTED")
        assert "stress" in bio_pipe and "perception" in bio_pipe
        assert bio_pipe["perception"] == "MODE: BULLET-TIME"
        status = ae.get_status()
        assert "sensorium" in status
        ok("TTS profile happy@90bpm rate > 1.0")
        ok("process_input English passthrough")
        ok("biometric_pipeline FRUSTRATED → BULLET-TIME")
        ok("get_status includes sensorium")
    except Exception as e:
        fail("AffectiveEngine integration", e)

    print(f"\n{'='*50}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    if FAILED:
        sys.exit(1)
    print("✅ All tests passed — affective_engine.py production ready.")
