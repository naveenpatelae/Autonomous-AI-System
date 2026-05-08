#!/usr/bin/env python3
# =====================================================================
# 👁️  WAKE DETECTOR — Clap / "Hello" wake + Facial Emotion Sensing
#
# Responsibilities:
#   - Listen for clap (audio transient > threshold) or "hello" keyword
#   - Detect facial emotion (happy/sad/neutral/excited/angry) via webcam
#   - Fire callbacks: on_wake(), on_emotion(emotion_str), on_sleep()
#   - All in background daemon threads — zero blocking
# =====================================================================

from __future__ import annotations

import threading
import time
import logging
import math
from typing import Optional, Callable

logger = logging.getLogger("WakeDetector")

# ── Optional deps ─────────────────────────────────────────────────────
try:
    import numpy as np
    _NP_OK = True
except ImportError:
    _NP_OK = False

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False
    logger.warning("cv2 not found — emotion sensing disabled.")

try:
    import pyaudio
    _PA_OK = True
except ImportError:
    _PA_OK = False
    logger.warning("pyaudio not found — clap detection disabled.")

try:
    import speech_recognition as sr
    _SR_OK = True
except ImportError:
    _SR_OK = False
    logger.warning("SpeechRecognition not found — keyword wake disabled.")

# ── Emotion model: try deepface, fallback to haar cascade + heuristic ─
_DEEPFACE_OK = False
try:
    from deepface import DeepFace
    _DEEPFACE_OK = True
except ImportError:
    pass

# ── Constants ─────────────────────────────────────────────────────────
CLAP_THRESHOLD        = 0.35   # RMS amplitude 0-1 for clap detection
CLAP_DEBOUNCE_SEC     = 1.5    # seconds between clap triggers
EMOTION_SAMPLE_SEC    = 4.0    # how often to sample emotion
WAKE_KEYWORDS         = {"hello", "hey", "swayambhu", "wake up", "hi"}
AUDIO_CHUNK           = 1024
AUDIO_RATE            = 16000
AUDIO_CHANNELS        = 1


class WakeDetector:
    """
    Dual-mode wake detector:
      1. Audio — clap transient OR keyword ("hello"/"swayambhu")
      2. Vision — continuous emotion sensing from webcam

    Callbacks
    ---------
    on_wake(source: str)       — fired when wake trigger detected
    on_emotion(emotion: str)   — fired every EMOTION_SAMPLE_SEC
    on_sleep()                 — called when system transitions to sleep
    """

    def __init__(
        self,
        camera_index: int = 0,
        on_wake: Optional[Callable[[str], None]] = None,
        on_emotion: Optional[Callable[[str], None]] = None,
        on_sleep: Optional[Callable] = None,
        enable_clap: bool = True,
        enable_keyword: bool = True,
        enable_emotion: bool = True,
    ):
        self._cam_idx       = camera_index
        self.on_wake        = on_wake
        self.on_emotion     = on_emotion
        self.on_sleep       = on_sleep

        self._enable_clap    = enable_clap and _PA_OK and _NP_OK
        self._enable_keyword = enable_keyword and _SR_OK
        self._enable_emotion = enable_emotion and _CV2_OK

        self._stop_evt      = threading.Event()
        self._awake         = False
        self._last_clap     = 0.0
        self._lock          = threading.Lock()
        self.is_running     = False

        # Haar cascade for face detection (always available in cv2)
        self._face_cascade  = None
        if _CV2_OK:
            cc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cc_path)

    # ── Public API ────────────────────────────────────────────────────
    def start(self):
        if self.is_running:
            return
        self._stop_evt.clear()
        self.is_running = True

        if self._enable_clap:
            threading.Thread(
                target=self._clap_loop, daemon=True, name="ClapDetector"
            ).start()

        if self._enable_keyword:
            threading.Thread(
                target=self._keyword_loop, daemon=True, name="KeywordWake"
            ).start()

        if self._enable_emotion:
            threading.Thread(
                target=self._emotion_loop, daemon=True, name="EmotionSensor"
            ).start()

        logger.info("WakeDetector started.")

    def stop(self):
        self._stop_evt.set()
        self.is_running = False
        logger.info("WakeDetector stopped.")

    def set_sleeping(self):
        """Tell the detector the system is now asleep."""
        with self._lock:
            self._awake = False
        if self.on_sleep:
            try:
                self.on_sleep()
            except Exception:
                pass

    def set_awake(self):
        with self._lock:
            self._awake = True

    @property
    def is_awake(self) -> bool:
        return self._awake

    def get_status(self) -> dict:
        return {
            "running": self.is_running,
            "awake": self._awake,
            "clap_enabled": self._enable_clap,
            "keyword_enabled": self._enable_keyword,
            "emotion_enabled": self._enable_emotion,
            "deepface": _DEEPFACE_OK,
        }

    # ── Clap detection ────────────────────────────────────────────────
    def _clap_loop(self):
        """Listens for audio transients (claps) via PyAudio."""
        if not _PA_OK or not _NP_OK:
            return
        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_RATE,
                input=True,
                frames_per_buffer=AUDIO_CHUNK,
            )
            logger.info("[WakeDetector] Clap listener active.")
            while not self._stop_evt.is_set():
                try:
                    data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                    samples = np.frombuffer(data, dtype=np.float32)
                    rms = float(np.sqrt(np.mean(samples ** 2)))
                    now = time.time()
                    if rms > CLAP_THRESHOLD and (now - self._last_clap) > CLAP_DEBOUNCE_SEC:
                        self._last_clap = now
                        logger.info(f"[WakeDetector] Clap detected! RMS={rms:.3f}")
                        self._fire_wake("clap")
                except Exception as e:
                    logger.debug(f"[ClapDetector] read error: {e}")
                    time.sleep(0.1)
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except Exception as e:
            logger.warning(f"[ClapDetector] Failed to open audio: {e}")

    # ── Keyword wake ─────────────────────────────────────────────────
    def _keyword_loop(self):
        """Listens for wake keywords using SpeechRecognition."""
        if not _SR_OK:
            return
        try:
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            mic = sr.Microphone(sample_rate=AUDIO_RATE)
            logger.info("[WakeDetector] Keyword listener active.")
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)

            while not self._stop_evt.is_set():
                try:
                    with mic as source:
                        audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
                    text = recognizer.recognize_google(audio).lower().strip()
                    logger.debug(f"[Keyword] Heard: '{text}'")
                    for kw in WAKE_KEYWORDS:
                        if kw in text:
                            logger.info(f"[WakeDetector] Keyword wake: '{kw}'")
                            self._fire_wake(f"keyword:{kw}")
                            break
                except sr.WaitTimeoutError:
                    pass
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    logger.debug(f"[Keyword] SR request error: {e}")
                    time.sleep(2)
                except Exception as e:
                    logger.debug(f"[Keyword] error: {e}")
                    time.sleep(1)
        except Exception as e:
            logger.warning(f"[KeywordWake] Init failed: {e}")

    # ── Emotion sensing ───────────────────────────────────────────────
    def _emotion_loop(self):
        """Samples webcam every EMOTION_SAMPLE_SEC and detects emotion."""
        if not _CV2_OK:
            return
        try:
            cap = cv2.VideoCapture(self._cam_idx)
            if not cap.isOpened():
                logger.warning("[EmotionSensor] Cannot open camera.")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            logger.info("[WakeDetector] Emotion sensor active.")

            while not self._stop_evt.is_set():
                time.sleep(EMOTION_SAMPLE_SEC)
                if self._stop_evt.is_set():
                    break
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.5)
                    continue
                emotion = self._detect_emotion(frame)
                if emotion and self.on_emotion:
                    try:
                        self.on_emotion(emotion)
                    except Exception:
                        pass

            cap.release()
        except Exception as e:
            logger.warning(f"[EmotionSensor] Loop error: {e}")

    def _detect_emotion(self, frame) -> Optional[str]:
        """
        Detect dominant emotion from a BGR frame.
        Tries DeepFace first; falls back to face presence + brightness heuristic.
        Returns one of: happy, sad, angry, surprised, neutral, excited
        """
        # ── DeepFace path ─────────────────────────────────────────────
        if _DEEPFACE_OK:
            try:
                result = DeepFace.analyze(
                    frame,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                if isinstance(result, list):
                    result = result[0]
                dominant = result.get("dominant_emotion", "neutral")
                # Map to our vocabulary
                mapping = {
                    "happy": "happy",
                    "sad": "sad",
                    "angry": "angry",
                    "surprise": "excited",
                    "fear": "sad",
                    "disgust": "angry",
                    "neutral": "neutral",
                }
                return mapping.get(dominant, "neutral")
            except Exception as e:
                logger.debug(f"[DeepFace] error: {e}")

        # ── Haar cascade fallback — face presence + brightness delta ─
        if self._face_cascade is None:
            return "neutral"
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )
            if len(faces) == 0:
                return None  # No face — don't emit anything

            # Simple heuristic: brightness of upper-lip region suggests smile
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            mouth_roi = face_roi[int(h*0.65):int(h*0.90), int(w*0.25):int(w*0.75)]
            if mouth_roi.size > 0:
                brightness = float(np.mean(mouth_roi)) if _NP_OK else 128
                if brightness > 160:
                    return "happy"
                elif brightness < 80:
                    return "sad"
            return "neutral"
        except Exception as e:
            logger.debug(f"[HaarEmotion] error: {e}")
            return "neutral"

    # ── Internal helpers ──────────────────────────────────────────────
    def _fire_wake(self, source: str):
        """Fire wake callback and mark system as awake."""
        with self._lock:
            self._awake = True
        if self.on_wake:
            try:
                self.on_wake(source)
            except Exception as e:
                logger.warning(f"[WakeDetector] on_wake callback error: {e}")


# ── Module-level singleton ────────────────────────────────────────────
_wake_detector: Optional[WakeDetector] = None


def get_wake_detector(**kwargs) -> WakeDetector:
    global _wake_detector
    if _wake_detector is None:
        _wake_detector = WakeDetector(**kwargs)
    return _wake_detector


# ── Self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    print("👁️  WakeDetector self-test (10 seconds)\n")
    print(f"  numpy={_NP_OK}, cv2={_CV2_OK}, pyaudio={_PA_OK}")
    print(f"  sr={_SR_OK}, deepface={_DEEPFACE_OK}")

    wakes = []
    emotions = []

    det = WakeDetector(
        on_wake=lambda s: (wakes.append(s), print(f"  ✅ WAKE from: {s}")),
        on_emotion=lambda e: (emotions.append(e), print(f"  😀 Emotion: {e}")),
    )
    det.start()
    print("\n  Clap loudly or say 'hello' to trigger wake...\n")

    try:
        time.sleep(10)
    except KeyboardInterrupt:
        pass
    finally:
        det.stop()
        print(f"\n  Wake events: {wakes}")
        print(f"  Emotion samples: {emotions}")
        print("✅ WakeDetector self-test complete.")
