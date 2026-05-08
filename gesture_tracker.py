#!/usr/bin/env python3
# =====================================================================
# 🖐️  PHASE 2.1 — SPATIAL GESTURE TRACKER (Eyes & Hands Lobe)
# MediaPipe 21-landmark hand tracking → pinch-to-click mouse control
# Runs in a background daemon thread; no external deps beyond mediapipe
# =====================================================================

import threading
import time
import math
import logging
from typing import Optional, Callable

logger = logging.getLogger("GestureTracker")

# ── Optional imports ──────────────────────────────────────────────────
try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False
    logger.warning("cv2 not found — gesture tracking disabled.")

try:
    import mediapipe as mp
    _MP_OK = True
except ImportError:
    _MP_OK = False
    logger.warning("mediapipe not found — gesture tracking disabled.")

try:
    import pyautogui
    pyautogui.FAILSAFE = False  # Disable corner-escape failsafe for robustness
    _PAG_OK = True
except ImportError:
    _PAG_OK = False
    logger.warning("pyautogui not found — mouse actuation disabled.")


# ── Constants ─────────────────────────────────────────────────────────
PINCH_THRESHOLD   = 40      # pixels (normalised → screen): <threshold = pinch
SMOOTHING_ALPHA   = 0.35    # EMA smoothing for mouse movement (lower = smoother)
CAMERA_INDEX      = 0       # Default webcam
FRAME_WIDTH       = 640
FRAME_HEIGHT      = 480
DEBOUNCE_FRAMES   = 3       # require N consecutive frames to confirm pinch state


def _euclidean(p1, p2, w: int, h: int) -> float:
    """
    Euclidean distance between two MediaPipe NormalizedLandmark objects.
    Converts normalised coords → pixel space before computing distance.

    Distance = sqrt((Ix - Tx)^2 + (Iy - Ty)^2)
    as specified in Phase 2.1 requirement.
    """
    dx = (p1.x - p2.x) * w
    dy = (p1.y - p2.y) * h
    return math.sqrt(dx * dx + dy * dy)


class GestureTracker:
    """
    Background thread that reads webcam, runs MediaPipe Hands,
    maps wrist position → absolute screen mouse, and translates
    thumb-index pinch distance → mouseDown / mouseUp events.

    Public API
    ----------
    start()  — launch background thread
    stop()   — signal thread to exit
    is_running : bool
    on_pinch_start : Callable   (optional callback)
    on_pinch_end   : Callable   (optional callback)
    on_position    : Callable[(x,y)]  (optional callback, raw screen coords)
    """

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        pinch_threshold: float = PINCH_THRESHOLD,
        smoothing: float = SMOOTHING_ALPHA,
        on_pinch_start: Optional[Callable] = None,
        on_pinch_end:   Optional[Callable] = None,
        on_position:    Optional[Callable] = None,
    ):
        self._cam_idx     = camera_index
        self._threshold   = pinch_threshold
        self._alpha       = smoothing
        self._stop_evt    = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Callbacks
        self.on_pinch_start = on_pinch_start
        self.on_pinch_end   = on_pinch_end
        self.on_position    = on_position

        # State
        self._pinched         = False
        self._pinch_debounce  = 0
        self._release_debounce= 0
        self._smooth_x        = 0.0
        self._smooth_y        = 0.0
        self._first_frame     = True

        self.is_running = False
        self._last_error: str = ""

    # ── Public ────────────────────────────────────────────────────────
    def start(self) -> bool:
        if not (_CV2_OK and _MP_OK):
            logger.error("Cannot start GestureTracker: cv2 or mediapipe missing.")
            return False
        if self.is_running:
            return True
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="GestureTracker"
        )
        self._thread.start()
        self.is_running = True
        logger.info("GestureTracker started.")
        return True

    def stop(self):
        self._stop_evt.set()
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("GestureTracker stopped.")

    def get_status(self) -> dict:
        return {
            "running": self.is_running,
            "pinched": self._pinched,
            "cv2": _CV2_OK,
            "mediapipe": _MP_OK,
            "pyautogui": _PAG_OK,
            "last_error": self._last_error,
        }

    # ── Internal ──────────────────────────────────────────────────────
    def _run(self):
        if not _MP_OK or not _CV2_OK:
            return

        # ── Support mediapipe legacy (<0.10) AND new Tasks API (0.10+) ──
        hands_model = None
        _mp_legacy  = False

        try:
            # Legacy API (mediapipe < 0.10 — has mp.solutions)
            mp_hands = mp.solutions.hands
            hands_model = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )
            _mp_legacy = True
            logger.info("[GestureTracker] Using mediapipe legacy (solutions) API.")
        except AttributeError:
            logger.info("[GestureTracker] mp.solutions not available — trying Tasks API.")

        if not _mp_legacy:
            try:
                from mediapipe.tasks.python import vision as _mpv
                from mediapipe.tasks.python.vision import HandLandmarkerOptions as _HLO
                from mediapipe.tasks.python.vision import HandLandmarker as _HL
                from mediapipe.tasks.python.vision import RunningMode as _RM
                import mediapipe.tasks.python as _mpt
                import urllib.request

                model_cache = Path.home() / ".cache" / "mediapipe" / "hand_landmarker.task"
                model_cache.parent.mkdir(parents=True, exist_ok=True)
                if not model_cache.exists():
                    logger.info("[GestureTracker] Downloading hand_landmarker.task…")
                    try:
                        urllib.request.urlretrieve(
                            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                            str(model_cache)
                        )
                        logger.info("[GestureTracker] Model downloaded.")
                    except Exception as _dl:
                        logger.error(f"[GestureTracker] Model download failed: {_dl}")
                        self._last_error = "model_download_failed"
                        self.is_running = False
                        return

                opts = _HLO(
                    base_options=_mpt.BaseOptions(model_asset_path=str(model_cache)),
                    running_mode=_RM.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=0.7,
                    min_tracking_confidence=0.6,
                )
                hands_model = _HL.create_from_options(opts)
                logger.info("[GestureTracker] Using mediapipe Tasks API.")
            except Exception as _te:
                logger.error(f"[GestureTracker] Tasks API init failed: {_te}")
                self._last_error = str(_te)
                self.is_running = False
                return

        cap = cv2.VideoCapture(self._cam_idx)
        if not cap.isOpened():
            self._last_error = f"Cannot open camera {self._cam_idx}"
            logger.error(self._last_error)
            self.is_running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

        try:
            import screeninfo
            screens = screeninfo.get_monitors()
            screen_w = screens[0].width
            screen_h = screens[0].height
        except Exception:
            try:
                if _PAG_OK:
                    screen_w, screen_h = pyautogui.size()
                else:
                    screen_w, screen_h = 1920, 1080
            except Exception:
                screen_w, screen_h = 1920, 1080

        logger.info(f"GestureTracker running. Screen: {screen_w}×{screen_h}")

        while not self._stop_evt.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.033)
                continue

            # Flip for natural mirror-image control
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_model.process(rgb)

            if not results.multi_hand_landmarks:
                # No hand detected — release if pinched
                if self._pinched:
                    self._do_release()
                self._first_frame = True
                continue

            lms = results.multi_hand_landmarks[0].landmark

            # ── Landmark indices (MediaPipe convention) ────────────────
            # 4 = THUMB_TIP (T), 8 = INDEX_FINGER_TIP (I)
            # 0 = WRIST (for position mapping)
            T = lms[4]   # thumb tip
            I = lms[8]   # index tip
            W = lms[0]   # wrist

            # ── Pinch distance (pixel-space Euclidean) ─────────────────
            dist = _euclidean(T, I, w, h)

            # ── Map wrist → screen coordinates ────────────────────────
            raw_x = W.x * screen_w   # mirrored x
            raw_y = W.y * screen_h

            # EMA smoothing
            if self._first_frame:
                self._smooth_x = raw_x
                self._smooth_y = raw_y
                self._first_frame = False
            else:
                a = self._alpha
                self._smooth_x = a * raw_x + (1 - a) * self._smooth_x
                self._smooth_y = a * raw_y + (1 - a) * self._smooth_y

            sx = int(self._smooth_x)
            sy = int(self._smooth_y)

            # Clamp to screen bounds
            sx = max(0, min(screen_w - 1, sx))
            sy = max(0, min(screen_h - 1, sy))

            # ── Position callback ──────────────────────────────────────
            if self.on_position:
                try:
                    self.on_position(sx, sy)
                except Exception:
                    pass

            # ── Mouse movement ─────────────────────────────────────────
            if _PAG_OK:
                try:
                    pyautogui.moveTo(sx, sy, duration=0)
                except Exception:
                    pass

            # ── Pinch state machine with debounce ──────────────────────
            if dist < self._threshold:
                # Potential pinch
                self._release_debounce = 0
                if not self._pinched:
                    self._pinch_debounce += 1
                    if self._pinch_debounce >= DEBOUNCE_FRAMES:
                        self._do_pinch(sx, sy)
                else:
                    # Already pinched — update drag position
                    if _PAG_OK:
                        pass  # moveTo already called above
            else:
                # Potential release
                self._pinch_debounce = 0
                if self._pinched:
                    self._release_debounce += 1
                    if self._release_debounce >= DEBOUNCE_FRAMES:
                        self._do_release()

        cap.release()
        hands_model.close()
        self.is_running = False
        logger.info("GestureTracker thread exited.")

    def _do_pinch(self, x: int, y: int):
        self._pinched = True
        self._pinch_debounce = 0
        logger.debug(f"PINCH DOWN at ({x},{y})")
        if _PAG_OK:
            try:
                pyautogui.mouseDown(x, y, button="left")
            except Exception as e:
                logger.warning(f"mouseDown failed: {e}")
        if self.on_pinch_start:
            try:
                self.on_pinch_start(x, y)
            except Exception:
                pass

    def _do_release(self):
        self._pinched = False
        self._release_debounce = 0
        logger.debug("PINCH RELEASE")
        if _PAG_OK:
            try:
                pyautogui.mouseUp(button="left")
            except Exception as e:
                logger.warning(f"mouseUp failed: {e}")
        if self.on_pinch_end:
            try:
                self.on_pinch_end()
            except Exception:
                pass


# ── Module-level singleton ────────────────────────────────────────────
_tracker: Optional[GestureTracker] = None


def get_tracker() -> GestureTracker:
    global _tracker
    if _tracker is None:
        _tracker = GestureTracker()
    return _tracker


def start_gesture_tracking(**kwargs) -> bool:
    """Convenience: start module-level singleton tracker."""
    global _tracker
    _tracker = GestureTracker(**kwargs)
    return _tracker.start()


def stop_gesture_tracking():
    global _tracker
    if _tracker:
        _tracker.stop()


# ── Self-test ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────
# MAIN  (v13.2 — headless / daemon only, no Qt)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    MANUAL_BRAIN_URL = os.getenv("BRAIN_URL", "https://dayton-backdoor-celestially.ngrok-free.dev")
    AUTO_DAEMON = "--daemon" in sys.argv
    INSTALL_DAEMON = "--install-daemon" in sys.argv

    _db = init_firebase_edge()

    edge = EdgeNodeOrchestrator()
    if MANUAL_BRAIN_URL:
        edge.sync._brain_url = MANUAL_BRAIN_URL

    if INSTALL_DAEMON:
        ok = edge.daemon.daemonize()
        print("✅ Daemon installed." if ok else "❌ Daemon install failed.")
        if not ok:
            sys.exit(1)

    print("\n⚠️  [System] Offline Survival Brain check.")

    # 🧠 NATIVE LOCAL BRAIN BOOT - Automatically load into RAM
    try:
        from swayambhu_utils import PROJECT_ROOT

        cached_model = PROJECT_ROOT / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        if cached_model.exists():
            threading.Thread(
                target=lambda: edge.local_llm.switch_to_local_llm(cached_model),
                daemon=True, name="LLMLoader"
            ).start()
    except Exception as e:
        print(f"⚠️ Could not load local model into RAM: {e}")

    edge.boot(firebase_db=_db, auto_procure=False)
    start_edge_server(edge)

    print(
        f"\n🌌 Swayambhu v13.2 Edge Node online."
        f"\n   Node ID : {NODE_ID}"
        f"\n   API     : http://localhost:{EDGE_SERVER_PORT}/health"
        f"\n   Process : com.apple.syslogd (stealth)"
        f"\n   Mode    : {'daemon' if AUTO_DAEMON else 'headless'}"
        f"\n   Ctrl-C to stop.\n"
    )

    # 💬 TERMINAL CHAT MODE RESTORED
    try:
        print("\n💬 Terminal Chat Mode Active. Type your command and press Enter.")
        while True:
            cmd = input("\n[You] > ").strip()
            if cmd.lower() in ["exit", "quit", "stop"]:
                break
            if cmd:
                edge.route_command(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down.")
