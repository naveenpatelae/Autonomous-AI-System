#!/usr/bin/env python3
# =====================================================================
# 👁️  PROACTIVE AGENCY — Autonomous OS Integration & Proactive Partner
#
# #59   Executive Function     — Calendar/Mail pre-load before meetings
# #65   Predictive Twin        — Predict next command, pre-compute answer
# #24   Liquid UI Signal       — Emit resize events to Qt Avatar window
# #86   Motor Cortex           — macOS Accessibility API semantic clicks
# #18   Neural Mirror          — Ambient Daemon learning desktop habits
# #101  Social Context Switch  — Tech jargon vs casual based on active app
# #111  Universal Linguistic   — Auto language detection + translation
# =====================================================================

from __future__ import annotations

import json
import logging
import os
import platform
import re
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ProactiveAgency")

# ── Optional heavy deps (torch + numpy) — graceful fallback if absent ─
try:
    import torch
    import torch.nn.functional as _F
    _TORCH_OK = True
except ImportError:
    torch = None          # type: ignore[assignment]
    _F    = None          # type: ignore[assignment]
    _TORCH_OK = False

try:
    import numpy as _np
    _NP_OK = True
except ImportError:
    _np    = None         # type: ignore[assignment]
    _NP_OK = False

try:
    from swayambhu_utils import PROJECT_ROOT
except ImportError:
    try:
        PROJECT_ROOT = Path(__file__).parent.resolve()
    except NameError:
        PROJECT_ROOT = Path(os.getcwd()).resolve()

_BASE_DIR   = Path(os.getenv("SWAYAMBHU_DIR", str(PROJECT_ROOT)))
_AGENCY_DIR = _BASE_DIR / "proactive_agency"
_AGENCY_DIR.mkdir(parents=True, exist_ok=True)

HABIT_LOG_PATH   = _AGENCY_DIR / "habits.json"
TWIN_CACHE_PATH  = _AGENCY_DIR / "twin_cache.json"


# ─────────────────────────────────────────────────────────────────────
# MOTOR CORTEX — macOS Accessibility API (#86)
# ─────────────────────────────────────────────────────────────────────
class MotorCortex:
    """
    Interacts with macOS apps via AppleScript Accessibility API.
    Uses semantic element names (menu titles, button labels) rather
    than fragile pixel coordinates.

    Falls back to pyautogui for non-macOS or unknown elements.
    """

    def __init__(self, confirm_fn: Optional[Callable[[str], bool]] = None):
        self._confirm = confirm_fn or (lambda msg: True)
        self._is_mac  = platform.system() == "Darwin"
        self._log: List[dict] = []

    def _run_applescript(self, script: str, timeout: int = 15) -> dict:
        if not self._is_mac:
            logger.info(f"[MotorCortex] DRY_RUN (non-macOS): {script[:60]}")
            return {"status": "DRY_RUN", "stdout": "", "stderr": ""}
        try:
            r = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=timeout,
            )
            return {
                "status": "OK" if r.returncode == 0 else "ERROR",
                "stdout": r.stdout.strip(),
                "stderr": r.stderr.strip(),
            }
        except subprocess.TimeoutExpired:
            return {"status": "TIMEOUT"}
        except Exception as e:
            return {"status": "EXCEPTION", "error": str(e)}

    def click_menu(self, app: str, menu: str, item: str) -> dict:
        """Click a named menu item in an app. e.g. click_menu('Finder','File','New Window')"""
        msg = f"Click {app} → {menu} → {item}"
        if not self._confirm(f"I want to: {msg}. Proceed?"):
            return {"status": "DECLINED"}
        script = (
            f'tell application "{app}" to activate\n'
            f'tell application "System Events"\n'
            f'  tell process "{app}"\n'
            f'    click menu item "{item}" of menu "{menu}" of menu bar 1\n'
            f'  end tell\n'
            f'end tell'
        )
        result = self._run_applescript(script)
        self._log.append({"ts": time.time(), "action": msg, "result": result["status"]})
        logger.info(f"[MotorCortex] {msg} → {result['status']}")
        return result

    def click_button(self, app: str, button_name: str) -> dict:
        """Click a named button in the frontmost window of an app."""
        msg = f"Click button '{button_name}' in {app}"
        if not self._confirm(f"I want to: {msg}. Proceed?"):
            return {"status": "DECLINED"}
        script = (
            f'tell application "{app}" to activate\n'
            f'tell application "System Events"\n'
            f'  tell process "{app}"\n'
            f'    click button "{button_name}" of window 1\n'
            f'  end tell\n'
            f'end tell'
        )
        result = self._run_applescript(script)
        self._log.append({"ts": time.time(), "action": msg, "result": result["status"]})
        return result

    def type_text(self, app: str, text: str) -> dict:
        """Type text into the frontmost text field of an app."""
        msg = f"Type text into {app}"
        if not self._confirm(f"I want to type into {app}: '{text[:40]}'. Proceed?"):
            return {"status": "DECLINED"}
        # Escape double quotes in text
        safe_text = text.replace('"', '\\"')
        script = (
            f'tell application "{app}" to activate\n'
            f'tell application "System Events"\n'
            f'  keystroke "{safe_text}"\n'
            f'end tell'
        )
        result = self._run_applescript(script)
        self._log.append({"ts": time.time(), "action": msg, "result": result["status"]})
        return result

    def open_app(self, app_name: str) -> dict:
        """Open an application by name."""
        msg = f"Open {app_name}"
        if not self._confirm(f"Open {app_name}?"):
            return {"status": "DECLINED"}
        script = f'tell application "{app_name}" to activate'
        result = self._run_applescript(script)
        self._log.append({"ts": time.time(), "action": msg, "result": result["status"]})
        return result

    def get_frontmost_app(self) -> str:
        """Returns the name of the currently active app."""
        if not self._is_mac:
            return "Unknown"
        script = (
            'tell application "System Events"\n'
            '  name of first application process whose frontmost is true\n'
            'end tell'
        )
        result = self._run_applescript(script)
        return result.get("stdout", "Unknown")

    def get_log(self) -> List[dict]:
        return list(self._log)

    def get_status(self) -> dict:
        return {
            "is_mac": self._is_mac,
            "actions_taken": len(self._log),
        }


# ─────────────────────────────────────────────────────────────────────
# EXECUTIVE FUNCTION — Calendar / Mail Pre-load (#59)
# ─────────────────────────────────────────────────────────────────────
class ExecutiveFunction:
    """
    Connects to Apple Calendar and Apple Mail via AppleScript.
    10 minutes before a meeting → autonomously pre-loads relevant docs.
    Reads and summarises unread mail on demand.
    """

    MEETING_LOOKAHEAD_MIN = 10   # alert N minutes before meeting

    def __init__(
        self,
        motor: MotorCortex,
        speak_fn: Optional[Callable[[str], None]] = None,
        llm_fn:   Optional[Callable[[str], str]]  = None,
    ):
        self._motor  = motor
        self._speak  = speak_fn or (lambda t: print(f"[Executive] {t}"))
        self._llm    = llm_fn
        self._stop   = threading.Event()
        self._is_mac = platform.system() == "Darwin"
        self._running = False

    # ── Calendar ─────────────────────────────────────────────────────
    def get_upcoming_events(self, hours: int = 24) -> List[dict]:
        """Fetch upcoming calendar events via AppleScript."""
        if not self._is_mac:
            return []
        script = (
            'set result to ""\n'
            'tell application "Calendar"\n'
            f'  set theEnd to (current date) + ({hours} * hours)\n'
            '  repeat with aCal in calendars\n'
            '    repeat with anEvent in (every event of aCal whose start date > '
            '(current date) and start date < theEnd)\n'
            '      set result to result & (summary of anEvent) & "|" & '
            '(start date of anEvent as string) & "\n"\n'
            '    end repeat\n'
            '  end repeat\n'
            'end tell\n'
            'result'
        )
        try:
            r = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=10,
            )
            events = []
            for line in r.stdout.strip().splitlines():
                parts = line.split("|")
                if len(parts) >= 2:
                    events.append({"title": parts[0].strip(), "start": parts[1].strip()})
            return events
        except Exception as e:
            logger.warning(f"[Executive] Calendar error: {e}")
            return []

    def get_unread_mail_summary(self, max_messages: int = 10) -> str:
        """Read unread mail subjects via AppleScript and summarise."""
        if not self._is_mac:
            return "Mail reading not available on this platform."
        script = (
            'set result to ""\n'
            'tell application "Mail"\n'
            f'  set msgs to (messages of inbox whose read status is false)\n'
            f'  set cnt to count of msgs\n'
            f'  if cnt > {max_messages} then set cnt to {max_messages}\n'
            '  repeat with i from 1 to cnt\n'
            '    set m to item i of msgs\n'
            '    set result to result & (sender of m) & ": " & (subject of m) & "\n"\n'
            '  end repeat\n'
            'end tell\n'
            'result'
        )
        try:
            r = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=10,
            )
            raw = r.stdout.strip()
            if not raw:
                return "No unread mail."
            if self._llm:
                prompt = (
                    f"Summarise these unread emails in 2-3 sentences. "
                    f"Highlight anything urgent:\n\n{raw}"
                )
                try:
                    return self._llm(prompt)
                except Exception:
                    pass
            return raw
        except Exception as e:
            logger.warning(f"[Executive] Mail error: {e}")
            return f"Mail read error: {e}"

    def _meeting_alert_loop(self):
        """Background: alert 10 min before calendar events. Includes 60s boot delay."""
        self._running = True
        alerted: set = set()

        # 🚀 ARCHITECTURAL FIX: 60-second grace period.
        # Let the LLMs load into RAM before slamming macOS with AppleScripts.
        self._stop.wait(timeout=60.0)

        while not self._stop.is_set():
            try:
                events = self.get_upcoming_events(hours=1)
                for ev in events:
                    key = ev["title"] + ev["start"]
                    if key in alerted:
                        continue
                    alerted_flag = False
                    try:
                        self._speak(
                            f"Heads up — '{ev['title']}' is coming up soon."
                        )
                        alerted.add(key)
                        alerted_flag = True
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"[Executive] Alert loop error: {e}")
            self._stop.wait(timeout=60)
        self._running = False

    def start(self):
        # 🚀 ARCHITECTURAL FIX: Disable autonomous background calendar polling.
        # The AI will now only check the calendar on-demand via the Universal Action Space.
        # threading.Thread(
        #     target=self._meeting_alert_loop, daemon=True, name="ExecutiveFunction"
        # ).start()
        logger.info("[ExecutiveFunction] Autonomous background polling disabled. Calendar is on-demand only.")

    def stop(self):
        self._stop.set()

    def get_status(self) -> dict:
        return {"running": self._running, "is_mac": self._is_mac}


# ─────────────────────────────────────────────────────────────────────
# NEURAL MIRROR / AMBIENT DAEMON (#18)
# ─────────────────────────────────────────────────────────────────────
class NeuralMirror:
    """
    Learns desktop habits by observing which apps are open at which times.
    At the predicted habit time, pre-loads apps and pre-warms AI context.
    Uses psutil to read running processes every 60 s.
    """

    SAMPLE_INTERVAL = 60   # seconds between samples
    MIN_HABIT_DAYS  = 2    # minimum days to confirm a habit

    def __init__(
        self,
        motor: MotorCortex,
        speak_fn: Optional[Callable[[str], None]] = None,
        llm_fn:   Optional[Callable[[str], str]]  = None,
    ):
        self._motor  = motor
        self._speak  = speak_fn or (lambda t: print(f"[Mirror] {t}"))
        self._llm    = llm_fn
        self._stop   = threading.Event()
        self._running = False

        # habits: {hour_key: {app_name: count}}
        self._habits: Dict[str, Dict[str, int]] = self._load_habits()
        self._prewarmed: set = set()  # which hours have been pre-warmed today

    # ── Persistence ───────────────────────────────────────────────────
    def _load_habits(self) -> Dict:
        if HABIT_LOG_PATH.exists():
            try:
                return json.loads(HABIT_LOG_PATH.read_text())
            except Exception:
                pass
        return {}

    def _save_habits(self):
        try:
            HABIT_LOG_PATH.write_text(json.dumps(self._habits, indent=2))
        except Exception as e:
            logger.warning(f"[NeuralMirror] Save error: {e}")

    # ── App sensing ───────────────────────────────────────────────────
    def _get_running_apps(self) -> List[str]:
        """Return list of running application names."""
        apps = []
        try:
            import psutil
            for proc in psutil.process_iter(["name"]):
                name = proc.info.get("name", "")
                if name and not name.startswith(".") and len(name) > 2:
                    apps.append(name)
        except ImportError:
            # Fallback: macOS only
            if platform.system() == "Darwin":
                try:
                    script = (
                        'tell application "System Events"\n'
                        '  name of every application process whose background only is false\n'
                        'end tell'
                    )
                    r = subprocess.run(
                        ["osascript", "-e", script],
                        capture_output=True, text=True, timeout=5,
                    )
                    raw = r.stdout.strip()
                    apps = [a.strip() for a in raw.split(",") if a.strip()]
                except Exception:
                    pass
        return list(set(apps))

    # ── Habit learning ────────────────────────────────────────────────
    def _record_sample(self):
        hour_key = str(time.localtime().tm_hour)
        apps     = self._get_running_apps()
        bucket   = self._habits.setdefault(hour_key, {})
        for app in apps:
            bucket[app] = bucket.get(app, 0) + 1
        self._save_habits()

    def _get_habitual_apps(self, hour: int, min_count: int = 3) -> List[str]:
        """Return apps that habitually run at this hour."""
        bucket = self._habits.get(str(hour), {})
        return [app for app, cnt in bucket.items() if cnt >= min_count]

    def _prewarm(self, hour: int):
        """Pre-open habitual apps and pre-warm AI context."""
        apps = self._get_habitual_apps(hour)
        if not apps:
            return

        key = f"{datetime.now().date()}_{hour}"
        if key in self._prewarmed:
            return
        self._prewarmed.add(key)

        self._speak(
            f"Good timing — I'm pre-loading your usual apps for {hour:02d}:00: "
            + ", ".join(apps[:4])
        )

        for app in apps[:4]:
            # Only open known productive apps, not system processes
            if any(safe in app.lower() for safe in [
                "code", "xcode", "terminal", "cursor", "pycharm",
                "slack", "notion", "safari", "chrome", "mail", "calendar",
            ]):
                self._motor.open_app(app)
                time.sleep(0.5)

        # Pre-warm LLM context
        if self._llm and any("code" in a.lower() for a in apps):
            try:
                self._llm("Brief: what are Python async best practices?")
            except Exception:
                pass

    # ── Background loop ───────────────────────────────────────────────
    def _run_loop(self):
        self._running = True
        while not self._stop.is_set():
            self._record_sample()

            # 5 minutes before the next hour → pre-warm
            now  = time.localtime()
            mins = now.tm_min
            if 55 <= mins <= 59:
                next_hour = (now.tm_hour + 1) % 24
                self._prewarm(next_hour)

            self._stop.wait(timeout=self.SAMPLE_INTERVAL)
        self._running = False

    def start(self):
        self._stop.clear()
        threading.Thread(
            target=self._run_loop, daemon=True, name="NeuralMirror"
        ).start()
        logger.info("[NeuralMirror] Ambient habit-learning started.")

    def stop(self):
        self._stop.set()

    def get_habits_summary(self) -> dict:
        summary = {}
        for hour, apps in self._habits.items():
            top = sorted(apps.items(), key=lambda x: x[1], reverse=True)[:5]
            summary[f"{hour}:00"] = [a for a, _ in top]
        return summary

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "hours_tracked": len(self._habits),
            "habits": self.get_habits_summary(),
        }


# ─────────────────────────────────────────────────────────────────────
# PREDICTIVE TWIN (#65)
# ─────────────────────────────────────────────────────────────────────
class PredictiveTwin:
    """
    Predicts the user's NEXT command based on time-of-day + recent history.
    Pre-computes the answer so it appears with zero latency.

    Prediction is probabilistic: uses habit log + LLM to generate
    the most likely next 3 commands, then pre-runs the top one.
    """

    MAX_HISTORY = 50

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        mirror: Optional[NeuralMirror] = None,
    ):
        self._llm     = llm_fn
        self._mirror  = mirror
        self._history: List[dict] = self._load_cache()
        self._cache: Dict[str, str] = {}   # predicted_command → pre-computed answer
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        self._running = False

    def _load_cache(self) -> List[dict]:
        if TWIN_CACHE_PATH.exists():
            try:
                data = json.loads(TWIN_CACHE_PATH.read_text())
                return data.get("history", [])
            except Exception:
                pass
        return []

    def _save_cache(self):
        try:
            TWIN_CACHE_PATH.write_text(json.dumps(
                {"history": self._history[-self.MAX_HISTORY:]}, indent=2
            ))
        except Exception:
            pass

    def record_command(self, command: str, response: str):
        """Call after every user command to build prediction history."""
        with self._lock:
            self._history.append({
                "ts":       time.time(),
                "hour":     time.localtime().tm_hour,
                "command":  command[:200],
                "response": response[:400],
            })
            if len(self._history) > self.MAX_HISTORY:
                self._history.pop(0)
        self._save_cache()

    def _predict_next(self) -> List[str]:
        """Use LLM + history to predict next 3 likely commands."""
        if not self._llm or not self._history:
            return []

        hour    = time.localtime().tm_hour
        recent  = self._history[-10:]
        history_text = "\n".join(
            f"- [{e['hour']}:xx] {e['command'][:80]}" for e in recent
        )

        habits_text = ""
        if self._mirror:
            habits = self._mirror.get_habits_summary()
            top_apps = habits.get(f"{hour}:00", [])
            if top_apps:
                habits_text = f"User's usual apps at {hour}:00: {', '.join(top_apps[:4])}"

        prompt = (
            f"Based on this user's command history and current time ({hour}:00), "
            f"predict the 3 most likely NEXT commands they will type.\n\n"
            f"Recent history:\n{history_text}\n"
            f"{habits_text}\n\n"
            f"Return ONLY a JSON array of 3 short command strings.\n"
            f'Example: ["check email", "open VS Code", "search Python docs"]'
        )
        try:
            raw = self._llm(prompt)
            raw = re.sub(r'```json|```', '', raw).strip()
            predictions = json.loads(raw)
            if isinstance(predictions, list):
                return [str(p) for p in predictions[:3]]
        except Exception as e:
            logger.debug(f"[PredictiveTwin] Predict error: {e}")
        return []

    def _prewarm_loop(self):
        """Background: predict + pre-compute every 5 minutes."""
        self._running = True
        while not self._stop.is_set():
            predictions = self._predict_next()
            if predictions and self._llm:
                top = predictions[0]
                if top not in self._cache:
                    try:
                        answer = self._llm(top)
                        with self._lock:
                            self._cache[top] = answer
                        logger.debug(f"[PredictiveTwin] Pre-warmed: '{top[:40]}'")
                    except Exception as e:
                        logger.debug(f"[PredictiveTwin] Pre-warm error: {e}")
            self._stop.wait(timeout=300)  # every 5 min
        self._running = False

    def get_cached(self, command: str) -> Optional[str]:
        """Return pre-computed answer if available (zero-latency path)."""
        with self._lock:
            # Fuzzy match: check if any cached key is substring of command
            for key, val in self._cache.items():
                if key.lower() in command.lower() or command.lower() in key.lower():
                    logger.info(f"[PredictiveTwin] Cache HIT for '{command[:40]}'")
                    return val
        return None

    def start(self):
        self._stop.clear()
        threading.Thread(
            target=self._prewarm_loop, daemon=True, name="PredictiveTwin"
        ).start()
        logger.info("[PredictiveTwin] Started.")

    def stop(self):
        self._stop.set()

    def get_status(self) -> dict:
        with self._lock:
            return {
                "running":        self._running,
                "history_len":    len(self._history),
                "cache_size":     len(self._cache),
                "cached_commands": list(self._cache.keys()),
            }


# ─────────────────────────────────────────────────────────────────────
# LIQUID UI SIGNAL (#24)
# ─────────────────────────────────────────────────────────────────────
class LiquidUISignal:
    """
    Emits resize/mode events to the Qt Avatar window.
    Subscribers (Qt slots) react to change window size / panel visibility.

    Modes:
      'minimal'  → small floating sphere (casual chat)
      'expanded' → full panel with charts / data view
      'code'     → side-panel with monospace text output
      'alert'    → full-screen urgent overlay
    """

    MODES = {"minimal", "expanded", "code", "alert", "normal"}

    def __init__(self):
        self._subscribers: list = []
        self._current_mode = "normal"
        self._lock = threading.Lock()

    def subscribe(self, callback):
        """Register a callable(mode: str) to receive mode changes."""
        self._subscribers.append(callback)

    def emit(self, mode: str):
        """Emit a mode change. All subscribers are notified."""
        if mode not in self.MODES:
            logger.warning(f"[LiquidUI] Unknown mode: {mode}")
            return
        with self._lock:
            if mode == self._current_mode:
                return
            self._current_mode = mode

        logger.info(f"[LiquidUI] Mode → {mode}")
        for cb in self._subscribers:
            try:
                cb(mode)
            except Exception as e:
                logger.debug(f"[LiquidUI] Subscriber error: {e}")

    def infer_mode(self, command: str, response: str) -> str:
        """Infer the appropriate UI mode from a command/response pair."""
        cmd = command.lower()
        resp_len = len(response)

        if any(k in cmd for k in ["chart", "plot", "graph", "analyse", "data", "compare"]):
            return "expanded"
        if any(k in cmd for k in ["code", "script", "function", "debug", "compile", "write"]):
            return "code"
        if any(k in cmd for k in ["urgent", "alert", "emergency", "critical", "alarm"]):
            return "alert"
        if resp_len < 100 and any(k in cmd for k in ["hi", "hello", "hey", "chat", "talk"]):
            return "minimal"
        return "normal"

    @property
    def current_mode(self) -> str:
        return self._current_mode


# ─────────────────────────────────────────────────────────────────────
# SOCIAL CONTEXT SWITCHER (#101)
# ─────────────────────────────────────────────────────────────────────
class SocialContextSwitcher:
    """
    Detects the active app window and adapts the AI's communication style:

    VS Code / PyCharm / Xcode  → Technical jargon, concise, code-first
    Slack / Discord / Messages  → Casual, empathetic, friendly tone
    Mail / Calendar             → Professional, structured
    Browser                    → Informative, citation-aware
    Default                    → Balanced
    """

    CONTEXT_RULES: Dict[str, dict] = {
        "technical": {
            "apps": ["code", "xcode", "pycharm", "cursor", "vim", "neovim",
                     "terminal", "iterm", "warp"],
            "system_addendum": (
                "The user is actively coding. Use technical terminology, "
                "include code snippets when relevant, be concise and precise. "
                "Skip social pleasantries."
            ),
            "style": "technical",
        },
        "casual": {
            "apps": ["slack", "discord", "messages", "telegram", "whatsapp",
                     "teams", "zoom"],
            "system_addendum": (
                "The user is chatting/communicating. Use casual, warm language. "
                "Be empathetic, conversational, and friendly. "
                "Avoid jargon unless asked."
            ),
            "style": "casual",
        },
        "professional": {
            "apps": ["mail", "outlook", "calendar", "notion", "confluence", "pages"],
            "system_addendum": (
                "The user is doing professional work. "
                "Use structured, professional language. "
                "Be clear, organised, and thorough."
            ),
            "style": "professional",
        },
        "research": {
            "apps": ["safari", "chrome", "firefox", "arc", "brave"],
            "system_addendum": (
                "The user is browsing/researching. "
                "Be informative and cite sources when possible. "
                "Offer to search for more information."
            ),
            "style": "research",
        },
    }

    def __init__(self, motor: Optional[MotorCortex] = None):
        self._motor   = motor
        self._current = "balanced"
        self._lock    = threading.Lock()

    def detect_context(self) -> Tuple[str, str]:
        """
        Returns (context_name, system_addendum).
        Detects active app via MotorCortex.
        """
        active_app = ""
        if self._motor:
            try:
                active_app = self._motor.get_frontmost_app().lower()
            except Exception:
                pass

        for ctx_name, rule in self.CONTEXT_RULES.items():
            if any(app_key in active_app for app_key in rule["apps"]):
                with self._lock:
                    self._current = ctx_name
                return ctx_name, rule["system_addendum"]

        return "balanced", ""

    def augment_system_prompt(self, base_prompt: str) -> str:
        """Inject context-appropriate style addendum into system prompt."""
        ctx_name, addendum = self.detect_context()
        if not addendum:
            return base_prompt
        logger.debug(f"[SocialContext] Active context: {ctx_name}")
        return f"{base_prompt}\n\n[SOCIAL CONTEXT — {ctx_name.upper()}]\n{addendum}"

    @property
    def current_context(self) -> str:
        return self._current


# ─────────────────────────────────────────────────────────────────────
# UNIVERSAL LINGUISTIC LOBE (#111)
# ─────────────────────────────────────────────────────────────────────
class UniversalLinguisticLobe:
    """
    Auto-detects language of incoming text and translates seamlessly.
    Uses LLM for translation (no external API needed).
    Also handles code comments in foreign languages.
    """

    # Simple heuristics for language detection
    # ORDER MATTERS: more specific scripts before CJK overlap
    LANG_PATTERNS: Dict[str, str] = {
        "Arabic":   r'[\u0600-\u06FF]',
        "Japanese": r'[\u3040-\u30ff]',   # hiragana/katakana — before Chinese
        "Korean":   r'[\uac00-\ud7af]',
        "Chinese":  r'[\u4e00-\u9fff]',   # CJK unified — after Japanese
        "Hindi":    r'[\u0900-\u097F]',
        "Telugu":   r'[\u0C00-\u0C7F]',
        "Tamil":    r'[\u0B80-\u0BFF]',
        "Russian":  r'[\u0400-\u04FF]',
        "Greek":    r'[\u0370-\u03FF]',
        "Thai":     r'[\u0E00-\u0E7F]',
    }

    # Common non-English Latin patterns
    LATIN_PATTERNS: Dict[str, List[str]] = {
        "Spanish":  ["el ", "la ", "los ", "las ", "que ", "con ", "para ", "una "],
        "French":   ["le ", "la ", "les ", "des ", "que ", "une ", "avec ", "pour "],
        "German":   ["der ", "die ", "das ", "und ", "mit ", "für ", "ein ", "eine "],
        "Portuguese": ["para ", "com ", "uma ", "que ", "não ", "por ", "ela "],
        "Italian":  ["della ", "degli ", "alle ", "con ", "per ", "una ", "che "],
    }

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        self._llm = llm_fn
        self._detected_lang = "English"

    def detect_language(self, text: str) -> str:
        """Detect language from text. Returns language name string."""
        # Unicode script detection
        for lang, pattern in self.LANG_PATTERNS.items():
            if re.search(pattern, text):
                self._detected_lang = lang
                return lang

        # Latin-script language heuristics
        text_lower = text.lower()
        scores: Dict[str, int] = defaultdict(int)
        for lang, keywords in self.LATIN_PATTERNS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[lang] += 1

        if scores:
            best = max(scores, key=lambda k: scores[k])
            if scores[best] >= 1:
                self._detected_lang = best
                return best

        return "English"

    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English using LLM."""
        if not self._llm or source_lang == "English":
            return text
        prompt = (
            f"Translate this {source_lang} text to English accurately. "
            f"Return ONLY the English translation:\n\n{text}"
        )
        try:
            return self._llm(prompt).strip()
        except Exception as e:
            logger.warning(f"[LinguisticLobe] Translate error: {e}")
            return text

    def process(self, text: str) -> Tuple[str, str, bool]:
        """
        Detect language, translate if not English.
        Returns (processed_text, detected_lang, was_translated).
        """
        lang = self.detect_language(text)
        if lang == "English":
            return text, lang, False

        logger.info(f"[LinguisticLobe] Detected {lang} — translating to English.")
        translated = self.translate_to_english(text, lang)
        return translated, lang, True

    @property
    def last_detected_language(self) -> str:
        return self._detected_lang



# ─────────────────────────────────────────────────────────────────────
# DIGITAL TWIN — EMA user personality model (Cell 6 / Level IV)
# ─────────────────────────────────────────────────────────────────────
class DigitalTwin:
    """
    Tracks a running estimate of the user's personality/preference vector
    using exponential moving average over interaction embeddings.

    Body-side role: the edge node records every command/response pair
    so the twin drifts toward the user's actual patterns over time.
    The accumulated vector can be serialised and synced to the cloud
    brain to personalise LLM system prompts.

    Notebook source: Cell 6 / LEVEL IV — EXECUTIVE CORTEX
    Original line:   class DigitalTwin

    Works with torch if available; falls back to a pure-float list
    representation so it never crashes on devices without PyTorch.
    """

    DIM = 64
    DECAY = 0.9    # EMA decay (0.9 = slow drift, 0.5 = fast adaptation)

    def __init__(self, dim: int = DIM, decay: float = DECAY):
        self._dim   = dim
        self._decay = decay
        self._torch = _TORCH_OK

        if self._torch:
            self.user_personality_vector = torch.zeros(1, dim)
        else:
            self.user_personality_vector: List[float] = [0.0] * dim

        self._interaction_count = 0

    def record_interaction(self, v):
        """
        Update personality vector with new interaction embedding v.
        v can be a torch.Tensor (shape [1, dim]) or a list/tuple of floats.
        Returns the updated personality vector in the same type as v.
        """
        self._interaction_count += 1

        if self._torch and isinstance(v, torch.Tensor):
            self.user_personality_vector = (
                self._decay * self.user_personality_vector + (1 - self._decay) * v
            )
            return self.user_personality_vector

        # Pure-float path
        v_list = list(v) if not isinstance(v, list) else v
        if len(v_list) != self._dim:
            logger.warning(
                f"[DigitalTwin] Expected dim={self._dim}, got {len(v_list)}. Truncating/padding."
            )
            v_list = (v_list + [0.0] * self._dim)[: self._dim]

        self.user_personality_vector = [
            self._decay * p + (1 - self._decay) * vi
            for p, vi in zip(self.user_personality_vector, v_list)
        ]
        return self.user_personality_vector

    def similarity_to(self, other_v) -> float:
        """
        Cosine similarity between current personality vector and other_v.
        Returns float in [-1, 1].  0.0 on failure.
        """
        if self._torch and isinstance(other_v, torch.Tensor):
            try:
                return float(
                    _F.cosine_similarity(
                        self.user_personality_vector, other_v
                    ).item()
                )
            except Exception:
                return 0.0

        if _NP_OK:
            a = _np.array(self.user_personality_vector, dtype=float).flatten()
            b = _np.array(other_v, dtype=float).flatten()
            denom = (_np.linalg.norm(a) * _np.linalg.norm(b)) + 1e-9
            return float(_np.dot(a, b) / denom)

        # Scalar fallback
        dot = sum(ai * bi for ai, bi in zip(self.user_personality_vector, other_v))
        na  = sum(x ** 2 for x in self.user_personality_vector) ** 0.5
        nb  = sum(x ** 2 for x in other_v) ** 0.5
        return dot / (na * nb + 1e-9)

    def reset(self):
        """Zero the personality vector (e.g. new user session)."""
        if self._torch:
            self.user_personality_vector = torch.zeros(1, self._dim)
        else:
            self.user_personality_vector = [0.0] * self._dim
        self._interaction_count = 0

    def to_list(self) -> List[float]:
        """Serialisable representation — safe to JSON-dump."""
        if self._torch and isinstance(self.user_personality_vector, torch.Tensor):
            return self.user_personality_vector.squeeze().tolist()
        return list(self.user_personality_vector)

    def get_status(self) -> dict:
        vec = self.to_list()
        return {
            "dim":                self._dim,
            "decay":              self._decay,
            "interaction_count":  self._interaction_count,
            "torch_backend":      self._torch,
            "vector_norm":        float(sum(x ** 2 for x in vec) ** 0.5),
        }


# ─────────────────────────────────────────────────────────────────────
# SILICON DREAMER — Monte Carlo design evaluator (Cell 6 / Level IV)
# ─────────────────────────────────────────────────────────────────────
class SiliconDreamer:
    """
    Runs lightweight Monte Carlo simulations ("dreams") to stress-test
    proposed designs before committing them to physical execution.

    Body-side role: before the Motor Cortex executes a multi-step plan,
    the Silicon Dreamer can pre-evaluate its stability by simulating
    N random outcomes.  Plans with a success rate below the threshold
    are flagged for human review.

    Notebook source: Cell 6 / LEVEL IV — EXECUTIVE CORTEX
    Original lines:  class SiliconDreamer

    Requires numpy for randomness.  Falls back to Python's random module
    so it remains functional on minimal edge devices.
    """

    # Physics constants baked in from the notebook
    _GRAVITY   = 9.81
    _DRAG      = 0.05
    _NOISE_STD = 0.1

    def __init__(
        self,
        default_thrust: float = 12.5,
        noise_std: float = 0.1,
    ):
        self._default_thrust = default_thrust
        self._noise_std      = noise_std
        self._gravity        = 9.81
        self._drag           = 0.05
        self._dream_history:  List[dict] = []

    def simulate_dream(self, thrust: float, integrity: float) -> float:
        """
        Single Monte Carlo sample.
        Returns net force = (thrust × integrity) − (gravity + drag + noise).
        Positive → design survives this iteration; negative → it fails.

        Matches notebook exactly:
            return (thrust * integrity) - (9.81 + 0.05 + np.random.normal(0, 0.1))
        """
        if _NP_OK:
            noise = float(_np.random.normal(0, self._noise_std))
        else:
            import random
            noise = random.gauss(0, self._noise_std)

        return (thrust * integrity) - (self._gravity + self._drag + noise)

    def evaluate_design(
        self,
        integrity: float,
        iterations: int = 100,
        thrust: Optional[float] = None,
    ) -> tuple:
        """
        Run `iterations` dream simulations and return (successes, variance).

        Matches notebook exactly:
            dreams = np.array([self.simulate_dream(12.5, integrity) for _ in range(iterations)])
            return np.sum(dreams > 0), np.var(dreams)

        Returns (successes: int, variance: float).
        """
        t = thrust if thrust is not None else self._default_thrust
        samples = [self.simulate_dream(t, integrity) for _ in range(iterations)]

        if _NP_OK:
            arr      = _np.array(samples)
            successes = int(_np.sum(arr > 0))
            variance  = float(_np.var(arr))
        else:
            successes = sum(1 for s in samples if s > 0)
            mean      = sum(samples) / len(samples)
            variance  = sum((s - mean) ** 2 for s in samples) / len(samples)

        self._dream_history.append({
            "integrity":  integrity,
            "thrust":     t,
            "iterations": iterations,
            "successes":  successes,
            "variance":   variance,
            "pass_rate":  successes / iterations,
        })

        return successes, variance

    def evaluate_plan(
        self,
        integrity: float,
        iterations: int = 200,
        pass_threshold: float = 0.75,
    ) -> dict:
        """
        High-level plan evaluation for the Motor Cortex gate.
        Returns a dict with pass/fail verdict and confidence metrics.
        """
        successes, variance = self.evaluate_design(integrity, iterations)
        pass_rate = successes / iterations
        return {
            "pass":        pass_rate >= pass_threshold,
            "pass_rate":   round(pass_rate, 4),
            "variance":    round(variance, 6),
            "successes":   successes,
            "iterations":  iterations,
            "threshold":   pass_threshold,
            "verdict":     "APPROVED" if pass_rate >= pass_threshold else "FLAGGED",
        }

    def clear_history(self):
        self._dream_history.clear()

    def get_status(self) -> dict:
        last = self._dream_history[-1] if self._dream_history else {}
        return {
            "numpy_backend":  _NP_OK,
            "dreams_run":     len(self._dream_history),
            "last_verdict":   last.get("verdict", last.get("pass_rate", "—")),
            "default_thrust": self._default_thrust,
            "noise_std":      self._noise_std,
        }


# ─────────────────────────────────────────────────────────────────────
# FATE WEAVER — Simulated annealing timeline optimizer (Cell 6 / Level IV)
# ─────────────────────────────────────────────────────────────────────
class FateWeaver:
    """
    Uses simulated annealing to collapse the optimal execution timeline
    from a set of candidate routes/plans.

    Body-side role: when the edge node has multiple candidate action
    sequences (e.g. different orderings of IoT commands), FateWeaver
    scores them by simulated annealing convergence speed and returns
    the sequence most likely to succeed under current thermal (system
    load) conditions.

    Notebook source: Cell 6 / LEVEL IV — EXECUTIVE CORTEX
    Original lines:  class FateWeaver

    Pure stdlib — no torch or numpy required.
    """

    def __init__(
        self,
        initial_temp:  float = 100.0,
        cooling_rate:  float = 0.95,
        stop_temp:     float = 0.1,
    ):
        self._initial_temp = initial_temp
        self._cooling_rate = cooling_rate
        self._stop_temp    = stop_temp
        self._anneal_log:  List[dict] = []

    def anneal(
        self,
        route_length:  int,
        initial_temp:  Optional[float] = None,
        cooling_rate:  Optional[float] = None,
    ) -> tuple:
        """
        Run simulated annealing until temperature drops below stop_temp.
        Returns ('OPTIMAL_TIMELINE_COLLAPSED', steps_taken).

        Matches notebook exactly:
            temp, steps = initial_temp, 0
            while temp > 0.1: temp *= cooling_rate; steps += 1
            return 'OPTIMAL_TIMELINE_COLLAPSED', steps
        """
        temp  = initial_temp  if initial_temp  is not None else self._initial_temp
        rate  = cooling_rate  if cooling_rate  is not None else self._cooling_rate
        steps = 0

        while temp > self._stop_temp:
            temp  *= rate
            steps += 1

        self._anneal_log.append({
            "route_length": route_length,
            "steps":        steps,
            "final_temp":   round(temp, 6),
        })

        return "OPTIMAL_TIMELINE_COLLAPSED", steps

    def rank_plans(self, plans: List[dict]) -> List[dict]:
        """
        Score a list of plan dicts by annealing convergence speed.
        Each plan dict should have an 'integrity' key (float 0–1).
        Returns the list sorted by convergence speed (fewest steps = best).
        """
        scored = []
        for plan in plans:
            integrity   = float(plan.get("integrity", 0.5))
            route_len   = len(plan.get("steps", [plan]))
            # Higher integrity → warmer start, faster convergence
            start_temp  = self._initial_temp * (1.0 + integrity)
            _, steps    = self.anneal(route_len, initial_temp=start_temp)
            scored.append({**plan, "_anneal_steps": steps})

        scored.sort(key=lambda p: p["_anneal_steps"])
        return scored

    def get_best_timeline(self, plans: List[dict]) -> Optional[dict]:
        """Return the single best plan after annealing rank, or None."""
        ranked = self.rank_plans(plans)
        return ranked[0] if ranked else None

    def clear_log(self):
        self._anneal_log.clear()

    def get_status(self) -> dict:
        last = self._anneal_log[-1] if self._anneal_log else {}
        return {
            "initial_temp":  self._initial_temp,
            "cooling_rate":  self._cooling_rate,
            "stop_temp":     self._stop_temp,
            "anneals_run":   len(self._anneal_log),
            "last_steps":    last.get("steps", 0),
        }



class UniversalSignalBridge:
    """
    Controls physical room IoT devices (Philips Hue / HomeAssistant).
    Gracefully falls back to dry-run if not configured.
    """

    def __init__(self):
        self.ha_url = os.getenv("HA_URL", "http://homeassistant.local:8123")
        self.ha_token = os.getenv("HA_TOKEN", "")

    def set_room_mood(self, mood: str) -> dict:
        if not self.ha_token:
            logger.info(f"💡 [IoT] DRY RUN: Dimming/Setting physical room lights to '{mood}'")
            return {"status": "dry_run", "mood": mood}

        # Example HomeAssistant integration
        import urllib.request
        import json

        # Map AI states to lighting states
        brightness = 25 if mood == "sleep" else 255
        color_temp = 500 if mood == "sleep" else 250  # Warmer for sleep

        payload = json.dumps({
            "entity_id": "light.room_main",
            "brightness": brightness,
            "color_temp": color_temp
        }).encode('utf-8')

        req = urllib.request.Request(
            f"{self.ha_url}/api/services/light/turn_on",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.ha_token}",
                "Content-Type": "application/json"
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=2.0) as response:
                return {"status": "success", "ha_code": response.getcode()}
        except Exception as e:
            logger.warning(f"💡 [IoT] Failed to set room lights: {e}")
            return {"status": "error", "error": str(e)}


# ─────────────────────────────────────────────────────────────────────
# P2P ORCHESTRATION LOBE (Mobile Tether)
# ─────────────────────────────────────────────────────────────────────
class P2POrchestrationLobe:
    """Handles incoming data from iPhone Shortcuts (Background Tether)."""

    def handle_arrival(self, payload: dict) -> dict:
        note = payload.get("latest_note", "")
        location = payload.get("location", "Unknown")
        battery = payload.get("battery", 100)

        logger.info(f"📱 [P2P Tether] User arrived at {location}. Battery: {battery}%")
        if note:
            logger.info(f"📱 [P2P Tether] Ingested Mobile Note: {note[:50]}...")

        return {"status": "processed", "location": location, "note": note}

# ─────────────────────────────────────────────────────────────────────
# PROACTIVE AGENCY — Top-level facade
# ─────────────────────────────────────────────────────────────────────
class ProactiveAgency:
    """
    Unified facade — wires all proactive subsystems together.

    Call from EdgeNodeOrchestrator (swayambhu_v13.py):
        agency = ProactiveAgency(motor, speak_fn, llm_fn, confirm_fn)
        agency.start()
        # On every command:
        system_prompt = agency.augment_system_prompt(base_prompt)
        cached = agency.get_cached_answer(command)
        agency.on_command_response(command, response)
        agency.emit_ui_mode(command, response)
    """

    def __init__(
        self,
        speak_fn:    Optional[Callable[[str], None]] = None,
        llm_fn:      Optional[Callable[[str], str]]  = None,
        confirm_fn:  Optional[Callable[[str], bool]] = None,
    ):
        self.motor          = MotorCortex(confirm_fn=confirm_fn)
        self.executive      = ExecutiveFunction(self.motor, speak_fn, llm_fn)
        self.mirror         = NeuralMirror(self.motor, speak_fn, llm_fn)
        self.twin           = PredictiveTwin(llm_fn, self.mirror)
        self.liquid_ui      = LiquidUISignal()
        self.social_ctx     = SocialContextSwitcher(self.motor)
        self.linguistic     = UniversalLinguisticLobe(llm_fn)
        # ── Cell 6 / Level IV migrants ────────────────────────────────
        self.digital_twin    = DigitalTwin()
        self.silicon_dreamer = SiliconDreamer()
        self.fate_weaver     = FateWeaver()

    def start(self):
        """Start all background subsystems."""
        self.executive.start()
        self.mirror.start()
        self.twin.start()
        logger.info("[ProactiveAgency] All subsystems started.")

    def stop(self):
        self.executive.stop()
        self.mirror.stop()
        self.twin.stop()

    def preprocess_command(self, command: str) -> Tuple[str, str, bool]:
        """
        Preprocess incoming command:
        1. Detect & translate language if needed
        2. Record for habit/twin learning
        Returns (processed_command, detected_lang, was_translated).
        """
        processed, lang, translated = self.linguistic.process(command)
        return processed, lang, translated

    def augment_system_prompt(self, base_prompt: str) -> str:
        """Add social context style to system prompt."""
        return self.social_ctx.augment_system_prompt(base_prompt)

    def get_cached_answer(self, command: str) -> Optional[str]:
        """Return zero-latency pre-computed answer if available."""
        return self.twin.get_cached(command)

    def on_command_response(self, command: str, response: str):
        """Call after every command/response to train twin + mirror."""
        self.twin.record_command(command, response)
        self.mirror._record_sample()

    def emit_ui_mode(self, command: str, response: str):
        """Infer and emit Liquid UI mode change."""
        mode = self.liquid_ui.infer_mode(command, response)
        self.liquid_ui.emit(mode)

    def read_mail_summary(self) -> str:
        return self.executive.get_unread_mail_summary()

    def get_calendar_events(self, hours: int = 24) -> List[dict]:
        return self.executive.get_upcoming_events(hours)

    def click_menu(self, app: str, menu: str, item: str) -> dict:
        return self.motor.click_menu(app, menu, item)

    def open_app(self, app_name: str) -> dict:
        return self.motor.open_app(app_name)

    def type_text(self, app: str, text: str) -> dict:
        return self.motor.type_text(app, text)

    # ── Cell 6 / Level IV facade methods ─────────────────────────────

    def record_interaction(self, v) -> list:
        """
        Update the DigitalTwin EMA personality vector with interaction
        embedding v (torch.Tensor [1, 64] or list of floats).
        Returns the updated vector as a plain list (always serialisable).
        """
        result = self.digital_twin.record_interaction(v)
        if _TORCH_OK and isinstance(result, torch.Tensor):
            return result.squeeze().tolist()
        return list(result)

    def evaluate_design(
        self,
        integrity: float,
        iterations: int = 100,
        thrust: Optional[float] = None,
    ) -> dict:
        """
        Run SiliconDreamer Monte Carlo evaluation on a proposed plan.
        Returns full verdict dict with pass/fail, pass_rate, variance.
        """
        return self.silicon_dreamer.evaluate_plan(
            integrity, iterations=iterations
        )

    def anneal_timeline(
        self,
        route_length: int = 10,
        initial_temp: Optional[float] = None,
        cooling_rate: Optional[float] = None,
    ) -> dict:
        """
        Run FateWeaver simulated annealing to collapse the optimal timeline.
        Returns {'result': 'OPTIMAL_TIMELINE_COLLAPSED', 'steps': N}.
        """
        verdict, steps = self.fate_weaver.anneal(
            route_length, initial_temp=initial_temp, cooling_rate=cooling_rate
        )
        return {"result": verdict, "steps": steps}

    def rank_plans(self, plans: List[dict]) -> List[dict]:
        """Delegate plan ranking to FateWeaver."""
        return self.fate_weaver.rank_plans(plans)

    def get_status(self) -> dict:
        return {
            "motor":           self.motor.get_status(),
            "executive":       self.executive.get_status(),
            "mirror":          self.mirror.get_status(),
            "twin":            self.twin.get_status(),
            "liquid_ui":       {"current_mode": self.liquid_ui.current_mode},
            "social_ctx":      {"current": self.social_ctx.current_context},
            "linguistic":      {"last_lang": self.linguistic.last_detected_language},
            "digital_twin":    self.digital_twin.get_status(),
            "silicon_dreamer": self.silicon_dreamer.get_status(),
            "fate_weaver":     self.fate_weaver.get_status(),
        }


# ── Module-level singleton ────────────────────────────────────────────
_agency: Optional[ProactiveAgency] = None


def get_proactive_agency(**kwargs) -> ProactiveAgency:
    global _agency
    if _agency is None:
        _agency = ProactiveAgency(**kwargs)
    return _agency


# ── Self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    print("👁️  ProactiveAgency self-test\n")

    # ── Mock LLM ──────────────────────────────────────────────────────
    def mock_llm(prompt: str) -> str:
        if "predict" in prompt.lower() or "next command" in prompt.lower():
            return '["open VS Code", "check email", "run tests"]'
        if "translate" in prompt.lower():
            return "This is the English translation."
        if "summarise" in prompt.lower() or "email" in prompt.lower():
            return "You have 3 unread emails. One is urgent from your manager."
        if "meeting" in prompt.lower() or "calendar" in prompt.lower():
            return "Team standup in 15 minutes."
        return f"Mock response: {prompt[:40]}"

    confirmed_items: list = []
    def mock_confirm(msg: str) -> bool:
        confirmed_items.append(msg)
        return False   # decline all actions in test — safe

    # ── Test 1: Motor Cortex ──────────────────────────────────────────
    print("=== Test 1: Motor Cortex ===")
    motor = MotorCortex(confirm_fn=mock_confirm)
    result = motor.click_menu("Finder", "File", "New Window")
    assert result["status"] == "DECLINED", f"Should be DECLINED: {result}"
    result_open = motor.open_app("Terminal")
    assert result_open["status"] == "DECLINED"
    print(f"✅ Motor Cortex: actions properly require confirmation. Confirmed prompts: {len(confirmed_items)}")

    # ── Test 2: Language Detection ────────────────────────────────────
    print("\n=== Test 2: Universal Linguistic Lobe ===")
    ling = UniversalLinguisticLobe(llm_fn=mock_llm)

    tests = [
        ("Hello, how are you?",          "English",  False),
        ("مرحبا كيف حالك",               "Arabic",   True),
        ("こんにちは、元気ですか",           "Japanese", True),
        ("Hola, ¿cómo estás? para una",  "Spanish",  True),
        ("你好，最近怎么样",               "Chinese",  True),
    ]
    for text, expected_lang, expect_translate in tests:
        processed, lang, translated = ling.process(text)
        # Language detection check
        assert lang == expected_lang, f"Expected {expected_lang}, got {lang} for: {text[:30]}"
        assert translated == expect_translate, \
            f"translate flag wrong for {lang}: expected {expect_translate}, got {translated}"
        print(f"  ✅ '{text[:30]}' → {lang} translated={translated}")

    # ── Test 3: Social Context Switcher ───────────────────────────────
    print("\n=== Test 3: Social Context Switcher ===")
    social = SocialContextSwitcher(motor=None)
    ctx_name, addendum = social.detect_context()
    # No motor → balanced context
    assert ctx_name == "balanced", f"Expected balanced without motor, got {ctx_name}"
    augmented = social.augment_system_prompt("You are Swayambhu.")
    assert "Swayambhu" in augmented
    print(f"✅ Context: {ctx_name}. Prompt augmentation: {len(augmented)} chars")

    # ── Test 4: Liquid UI Signal ──────────────────────────────────────
    print("\n=== Test 4: Liquid UI Signal ===")
    ui_events: list = []
    liquid = LiquidUISignal()
    liquid.subscribe(lambda mode: ui_events.append(mode))

    mode_map = [
        ("show me a chart of the data", "long response " * 20, "expanded"),
        ("write a Python script",       "def foo(): pass",     "code"),
        ("hey how are you",             "hi!",                  "minimal"),
        ("urgent alert critical now",   "response",            "alert"),
    ]
    for cmd, resp, expected_mode in mode_map:
        inferred = liquid.infer_mode(cmd, resp)
        liquid.emit(inferred)
        assert inferred == expected_mode, f"Expected {expected_mode}, got {inferred} for '{cmd}'"
        print(f"  ✅ '{cmd[:30]}' → mode={inferred}")

    assert len(ui_events) == len(mode_map), f"Expected {len(mode_map)} events, got {len(ui_events)}"
    print(f"✅ Liquid UI: {len(ui_events)} mode events emitted correctly")

    # ── Test 5: Predictive Twin ───────────────────────────────────────
    print("\n=== Test 5: Predictive Twin ===")
    twin = PredictiveTwin(llm_fn=mock_llm)
    twin._history.clear()   # ensure clean slate regardless of cached file
    twin.record_command("open VS Code", "Opened VS Code.")
    twin.record_command("run pytest",   "All tests passed.")
    twin.record_command("check email",  "3 unread emails.")
    assert len(twin._history) == 3, f"Expected 3, got {len(twin._history)}"
    hit = twin.get_cached("open VS Code")
    print(f"✅ Twin: {len(twin._history)} commands recorded, cache={'HIT' if hit else 'MISS'}")

    # ── Test 6: Full ProactiveAgency facade ───────────────────────────
    print("\n=== Test 6: ProactiveAgency Facade ===")
    agency = ProactiveAgency(
        speak_fn=lambda t: None,
        llm_fn=mock_llm,
        confirm_fn=lambda msg: False,
    )

    # preprocess Arabic → should translate
    cmd_ar = "مرحبا، ما الجديد"
    processed, lang, translated = agency.preprocess_command(cmd_ar)
    assert lang == "Arabic"
    assert translated
    print(f"  ✅ Arabic command processed: lang={lang} translated={translated}")

    # preprocess English → passthrough
    cmd_en = "What's the weather today?"
    processed_en, lang_en, trans_en = agency.preprocess_command(cmd_en)
    assert lang_en == "English"
    assert not trans_en
    print(f"  ✅ English command passthrough: lang={lang_en}")

    # augment system prompt
    base = "You are Swayambhu."
    augmented_prompt = agency.augment_system_prompt(base)
    assert "Swayambhu" in augmented_prompt
    print(f"  ✅ System prompt augmented: {len(augmented_prompt)} chars")

    # emit UI mode
    agency.emit_ui_mode("analyse this data chart", "Here is the analysis...")
    assert agency.liquid_ui.current_mode == "expanded"
    print(f"  ✅ UI mode after chart command: {agency.liquid_ui.current_mode}")

    # record command
    before = len(agency.twin._history)
    agency.on_command_response("open VS Code", "VS Code opened.")
    assert len(agency.twin._history) == before + 1
    print(f"  ✅ Twin history: {len(agency.twin._history)} entry")

    # status
    status = agency.get_status()
    assert "motor" in status
    assert "executive" in status
    assert "mirror" in status
    assert "twin" in status
    assert "liquid_ui" in status
    assert "social_ctx" in status
    assert "linguistic" in status
    print(f"  ✅ Status keys: {list(status.keys())}")

    print("\n✅ All ProactiveAgency tests passed.")

    # ── Test 7: DigitalTwin ───────────────────────────────────────────
    print("\n=== Test 7: DigitalTwin ===")
    dt = DigitalTwin(dim=8, decay=0.9)

    # Initial vector is zeros
    assert dt.to_list() == [0.0] * 8, "Initial vector should be all zeros"

    # Float-list interaction
    v1 = [1.0] * 8
    result1 = dt.record_interaction(v1)
    assert len(result1) == 8, "Result must have dim=8 elements"
    # After one update: 0.9*0 + 0.1*1.0 = 0.1
    assert abs(result1[0] - 0.1) < 1e-6, f"Expected 0.1 after first update, got {result1[0]}"
    assert dt._interaction_count == 1

    # Second interaction — EMA drifts further
    result2 = dt.record_interaction(v1)
    # 0.9*0.1 + 0.1*1.0 = 0.19
    assert abs(result2[0] - 0.19) < 1e-6, f"Expected 0.19 after second update, got {result2[0]}"
    assert dt._interaction_count == 2

    # Serialisable
    serialised = dt.to_list()
    assert all(isinstance(x, float) for x in serialised), "to_list must return floats"

    # Similarity to identical vector → ~1.0
    sim = dt.similarity_to(result2)
    assert abs(sim - 1.0) < 1e-4, f"Self-similarity should be ~1.0, got {sim}"

    # Similarity to opposite vector → ~-1.0
    neg = [-x for x in result2]
    sim_neg = dt.similarity_to(neg)
    assert sim_neg < -0.9, f"Opposite similarity should be ~-1.0, got {sim_neg}"

    # Reset clears vector and count
    dt.reset()
    assert dt._interaction_count == 0
    assert all(abs(x) < 1e-9 for x in dt.to_list()), "Reset must zero the vector"

    # Status dict
    st = dt.get_status()
    assert st["dim"] == 8
    assert st["decay"] == 0.9
    assert st["interaction_count"] == 0
    assert "vector_norm" in st
    print(f"  ✅ DigitalTwin: EMA, similarity, reset, status all correct")

    # Torch path (if available)
    if _TORCH_OK:
        dt_torch = DigitalTwin(dim=16, decay=0.8)
        v_t = torch.ones(1, 16)
        r_t = dt_torch.record_interaction(v_t)
        assert isinstance(r_t, torch.Tensor), "Torch input should yield Tensor output"
        assert r_t.shape == (1, 16), f"Wrong shape: {r_t.shape}"
        expected_val = 0.2  # 0.8*0 + 0.2*1.0
        assert abs(r_t[0, 0].item() - expected_val) < 1e-5
        print(f"  ✅ DigitalTwin: torch path correct (val={r_t[0,0].item():.4f})")
    else:
        print(f"  ℹ️  DigitalTwin: torch not available — float path only")

    print(f"✅ Test 7 complete")

    # ── Test 8: SiliconDreamer ────────────────────────────────────────
    print("\n=== Test 8: SiliconDreamer ===")
    sd = SiliconDreamer(default_thrust=12.5, noise_std=0.1)

    # Single dream — returns a float
    dream = sd.simulate_dream(thrust=12.5, integrity=1.0)
    assert isinstance(dream, float), f"simulate_dream must return float, got {type(dream)}"
    # 12.5 * 1.0 - (9.81 + 0.05 + tiny_noise) ≈ 2.64
    assert 0.0 < dream < 5.0, f"Expected dream in (0, 5), got {dream}"

    # evaluate_design returns (successes, variance)
    successes, variance = sd.evaluate_design(integrity=1.0, iterations=500)
    assert isinstance(successes, int), "successes must be int"
    assert isinstance(variance, float), "variance must be float"
    # At full integrity with default thrust, vast majority should succeed
    assert successes > 400, f"Expected >400/500 successes at integrity=1.0, got {successes}"

    # Low integrity → most fail
    s_low, _ = sd.evaluate_design(integrity=0.5, iterations=500)
    # 12.5 * 0.5 = 6.25, gravity+drag = 9.86 → almost all fail
    assert s_low < 50, f"Expected <50/500 successes at integrity=0.5, got {s_low}"

    # evaluate_plan returns full verdict dict
    verdict = sd.evaluate_plan(integrity=1.0, iterations=200, pass_threshold=0.75)
    assert verdict["pass"] is True, f"High integrity should pass: {verdict}"
    assert "pass_rate" in verdict and "variance" in verdict
    assert "verdict" in verdict and verdict["verdict"] == "APPROVED"

    verdict_fail = sd.evaluate_plan(integrity=0.5, iterations=200, pass_threshold=0.75)
    assert verdict_fail["pass"] is False, f"Low integrity should fail: {verdict_fail}"
    assert verdict_fail["verdict"] == "FLAGGED"

    # History accumulates
    assert len(sd._dream_history) >= 2

    # clear_history resets
    sd.clear_history()
    assert len(sd._dream_history) == 0

    # get_status
    st = sd.get_status()
    assert "numpy_backend" in st
    assert "default_thrust" in st
    assert st["default_thrust"] == 12.5
    print(f"  ✅ SiliconDreamer: simulate_dream, evaluate_design, evaluate_plan, status correct")
    print(f"✅ Test 8 complete")

    # ── Test 9: FateWeaver ────────────────────────────────────────────
    print("\n=== Test 9: FateWeaver ===")
    fw = FateWeaver(initial_temp=100.0, cooling_rate=0.95, stop_temp=0.1)

    # anneal returns tuple
    result_str, steps = fw.anneal(route_length=10)
    assert result_str == "OPTIMAL_TIMELINE_COLLAPSED", f"Wrong return: {result_str}"
    assert isinstance(steps, int) and steps > 0, f"steps must be positive int, got {steps}"

    # Steps are deterministic (no randomness in anneal)
    _, steps2 = fw.anneal(route_length=10)
    assert steps == steps2, "anneal steps must be deterministic"

    # Compute expected steps manually
    temp, expected_steps = 100.0, 0
    while temp > 0.1:
        temp *= 0.95
        expected_steps += 1
    assert steps == expected_steps, f"Expected {expected_steps} steps, got {steps}"

    # Custom params override defaults
    _, steps_fast = fw.anneal(route_length=5, cooling_rate=0.5)
    assert steps_fast < steps, "Faster cooling must converge in fewer steps"

    # rank_plans sorts by annealing convergence
    plans = [
        {"name": "plan_A", "integrity": 0.3, "steps": ["s1", "s2"]},
        {"name": "plan_B", "integrity": 0.9, "steps": ["s1", "s2", "s3"]},
        {"name": "plan_C", "integrity": 0.6, "steps": ["s1"]},
    ]
    ranked = fw.rank_plans(plans)
    assert len(ranked) == 3, "All plans must be ranked"
    assert all("_anneal_steps" in p for p in ranked), "Each plan must have _anneal_steps"
    # Higher integrity → higher start_temp → more steps — verify monotone ordering
    for i in range(len(ranked) - 1):
        assert ranked[i]["_anneal_steps"] <= ranked[i + 1]["_anneal_steps"], \
            "Plans must be sorted ascending by anneal steps"

    # get_best_timeline returns first ranked plan
    best = fw.get_best_timeline(plans)
    assert best is not None
    assert "_anneal_steps" in best
    assert best["_anneal_steps"] == ranked[0]["_anneal_steps"]

    # Log accumulates then clears
    assert len(fw._anneal_log) > 0
    fw.clear_log()
    assert len(fw._anneal_log) == 0

    # get_status
    st = fw.get_status()
    assert st["initial_temp"] == 100.0
    assert st["cooling_rate"] == 0.95
    assert st["anneals_run"] == 0
    print(f"  ✅ FateWeaver: anneal determinism, rank_plans, get_best_timeline, status correct")
    print(f"  ✅ Expected steps={expected_steps}, got {steps}")
    print(f"✅ Test 9 complete")

    # ── Test 10: ProactiveAgency facade with new methods ──────────────
    print("\n=== Test 10: ProactiveAgency facade — new subsystems ===")
    agency2 = ProactiveAgency(
        speak_fn=lambda t: None,
        llm_fn=mock_llm,
        confirm_fn=lambda msg: False,
    )

    # digital_twin exists on agency
    assert hasattr(agency2, "digital_twin")
    assert hasattr(agency2, "silicon_dreamer")
    assert hasattr(agency2, "fate_weaver")

    # record_interaction returns list
    ri = agency2.record_interaction([0.5] * 64)
    assert isinstance(ri, list) and len(ri) == 64
    # EMA: 0.9*0 + 0.1*0.5 = 0.05
    assert abs(ri[0] - 0.05) < 1e-6, f"Expected 0.05, got {ri[0]}"

    # evaluate_design returns verdict dict
    vd = agency2.evaluate_design(integrity=1.0, iterations=100)
    assert isinstance(vd, dict) and "pass" in vd and "pass_rate" in vd
    assert vd["pass"] is True, f"High integrity must pass: {vd}"

    # anneal_timeline returns dict with result and steps
    at = agency2.anneal_timeline(route_length=5)
    assert at["result"] == "OPTIMAL_TIMELINE_COLLAPSED"
    assert isinstance(at["steps"], int) and at["steps"] > 0

    # rank_plans delegates correctly
    test_plans = [
        {"name": "x", "integrity": 0.2, "steps": ["a"]},
        {"name": "y", "integrity": 0.8, "steps": ["a", "b"]},
    ]
    ranked2 = agency2.rank_plans(test_plans)
    assert len(ranked2) == 2
    assert all("_anneal_steps" in p for p in ranked2)

    # get_status includes all 10 keys
    st2 = agency2.get_status()
    for key in ["motor", "executive", "mirror", "twin", "liquid_ui",
                "social_ctx", "linguistic", "digital_twin", "silicon_dreamer", "fate_weaver"]:
        assert key in st2, f"Missing key '{key}' in get_status()"
    assert "interaction_count" in st2["digital_twin"]
    assert "numpy_backend" in st2["silicon_dreamer"]
    assert "anneals_run" in st2["fate_weaver"]
    print(f"  ✅ ProactiveAgency: 10 status keys present: {list(st2.keys())}")
    print(f"  ✅ record_interaction, evaluate_design, anneal_timeline, rank_plans all correct")
    print(f"✅ Test 10 complete")

    print("\n✅ All ProactiveAgency + Cell 6 migration tests passed.")

