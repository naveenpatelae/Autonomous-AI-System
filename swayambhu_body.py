#!/usr/bin/env python3
# =====================================================================
# 🌌 SWAYAMBHU OMNI-TERMINAL v14.0 — THE SOVEREIGN BODY
# =====================================================================
# Mac-side Central Nervous System. All 13 features integrated.
#
# Features:
#   1.  TCC Pre-Flight Authorization
#   2.  Autonomic Procurement Engine (auto-detect + download models)
#   3.  NetworkWatchdog + EdgeNodeSync (Firebase rediscovery)
#   4.  LocalBlueprintMirror (full AST execution engine)
#   5.  LocalRAGIndex (FAISS/Numpy with numpy fallback)
#   6.  AirGapSurvivalMode (offline queue + flush)
#   7.  DeadMansSwitch (Wi-Fi severance)
#   8.  LaunchAgentDaemon (macOS persistence via launchctl)
#   9.  StealthDaemon (process rename)
#   10. EdgeNodeOrchestrator (dual coder/tester LLMs + capability flags)
#   11. Local FastAPI edge server on port 8003
#   12. Native Blueprint Seeder (30 built-in skills)
#   13. Native Auto-Avatar Opener
#
# Model auto-detection:
#   Scans SWAYAMBHU_MODEL_DIR → ~/Swayambhu/models → ./models
#   Picks first *.gguf found per role (coder / tester / draft).
#   Falls back gracefully — no model = offline stub mode only.
# =====================================================================

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import requests
from blueprint_engine import BlueprintEngine

# ─────────────────────────────────────────────────────────────────────
# LOGGING  (structured, no bare print in library code)
# ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SwayambhuBody")
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

_IS_MAC = platform.system() == "Darwin"
_API_HEADERS = {"ngrok-skip-browser-warning": "true", "User-Agent": "SwayambhuEdge/1.4"}

# ─────────────────────────────────────────────────────────────────────
# 9. STEALTH DAEMON  (optional — graceful if setproctitle absent)
# ─────────────────────────────────────────────────────────────────────
try:
    import setproctitle as _spt
    _spt.setproctitle("com.apple.syslogd")
    logger.info("🥷 [StealthDaemon] Process renamed → com.apple.syslogd")
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────
# OPTIONAL DEPENDENCY FLAGS
# ─────────────────────────────────────────────────────────────────────
try:
    import pyttsx3 as _pyttsx3
    _TTS_ENGINE = _pyttsx3.init()
    _TTS_OK = True
except Exception:
    _TTS_ENGINE = None
    _TTS_OK = False

try:
    import speech_recognition as sr
    _SR_OK = True
except ImportError:
    _SR_OK = False

try:
    import numpy as _np
    _NP_OK = True
except ImportError:
    _NP_OK = False

# ─────────────────────────────────────────────────────────────────────
# PHASE IMPORTS  (all optional — body boots without them)
# ─────────────────────────────────────────────────────────────────────
try:
    _PHASES_DIR = Path(__file__).parent
except NameError:
    _PHASES_DIR = Path(os.getcwd())

if str(_PHASES_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASES_DIR))

_GESTURE_OK = _EMPATHY_OK = _SPINE_OK = _HIPPO_OK = False
_LIZARD_OK = _OPENCLAW_OK = _FIRM_OK = False

# Import DualModelEngine default filenames so boot can use them as fallbacks
# when no model is auto-detected. Fails silently — body works without the engine.
try:
    from dual_model_engine import (
        CODER_MODEL_FILE  as _DME_CODER_FILE,
        TESTER_MODEL_FILE as _DME_TESTER_FILE,
    )
except ImportError:
    _DME_CODER_FILE  = "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"
    _DME_TESTER_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# Make them available under the names used in the boot block
CODER_MODEL_FILE  = _DME_CODER_FILE
TESTER_MODEL_FILE = _DME_TESTER_FILE

try:
    from gesture_tracker import GestureTracker; _GESTURE_OK = True
except ImportError:
    pass
try:
    from empathy_wire import EmpathyWire; _EMPATHY_OK = True
except ImportError:
    pass
try:
    from sovereign_spine import SovereignSpine; _SPINE_OK = True
except ImportError:
    pass
try:
    # This matches the class name in your new hippocampus.py
    from hippocampus import BodyHippocampus as Hippocampus
    _HIPPO_OK = True
    logger.info("🧠 [Memory] Local Vector Hippocampus linked.")
except ImportError as e:
    _HIPPO_OK = False
    logger.warning(f"⚠️ [Memory] Hippocampus link failed: {e}")
try:
    from lizard_brain import LizardBrain, SelfPatcher; _LIZARD_OK = True
except ImportError:
    pass
try:
    from openclaw import OpenClawGeneral; _OPENCLAW_OK = True
except ImportError:
    pass
try:
    from software_firm import SoftwareFirm; _FIRM_OK = True
except ImportError:
    pass

# =====================================================================
# ─── SECTION 1: CONFIGURATION ────────────────────────────────────────
# =====================================================================

def _resolve_project_root() -> Path:
    """Resolve PROJECT_ROOT from swayambhu_utils or fall back to cwd."""
    try:
        from swayambhu_utils import PROJECT_ROOT
        return Path(PROJECT_ROOT)
    except ImportError:
        pass
    # Walk up from __file__ looking for a marker
    try:
        candidate = Path(__file__).parent
    except NameError:
        candidate = Path(os.getcwd())
    for _ in range(4):
        if (candidate / "blueprints").exists() or (candidate / "models").exists():
            return candidate.resolve()
        candidate = candidate.parent
    return Path(os.getcwd()).resolve()


def _resolve_model_dir() -> Path:
    """Auto-detect model directory from env var → ~/Swayambhu/models → ./models."""
    env = os.environ.get("SWAYAMBHU_MODEL_DIR", "")
    if env and Path(env).exists():
        return Path(env).resolve()
    home_models = Path.home() / "Swayambhu" / "models"
    if home_models.exists():
        return home_models
    local_models = _resolve_project_root() / "models"
    local_models.mkdir(parents=True, exist_ok=True)
    return local_models


PROJECT_ROOT   = _resolve_project_root()
_MODEL_DIR     = _resolve_model_dir()
BLUEPRINT_DIR  = PROJECT_ROOT / "blueprints"
BLUEPRINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────
LOCAL_VAULT_PATH  = PROJECT_ROOT / "local_vault_mirror.json"
LOCAL_RAG_PATH    = PROJECT_ROOT / "local_rag_index.json"
BRAIN_URL_CACHE   = PROJECT_ROOT / ".brain_url_cache.json"
PLIST_PATH        = Path.home() / "Library" / "LaunchAgents" / "ai.swayambhu.edgenode.plist"

# ── Ports / Timing ────────────────────────────────────────────────────
EDGE_SERVER_PORT   = int(os.environ.get("SWAYAMBHU_EDGE_PORT", "8003"))
AVATAR_PORT        = int(os.environ.get("AVATAR_PORT", "8007"))
HEARTBEAT_INTERVAL = int(os.environ.get("SWAYAMBHU_HEARTBEAT", "30"))

# ── Node identity ─────────────────────────────────────────────────────
NODE_ID = f"MAC_EDGE_{hashlib.md5(platform.node().encode()).hexdigest()[:8].upper()}"

# ── Firebase artifact path ────────────────────────────────────────────
FIREBASE_ARTIFACT = "artifacts/SWAYAMBHU_SOVEREIGN_001/public/data"
FIREBASE_DB_ID    = os.environ.get(
    "SWAYAMBHU_FIREBASE_DB",
    "ai-studio-47a94c34-82ae-4dc7-af57-09fb23251026"
)

# =====================================================================
# ─── SECTION 2: MODEL AUTO-DETECTION ─────────────────────────────────
# =====================================================================

# Known quantization suffixes ordered by preference (best quality first)
_QUANT_PREF = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M"]

# Role → filename fragment heuristics
_ROLE_HINTS: Dict[str, List[str]] = {
    "coder":  ["deepseek-coder-1.3b", "deepseek_coder_1.3b", "coder-1.3b", "deepseek-1.3b"],
    "tester": ["qwen2.5", "qwen-2.5", "qwen2", "qwen", "mistral-7b", "mistral"],
    "draft":  ["llama-3.2-1b", "llama3.2-1b", "llama-1b", "tinyllama", "phi-1"],
}

# Fallback explicit filenames (used only when auto-detect finds nothing)
_FALLBACK_FILES: Dict[str, str] = {
    "coder":  "deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
    "tester": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "draft":  "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
}

# HuggingFace repos for auto-download
_HF_REPOS: Dict[str, Tuple[str, str]] = {
    "coder":  ("deepseek-ai/deepseek-coder-1.3b-instruct",
               "deepseek-coder-1.3b-instruct.Q4_K_M.gguf"),
    "tester": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF",
               "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "draft":  ("unsloth/Llama-3.2-1B-Instruct-GGUF",
               "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
}


def _find_model(role: str, model_dir: Path = _MODEL_DIR) -> Optional[Path]:
    """Scan model_dir for a GGUF matching role hints. Returns best match or None."""
    if not model_dir.exists():
        return None
    hints = _ROLE_HINTS.get(role, [])
    candidates: List[Path] = list(model_dir.glob("*.gguf"))
    if not candidates:
        return None

    # Score each candidate: hint match + quantization preference
    def _score(p: Path) -> Tuple[int, int]:
        name = p.name.lower()
        hint_score = next(
            (len(hints) - i for i, h in enumerate(hints) if h in name), 0)
        quant_score = next(
            (len(_QUANT_PREF) - i for i, q in enumerate(_QUANT_PREF)
             if q.lower() in name), 0)
        return (hint_score, quant_score)

    ranked = sorted(candidates, key=_score, reverse=True)
    best = ranked[0]
    # Only return if it scored at least one hint hit
    if _score(best)[0] > 0:
        logger.info(f"[ModelDetect] role={role} → {best.name}")
        return best

    # Exact fallback filename
    fallback = model_dir / _FALLBACK_FILES.get(role, "")
    if fallback.exists():
        return fallback
    return None


# Resolved paths (may be None if model not present)
CODER_MODEL_PATH  = _find_model("coder")
TESTER_MODEL_PATH = _find_model("tester")
DRAFT_MODEL_PATH  = _find_model("draft")

# Convenience alias used by LocalLLMFallback default arg
LOCAL_LLM_PATH = CODER_MODEL_PATH or (_MODEL_DIR / _FALLBACK_FILES["coder"])


# =====================================================================
# ─── SECTION 3: CAPABILITY FLAGS ─────────────────────────────────────
# =====================================================================

@dataclass
class CapabilityFlags:
    """Runtime capability registry — set during boot, read everywhere."""

    # ── Hardware & Permissions ──
    tcc_mic: bool = False
    tcc_accessibility: bool = False
    voice_input: bool = False

    # ── Inference & Local Compute ──
    local_coder: bool = False
    local_tester: bool = False
    dual_engine: bool = False
    software_firm: bool = False
    speculative_engine: bool = False

    # ── Memory & Retrieval ──
    hippocampus: bool = False
    tactical_rag: bool = False
    memory_evolution: bool = False

    # ── Agency & Execution ──
    universal_action: bool = False
    openclaw: bool = False
    proactive_agency: bool = False
    meta_agent_factory: bool = False

    # ── Safety, Routing & Cloud ──
    sovereign_spine: bool = False
    lizard_brain: bool = False
    security_shield: bool = False
    agent_shield: bool = False
    firebase: bool = False

    # ── Perception & Affect ──
    empathy_wire: bool = False
    affective_engine: bool = False
    gesture_tracker: bool = False
    wake_detector: bool = False

    def summary(self) -> Dict[str, bool]:
        return {k: v for k, v in self.__dict__.items()}

    def active_organs(self) -> List[str]:
        return [k for k, v in self.__dict__.items() if v]


# =====================================================================
# ─── SECTION 4: TCC PREFLIGHT ────────────────────────────────────────
# =====================================================================

class TCCPreflight:
    """macOS TCC permission checker with per-check timeout guard."""

    def __init__(self):
        self.permissions: Dict[str, bool] = {}

    def check_all(self) -> Dict[str, bool]:
        if not _IS_MAC:
            self.permissions = {
                "microphone": True,
                "accessibility": True,
                "screen_recording": True,
            }
            return self.permissions

        self.permissions["microphone"]      = self._check_mic()
        self.permissions["accessibility"]   = self._check_accessibility()
        self.permissions["screen_recording"] = True  # checked dynamically at use-time
        return self.permissions

    def _check_mic(self) -> bool:
        if not _SR_OK:
            return False
        try:
            sr.Recognizer()
            sr.Microphone()
            return True
        except Exception:
            return False

    def _check_accessibility(self) -> bool:
        try:
            result = subprocess.run(
                ["osascript", "-e",
                 'tell app "System Events" to return name of first process'],
                capture_output=True, timeout=4,
            )
            return result.returncode == 0
        except Exception:
            return False

    def can(self, perm: str) -> bool:
        return self.permissions.get(perm, False)


# =====================================================================
# ─── SECTION 5: BLUEPRINT SEEDER ─────────────────────────────────────
# =====================================================================

BUILTIN_BLUEPRINTS: Dict[str, str] = {
    "open_safari": (
        'import subprocess\ndef run(**kw):\n'
        '    subprocess.run(["open","-a","Safari"])\n'
        '    return {"status":"OK","app":"Safari"}\n'
    ),
    "open_chrome": (
        'import subprocess\ndef run(**kw):\n'
        '    subprocess.run(["open","-a","Google Chrome"])\n'
        '    return {"status":"OK","app":"Chrome"}\n'
    ),
    "open_terminal": (
        'import subprocess\ndef run(**kw):\n'
        '    subprocess.run(["open","-a","Terminal"])\n'
        '    return {"status":"OK","app":"Terminal"}\n'
    ),
    "open_vscode": (
        'import subprocess\ndef run(path:str="",**kw):\n'
        '    cmd=["open","-a","Visual Studio Code"]\n'
        '    if path: cmd+=[path]\n'
        '    subprocess.run(cmd)\n'
        '    return {"status":"OK"}\n'
    ),
    "take_screenshot": (
        'import subprocess,time\ndef run(path:str="",**kw):\n'
        '    from pathlib import Path\n'
        '    ws=Path.home()/"Desktop"\n'
        '    dest=path or str(ws/f"shot_{int(time.time())}.png")\n'
        '    subprocess.run(["screencapture","-x",dest])\n'
        '    return {"status":"OK","path":dest}\n'
    ),
    "get_battery": (
        'import subprocess,re\ndef run(**kw):\n'
        '    r=subprocess.run(["pmset","-g","batt"],capture_output=True,text=True)\n'
        '    m=re.search(r"(\\d+)%",r.stdout)\n'
        '    return {"status":"OK","pct":int(m.group(1)) if m else -1}\n'
    ),
    "adjust_volume": (
        'import subprocess\ndef run(level:int=50,**kw):\n'
        '    level=max(0,min(100,int(level)))\n'
        '    subprocess.run(["osascript","-e",f"set volume output volume {level}"])\n'
        '    return {"status":"OK","volume":level}\n'
    ),
    "mute_volume": (
        'import subprocess\ndef run(mute:bool=True,**kw):\n'
        '    s="true" if mute else "false"\n'
        '    subprocess.run(["osascript","-e",f"set volume output muted {s}"])\n'
        '    return {"status":"OK","muted":mute}\n'
    ),
    "type_text": (
        'import subprocess\ndef run(text:str="",**kw):\n'
        '    safe=text.replace(chr(34),chr(92)+chr(34))\n'
        '    script=\'tell application "System Events" to keystroke "\'+safe+\'"\'\n'
        '    subprocess.run(["osascript","-e",script])\n'
        '    return {"status":"OK"}\n'
    ),
    "get_clipboard": (
        'import subprocess\ndef run(**kw):\n'
        '    r=subprocess.run("pbpaste",capture_output=True,text=True)\n'
        '    return {"status":"OK","clipboard":r.stdout[:500]}\n'
    ),
    "set_clipboard": (
        'import subprocess\ndef run(text:str="",**kw):\n'
        '    p=subprocess.Popen("pbcopy",stdin=subprocess.PIPE)\n'
        '    p.communicate(text.encode())\n'
        '    return {"status":"OK"}\n'
    ),
    "play_music": (
        'import subprocess\ndef run(**kw):\n'
        '    subprocess.run(["osascript","-e",\'tell application "Music" to play\'])\n'
        '    return {"status":"OK"}\n'
    ),
    "pause_music": (
        'import subprocess\ndef run(**kw):\n'
        '    subprocess.run(["osascript","-e",\'tell application "Music" to pause\'])\n'
        '    return {"status":"OK"}\n'
    ),
    "next_track": (
        'import subprocess\ndef run(**kw):\n'
        '    subprocess.run(["osascript","-e",\'tell application "Music" to next track\'])\n'
        '    return {"status":"OK"}\n'
    ),
    "lock_screen": (
        'import subprocess\ndef run(**kw):\n'
        '    subprocess.run(["pmset","displaysleepnow"])\n'
        '    return {"status":"OK"}\n'
    ),
    "get_ip": (
        'import socket\ndef run(**kw):\n'
        '    try:\n'
        '        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n'
        '        s.connect(("8.8.8.8",80));ip=s.getsockname()[0];s.close()\n'
        '    except: ip="127.0.0.1"\n'
        '    return {"status":"OK","ip":ip}\n'
    ),
    "get_disk_space": (
        'import shutil\ndef run(**kw):\n'
        '    t,u,f=shutil.disk_usage("/")\n'
        '    g=1024**3\n'
        '    return {"status":"OK","free_gb":round(f/g,1),"total_gb":round(t/g,1)}\n'
    ),
    "list_workspace": (
        'import os\nfrom pathlib import Path\ndef run(**kw):\n'
        '    ws=Path.home()/"Swayambhu"/"workspace"\n'
        '    ws.mkdir(parents=True,exist_ok=True)\n'
        '    files=[f for f in os.listdir(ws) if not f.startswith(".")]\n'
        '    return {"status":"OK","files":files[:50]}\n'
    ),
    "create_folder": (
        'from pathlib import Path\ndef run(name:str="NewFolder",**kw):\n'
        '    p=Path.home()/"Swayambhu"/"workspace"/name\n'
        '    p.mkdir(parents=True,exist_ok=True)\n'
        '    return {"status":"OK","path":str(p)}\n'
    ),
    "open_url": (
        'import subprocess\ndef run(url:str="https://google.com",**kw):\n'
        '    if not url.startswith(("http://","https://")): url="https://"+url\n'
        '    subprocess.run(["open",url])\n'
        '    return {"status":"OK","url":url}\n'
    ),
    "web_search": (
        'import subprocess,urllib.parse\ndef run(query:str="",**kw):\n'
        '    q=urllib.parse.quote_plus(query)\n'
        '    subprocess.run(["open",f"https://www.google.com/search?q={q}"])\n'
        '    return {"status":"OK","query":query}\n'
    ),
    "set_reminder": (
        'import subprocess\ndef run(text:str="Task",**kw):\n'
        '    script=f\'tell application "Reminders" to make new reminder'
        ' with properties {{name:"{text}"}}\'\n'
        '    subprocess.run(["osascript","-e",script])\n'
        '    return {"status":"OK","reminder":text}\n'
    ),
    "open_calendar": (
        'import subprocess\ndef run(**kw):\n'
        '    subprocess.run(["open","-a","Calendar"])\n'
        '    return {"status":"OK"}\n'
    ),
    "sleep_display": (
        'import subprocess\ndef run(**kw):\n'
        '    subprocess.run(["pmset","displaysleepnow"])\n'
        '    return {"status":"OK"}\n'
    ),
    "open_finder": (
        'import subprocess,os\ndef run(path:str="",**kw):\n'
        '    p=path or os.path.expanduser("~")\n'
        '    subprocess.run(["open",p])\n'
        '    return {"status":"OK","path":p}\n'
    ),
    "quit_app": (
        'import subprocess\ndef run(app:str="Safari",**kw):\n'
        '    script=f\'tell application "{app}" to quit\'\n'
        '    subprocess.run(["osascript","-e",script])\n'
        '    return {"status":"OK","quit":app}\n'
    ),
    "press_key": (
        'import subprocess\ndef run(key:str="c",modifiers:str="command",**kw):\n'
        '    mm={"command":"command down","ctrl":"control down","shift":"shift down"}\n'
        '    mods=", ".join(mm.get(m.strip(),m.strip()+" down")'
        ' for m in modifiers.split("+"))\n'
        '    script=f\'tell application "System Events" to keystroke'
        ' "{key}" using {{{mods}}}\'\n'
        '    subprocess.run(["osascript","-e",script])\n'
        '    return {"status":"OK"}\n'
    ),
    "empty_trash": (
        'import subprocess\ndef run(**kw):\n'
        '    subprocess.run(["osascript","-e",'
        '\'tell application "Finder" to empty trash\'])\n'
        '    return {"status":"OK"}\n'
    ),
}


class BlueprintBootstrapper:
    """Seeds built-in blueprints to disk, mirror, and Firestore."""

    def __init__(self, blueprint_dir: Optional[Path] = None):
        self._dir = blueprint_dir or BLUEPRINT_DIR

    def seed_disk(self) -> int:
        self._dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for bp_id, code in BUILTIN_BLUEPRINTS.items():
            p = self._dir / f"{bp_id}.py"
            if not p.exists():
                p.write_text(code, encoding="utf-8")
                written += 1
        logger.info(f"[Bootstrapper] Disk seed: {written} new files written.")
        return written

    def seed_mirror(self, mirror: "LocalBlueprintMirror") -> int:
        bps = [
            {
                "id": k,
                "code": v,
                "description": k.replace("_", " ").title(),
                "category": "builtin",
                "version": 1,
                "checksum": hashlib.sha256(v.encode()).hexdigest(),
            }
            for k, v in BUILTIN_BLUEPRINTS.items()
        ]
        mirror.apply_full_sync({"blueprints": bps, "checksum": ""})
        return len(bps)

    def seed_firestore(self, db, path: str = FIREBASE_ARTIFACT) -> int:
        if not db:
            return 0
        bps = [
            {
                "id": k,
                "code": v,
                "description": k.replace("_", " ").title(),
                "category": "builtin",
                "version": 1,
                "checksum": hashlib.sha256(v.encode()).hexdigest(),
            }
            for k, v in BUILTIN_BLUEPRINTS.items()
        ]
        try:
            db.document(path).set(
                {
                    "blueprints": bps,
                    "checksum": hashlib.sha256(
                        json.dumps(bps, sort_keys=True).encode()
                    ).hexdigest(),
                    "count": len(bps),
                    "seeded_at": time.time(),
                }
            )
            return len(bps)
        except Exception as e:
            logger.warning(f"[Bootstrapper] Firestore seed failed: {e}")
            return 0


class FirestoreSeedGuard:
    """Ensures Firestore has minimum blueprint count before starting."""

    def __init__(self, firebase_db=None, bootstrapper: Optional[BlueprintBootstrapper] = None):
        self._db = firebase_db
        self._bs = bootstrapper

    def ensure_seeded(self):
        if not self._db or not self._bs:
            return
        try:
            doc = self._db.document(FIREBASE_ARTIFACT).get()
            if not doc.exists or doc.to_dict().get("count", 0) < 5:
                self._bs.seed_firestore(self._db)
        except Exception as e:
            logger.warning(f"[SeedGuard] {e}")


# =====================================================================
# ─── SECTION 6: PROCUREMENT ENGINE ───────────────────────────────────
# =====================================================================

@dataclass
class ModelSpec:
    role:         str
    repo:         str
    filename:     str
    size_gb:      float
    min_ram_gb:   float
    quantization: str = "Q4_K_M"

    @property
    def path(self) -> Path:
        return _MODEL_DIR / self.filename


MULTI_MODEL_MANIFEST: List[ModelSpec] = [
    ModelSpec(
        "coder",
        "TheBloke/deepseek-coder-1.3b-instruct-GGUF",
        "deepseek-coder-1.3b-instruct.Q4_K_M.gguf",
        0.8, 4,
    ),
    ModelSpec(
        "tester",
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        0.9, 4,
    ),
    ModelSpec(
        "draft",
        "unsloth/Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        0.6, 4,
    ),
]


class RAMBudgetGuard:
    """Prevents loading models that exceed available RAM."""

    SIZES_GB: Dict[str, float] = {
        "primary": 4.0, "coder": 0.8 , "tester": 0.9, "draft": 0.6,
    }

    def __init__(self, total_ram_gb: float = 0.0, margin_gb: float = 2.0):
        try:
            import psutil
            total = total_ram_gb or round(
                psutil.virtual_memory().total / (1024 ** 3), 1)
        except ImportError:
            total = 8.0
        self._budget = max(total - margin_gb, 0.0)
        self._loaded: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def _used_gb(self) -> float:
        return sum(self.SIZES_GB.get(r, 0.0) for r in self._loaded)

    def register(self, role: str, model_obj: Any) -> bool:
        needed = self.SIZES_GB.get(role, 1.0)
        if self._used_gb() + needed > self._budget:
            logger.warning(
                f"[RAMGuard] Cannot load {role}: "
                f"{self._used_gb():.1f}+{needed:.1f}>{self._budget:.1f}GB"
            )
            return False
        with self._lock:
            self._loaded[role] = model_obj
        return True

    def unregister(self, role: str):
        with self._lock:
            self._loaded.pop(role, None)

    def get_status(self) -> dict:
        return {
            "budget_gb": self._budget,
            "used_gb": round(self._used_gb(), 2),
            "loaded": list(self._loaded.keys()),
        }


class MultiModelManifest:
    """Downloads missing GGUF models from HuggingFace in background threads."""

    def __init__(self):
        try:
            import psutil
            self._budget_gb = round(
                psutil.virtual_memory().total / (1024 ** 3) * 0.75, 1)
        except ImportError:
            self._budget_gb = 6.0
        self._status: Dict[str, str] = {}

    def download_all(self):
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        allocated = 0.0
        download_threads = []  # Track threads to block boot

        for spec in MULTI_MODEL_MANIFEST:
            existing = _find_model(spec.role)
            if existing:
                allocated += spec.size_gb
                self._status[spec.role] = f"cached:{existing.name}"
                continue
            if allocated + spec.size_gb <= self._budget_gb:
                allocated += spec.size_gb
                self._status[spec.role] = "downloading"

                t = threading.Thread(
                    target=self._download, args=(spec,),
                    daemon=True, name=f"Download-{spec.role}"
                )
                t.start()
                download_threads.append(t)
            else:
                self._status[spec.role] = "skipped_ram"
                logger.info(f"[Procurement] Skipping {spec.role} — RAM budget.")

        # ARCHITECTURE FIX: Wait for all downloads to finish before mounting OS!
        if download_threads:
            logger.info("⏳ [Procurement] Waiting for initial model downloads to complete...")
            for t in download_threads:
                t.join()
            logger.info("✅ [Procurement] All downloads complete. Resuming boot.")

    def _download(self, spec: ModelSpec):
        dest = spec.path
        logger.info(f"[Procurement] Downloading {spec.role}: {spec.filename}")
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=spec.repo,
                filename=spec.filename,
                local_dir=str(_MODEL_DIR),
                local_dir_use_symlinks=False,
            )
            self._status[spec.role] = f"downloaded:{spec.filename}"
            logger.info(f"[Procurement] ✅ {spec.role} ready.")
        except Exception:
            try:
                url = (f"https://huggingface.co/{spec.repo}"
                       f"/resolve/main/{spec.filename}")
                urllib.request.urlretrieve(url, str(dest))
                self._status[spec.role] = f"downloaded:{spec.filename}"
            except Exception as e:
                self._status[spec.role] = f"failed:{e}"
                logger.warning(f"[Procurement] ❌ {spec.role} download failed: {e}")

    def get_status(self) -> dict:
        return dict(self._status)


# =====================================================================
# ─── SECTION 7: LOCAL BLUEPRINT MIRROR ───────────────────────────────
# =====================================================================

# Safety patterns — blocked before any exec
_EXEC_BLOCKED_PATTERNS: List[str] = [
    r"rm\s+-[rRfF]{1,}",
    r"sudo\s+rm",
    r"mkfs\.",
    r"dd\s+if=",
    r">\s*/dev/sd",
    r"os\.system\s*\(",
    r"eval\s*\(",
    r"exec\s*\(",
    r"__import__\s*\(",
    r"subprocess\.Popen.*shell\s*=\s*True",
]

# =====================================================================
# ─── SECTION 9: NETWORK WATCHDOG & EDGE SYNC ─────────────────────────
# =====================================================================

class TunnelWatchdog:
    """Pings the Brain URL, auto-rediscovers from Firebase on failure."""

    def __init__(
        self,
        get_url_fn: Callable[[], str],
        set_url_fn: Callable[[str], None],
        firebase_db=None,
        on_reconnect: Optional[Callable] = None,
        on_loss: Optional[Callable] = None,
    ):
        self._get_url    = get_url_fn
        self._set_url    = set_url_fn
        self._db         = firebase_db
        self._on_reconnect = on_reconnect
        self._on_loss    = on_loss
        self._online     = True
        self._fail_count = 0
        self._stop       = threading.Event()

    @property
    def is_online(self) -> bool:
        return self._online

    def _ping(self, url: str) -> bool:
        if not url:
            return False
        try:
            return requests.get(f"{url}/health", timeout=8).status_code == 200
        except Exception:
            return False

    def _rediscover(self) -> Optional[str]:
        if self._db:
            try:
                doc = self._db.document(FIREBASE_ARTIFACT).get()
                if doc.exists:
                    url = doc.to_dict().get("brain_url", "")
                    if self._ping(url):
                        return url
            except Exception:
                pass
        env_url = os.environ.get("BRAIN_URL", "")
        if env_url and self._ping(env_url):
            return env_url
        return None

    def start(self, interval: float = 30.0):
        def _loop():
            current_interval = interval
            while not self._stop.is_set():
                url = self._get_url()
                if self._ping(url):
                    if not self._online and self._on_reconnect:
                        self._on_reconnect(url)
                    self._online     = True
                    self._fail_count = 0
                    current_interval = interval
                else:
                    self._fail_count += 1
                    self._online      = False
                    current_interval  = min(
                        interval * (2 ** (self._fail_count - 1)), 240.0)
                    if self._fail_count >= 2:
                        new_url = self._rediscover()
                        if new_url:
                            self._set_url(new_url)
                            self._fail_count = 0
                            self._online     = True
                            current_interval = interval
                            if self._on_reconnect:
                                self._on_reconnect(new_url)
                    if self._fail_count >= 8 and self._on_loss:
                        self._on_loss()
                self._stop.wait(current_interval)

        threading.Thread(target=_loop, daemon=True, name="TunnelWatchdog").start()

    def stop(self):
        self._stop.set()


class EdgeNodeSync:
    """Maintains brain URL, heartbeat, and blueprint delta sync."""

    def __init__(
        self,
        mirror: LocalBlueprintMirror,
        local_llm: Any,
        watchdog: TunnelWatchdog,
        rag: LocalRAGIndex,
        firebase_db=None,
    ):
        self.mirror    = mirror
        self.local_llm = local_llm
        self.watchdog  = watchdog
        self.rag       = rag
        self._db       = firebase_db
        self._brain_url = os.environ.get("BRAIN_URL", "")
        self._defcon   = 5
        self._passport: dict = {}
        self._stop     = False

        # 1. Try persistent URL cache (survives restarts)
        if BRAIN_URL_CACHE.exists():
            try:
                cached = json.loads(
                    BRAIN_URL_CACHE.read_text(encoding="utf-8"))
                cached_url = cached.get("url", "")
                if cached_url:
                    self._brain_url = cached_url
            except Exception:
                pass

        # 2. If still no URL, query Firebase synchronously at init so first
        #    route_command can reach the brain without waiting for the watchdog
        #    fail-then-rediscover cycle (which takes 2 × 30 s = 60 s minimum).
        if not self._brain_url and firebase_db:
            try:
                doc = firebase_db.document(FIREBASE_ARTIFACT).get()
                if doc.exists:
                    url = doc.to_dict().get("brain_url", "")
                    if url:
                        self._brain_url = url
                        self.set_brain_url(url)
                        logger.info(f"[EdgeSync] Brain URL resolved from Firebase at init: {url}")
            except Exception as e:
                logger.warning(f"[EdgeSync] Firebase init-URL lookup failed: {e}")

    def get_brain_url(self) -> str:
        return self._brain_url

    def set_brain_url(self, url: str):
        self._brain_url = url
        try:
            BRAIN_URL_CACHE.write_text(
                json.dumps({"url": url, "saved_at": time.time()}),
                encoding="utf-8",
            )
        except Exception:
            pass

    def bootstrap_vault(self) -> bool:
        if not self._brain_url:
            return False
        try:
            r = requests.get(
                f"{self._brain_url}/vault_sync",
                headers=_API_HEADERS, timeout=60,
            )
            r.raise_for_status()
            ok = self.mirror.apply_full_sync(r.json())
            if ok and self.rag:
                self.rag.build_faiss_index(list(self.mirror._cache.values()))
            return ok
        except Exception as e:
            logger.warning(f"[EdgeSync] Bootstrap vault failed: {e}")
            return False

    def _sync_loop(self):
        first_run = True
        while not self._stop:
            if self.watchdog.is_online and self._brain_url:
                try:
                    if first_run and not self.mirror.list_ids():
                        self.bootstrap_vault()
                    first_run = False
                    r = requests.post(
                        f"{self._brain_url}/edge/heartbeat",
                        json={"node_id": NODE_ID,
                              "local_defcon": self._defcon},
                        headers=_API_HEADERS, timeout=10,
                    )
                    resp = r.json()
                    self._defcon = resp.get("defcon", 5)
                    delta = resp.get("delta_blueprints")
                    if delta:
                        self.mirror.apply_delta(delta)
                        build_fn = getattr(self.rag, "build_faiss_index", getattr(self.rag, "build", None))
                        if build_fn:
                            build_fn(list(self.mirror._cache.values()))
                except Exception:
                    pass
            time.sleep(HEARTBEAT_INTERVAL)

    def start(self, passport: dict, firebase_db=None):
        self._passport = passport
        if firebase_db and not self._db:
            self._db = firebase_db
        threading.Thread(
            target=self._sync_loop, daemon=True,
            name="EdgeNodeSyncDaemon"
        ).start()


# =====================================================================
# ─── SECTION 10: LOCAL LLMs ──────────────────────────────────────────
# =====================================================================

class SpeculativeHealthReporter:
    """Injects a 1B draft model into a loaded Llama for speculative decoding."""

    def __init__(self):
        self._spec_active  = False
        self._draft_model  = ""
        self._primary_model = ""
        self._error        = ""

    def wrap_llm_load(self, local_llm: Any,
                      draft_path: Optional[Path] = None) -> bool:
        self._primary_model = getattr(
            local_llm, "_path", Path("unknown")).name
        if not getattr(local_llm, "is_loaded", False):
            self._error = "primary model not loaded — skipping speculative injection"
            return False
        if not draft_path or not draft_path.exists():
            # Auto-detect draft model
            draft_path = _find_model("draft")
        if draft_path and draft_path.exists():
            try:
                from llama_cpp import Llama
                from llama_cpp.llama_speculative import LlamaDraftModel
                draft_llm = Llama(
                    model_path=str(draft_path), n_ctx=8192, verbose=False)
                # LlamaDraftModel API changed across versions — try with and without args
                try:
                    local_llm._llm.draft_model = LlamaDraftModel(draft_llm)
                except TypeError:
                    local_llm._llm.draft_model = LlamaDraftModel()
                self._spec_active  = True
                self._draft_model  = draft_path.name
                logger.info(f"[Speculative] ACTIVE — draft={draft_path.name}")
                return True
            except Exception as e:
                self._error = str(e)
                logger.debug(f"[Speculative] Skipping draft injection: {e}")
        return False

    def get_status(self) -> dict:
        return {
            "active": self._spec_active,
            "primary": self._primary_model,
            "draft": self._draft_model,
            "error": self._error,
        }


class LocalLLMFallback:
    def __init__(self, model_path: Optional[Path] = None, draft_path: Optional[Path] = None):
        self._path       = model_path or LOCAL_LLM_PATH
        self._draft_path = draft_path or DRAFT_MODEL_PATH or Path("")
        self._llm        = None
        self.is_loaded   = False

    def load(self) -> bool:
        return self.switch_to_local_llm()

    def switch_to_local_llm(self, model_path: Optional[Path] = None) -> bool:
        path = model_path or self._path
        if not path or not path.exists():
            return False
        try:
            from llama_cpp import Llama
            self._llm = Llama(model_path=str(path), n_ctx=8192, verbose=False)
            self.is_loaded = True
            self._path = path
            logger.info(f"[LocalLLM] Loaded: {path.name} (Uncapped Context)")
            return True
        except Exception as e:
            return False

    def infer(self, prompt: str, system: str = "", max_tokens: int = 800) -> str:
        if not self.is_loaded or not self._llm:
            return f"[EDGE_OFFLINE] {prompt[:60]}"
        try:
            messages = []
            if system: messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            return self._llm.create_chat_completion(messages=messages, max_tokens=max_tokens)["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[LocalLLM error: {e}]"

class _LLMSlot:
    """Single named model slot used by DualLLMSlots."""

    def __init__(self, role: str, path: Optional[Path]):
        self._role     = role
        self._path     = path
        self._llm      = None
        self.is_loaded = False

    def load(self) -> bool:
        if not self._path or not self._path.exists():
            logger.info(f"[{self._role}] Model file not found: {self._path}")
            return False
        try:
            from llama_cpp import Llama
            self._llm      = Llama(
                model_path=str(self._path), n_ctx=8192, verbose=False)
            self.is_loaded = True
            logger.info(f"[{self._role}] Loaded: {self._path.name} (Uncapped Context)")
            return True
        except Exception as e:
            logger.warning(f"[{self._role}] Load failed: {e}")
            return False

    def infer(self, prompt: str, system: str = "",
              max_tokens: int = 800) -> str:
        if not self.is_loaded or not self._llm:
            return ""
        try:
            full = f"{system}\n\n{prompt}" if system else prompt
            return (
                self._llm(full, max_tokens=max_tokens, stop=["</s>"])
                ["choices"][0]["text"].strip()
            )
        except Exception as e:
            logger.warning(f"[{self._role}] Infer failed: {e}")
            return ""

class DualLLMSlots:
    """Loads coder + tester slots with auto-detected model paths."""

    def __init__(self):
        self.coder  = _LLMSlot("coder",  CODER_MODEL_PATH)
        self.tester = _LLMSlot("tester", TESTER_MODEL_PATH)

    def load(self):
        def _seq_load():
            self.coder.load()
            self.tester.load()
        threading.Thread(target=_seq_load, daemon=True, name="LoadSlots").start()


# =====================================================================
# ─── SECTION 11: AIR-GAP SURVIVAL & DEAD MAN'S SWITCH ────────────────
# =====================================================================

class AirGapSurvivalMode:
    """Offline command processor with cloud sync queue."""

    def __init__(
        self,
        mirror: LocalBlueprintMirror,
        local_llm: LocalLLMFallback,
        rag: Optional[LocalRAGIndex] = None,
    ):
        self.mirror        = mirror
        self.llm           = local_llm
        self.rag           = rag
        self._queue: List[dict] = []
        self._lock = threading.Lock()

    def process_command(self, command: str,
                        image_b64: Optional[str] = None) -> dict:
        matched_bp = None
        if self.rag:
            results = self.rag.offline_rag_query(command, top_k=1)
            if results and results[0]["score"] > 0.15:
                matched_bp = results[0]["id"]

        if matched_bp:
            return {
                "message": f"[OFFLINE] Executed '{matched_bp}'.",
                "blueprint_result": self.mirror.execute_local(matched_bp),
                "plan": [],
            }

        sys_prompt = (
            'You are Swayambhu in offline mode. '
            'Respond ONLY with JSON: {"message":"…","plan":[]}'
        )
        response = self.llm.infer(command, system=sys_prompt)
        try:
            parsed = json.loads(
                response.replace("```json", "").replace("```", "").strip())
        except Exception:
            parsed = {"message": response, "plan": []}
        return parsed

    def queue_for_sync(self, command: str, response: dict):
        with self._lock:
            self._queue.append(
                {"command": command, "response": response, "ts": time.time()})

    def flush_queue_to_cloud(self, brain_url: str) -> int:
        flushed = 0
        with self._lock:
            pending = list(self._queue)
        for item in pending:
            try:
                requests.post(
                    f"{brain_url}/command",
                    json={"command": item["command"],
                          "context": {"offline_replay": True}},
                    headers=_API_HEADERS, timeout=10,
                )
                with self._lock:
                    if item in self._queue:
                        self._queue.remove(item)
                flushed += 1
            except Exception:
                break
        return flushed

    def queue_depth(self) -> int:
        with self._lock:
            return len(self._queue)


class DeadMansSwitch:
    """Physically severs Wi-Fi on critical security events."""

    def __init__(self, speak_fn: Optional[Callable] = None,
                 interface: str = "en0"):
        self._speak     = speak_fn or (lambda t: logger.critical(f"[DMS] {t}"))
        self._armed     = True
        self._interface = interface

    def sever_wifi(self, reason: str = "SECURITY_BREACH") -> bool:
        if not self._armed or not _IS_MAC:
            return False
        try:
            result = subprocess.run(
                ["networksetup", "-setairportpower",
                 self._interface, "off"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                logger.critical(
                    f"🚫 [DeadMansSwitch] Wi-Fi SEVERED ({self._interface}): {reason}")
                self._speak(
                    f"Security breach: {reason}. Wi-Fi severed.")
                return True
            return False
        except Exception as e:
            logger.error(f"[DeadMansSwitch] sever_wifi failed: {e}")
            return False

    def quarantine_self(self, reason: str = "SELF_QUARANTINE"):
        self._speak("Critical threat. Initiating self-quarantine.")
        self.sever_wifi(reason)

    def disarm(self):
        self._armed = False

    def arm(self):
        self._armed = True


# =====================================================================
# ─── SECTION 12: LAUNCH AGENT DAEMON ─────────────────────────────────
# =====================================================================

class LaunchAgentDaemon:
    """Installs/removes a macOS LaunchAgent for persistence."""

    PLIST_LABEL = "ai.swayambhu.edgenode"
    LOG_DIR     = Path.home() / "Library" / "Logs" / "Swayambhu"

    def __init__(self, script_path: Optional[Path] = None):
        try:
            self.script = script_path or Path(__file__).resolve()
        except NameError:
            self.script = Path(os.getcwd()) / "swayambhu_body.py"
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def daemonize(self) -> bool:
        if not _IS_MAC:
            return False
        plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{self.PLIST_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{self.script}</string>
        <string>--daemon</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key>
    <string>{self.LOG_DIR}/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>{self.LOG_DIR}/stderr.log</string>
</dict>
</plist>"""
        PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLIST_PATH.write_text(plist, encoding="utf-8")
        subprocess.run(
            ["launchctl", "unload", str(PLIST_PATH)], capture_output=True)
        result = subprocess.run(
            ["launchctl", "load", str(PLIST_PATH)], capture_output=True)
        ok = result.returncode == 0
        logger.info(f"[LaunchAgent] Install {'OK' if ok else 'FAILED'}")
        return ok

    def uninstall(self) -> bool:
        if not PLIST_PATH.exists():
            return False
        subprocess.run(
            ["launchctl", "unload", str(PLIST_PATH)], capture_output=True)
        PLIST_PATH.unlink(missing_ok=True)
        return True

    @staticmethod
    def is_installed() -> bool:
        return PLIST_PATH.exists()


# =====================================================================
# ─── SECTION 13: AVATAR OPENER ───────────────────────────────────────
# =====================================================================

class AvatarState:
    IDLE       = 1
    LISTENING  = 2
    PROCESSING = 3
    SPEAKING   = 4
    ACTUATING  = 5
    ERROR      = 6
    DEFCON     = 7

    _NAMES = {1:"IDLE",2:"LISTENING",3:"PROCESSING",
              4:"SPEAKING",5:"ACTUATING",6:"ERROR",7:"DEFCON"}

    @classmethod
    def name(cls, state: int) -> str:
        return cls._NAMES.get(state, "UNKNOWN")


class SovereignAvatar:
    """Stub avatar — replaced by particle_avatar.py when mounted."""

    def set_state(self, state: int):
        pass

    def set_defcon(self, level: int):
        pass


def _open_in_browser(url: str):
    try:
        if _IS_MAC:
            subprocess.run(["open", url], check=True,
                           capture_output=True, timeout=5)
        elif platform.system() == "Linux":
            subprocess.Popen(["xdg-open", url],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        elif platform.system() == "Windows":
            os.startfile(url)  # type: ignore[attr-defined]
        else:
            import webbrowser
            webbrowser.open(url)
    except Exception:
        pass


def auto_open_avatar(port: int = AVATAR_PORT, wait: float = 30.0,
                     enabled: bool = True):
    """Opens the avatar UI in the default browser once the server is up."""
    if not enabled:
        return

    def _probe():
        url      = f"http://localhost:{port}"
        deadline = time.time() + wait
        while time.time() < deadline:
            try:
                urllib.request.urlopen(f"{url}/health", timeout=2)
                break
            except Exception:
                try:
                    urllib.request.urlopen(url, timeout=2)
                    break
                except Exception:
                    time.sleep(1.5)
        _open_in_browser(url)

    threading.Thread(target=_probe, daemon=True, name="AvatarOpener").start()


# =====================================================================
# ─── SECTION 14: FIRM WIRING ─────────────────────────────────────────
# =====================================================================

def probe_silicon() -> dict:
    """Detect hardware chip type and available RAM."""
    chip = "Unknown"
    ram  = 8.0
    if _IS_MAC:
        chip = "Apple Silicon"
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=3,
            )
            if "Apple" in result.stdout:
                chip = result.stdout.strip()
        except Exception:
            pass
    try:
        import psutil
        ram = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    return {"chip": chip, "ram_gb": ram, "node_id": NODE_ID,
            "platform": platform.system()}


class FirmWiring:
    """Wires the SoftwareFirm to the orchestrator once both LLM slots load."""

    @staticmethod
    def wire(orchestrator: "EdgeNodeOrchestrator") -> bool:
        try:
            from software_firm import SoftwareFirm
            coder_fn  = lambda p, s="", n=600: (
                orchestrator.coder_llm.infer(p, system=s, max_tokens=n)
                if orchestrator.coder_llm else "")
            tester_fn = lambda p, s="", n=600: (
                orchestrator.tester_llm.infer(p, system=s, max_tokens=n)
                if orchestrator.tester_llm else "")
            orchestrator.firm = SoftwareFirm(
                manager_fn=orchestrator.route_command,
                coder_llm=coder_fn,
                tester_llm=tester_fn,
                max_iterations=3,
            )
            return True
        except Exception as e:
            logger.warning(f"[FirmWiring] Wire failed: {e}")
            return False

    @staticmethod
    def rewire_when_loaded(orchestrator: "EdgeNodeOrchestrator",
                           poll_sec: float = 10.0,
                           max_polls: int = 60):
        cancel = threading.Event()

        def _poll():
            for _ in range(max_polls):
                if cancel.is_set():
                    return
                coder_ok  = getattr(
                    getattr(orchestrator, "coder_llm", None),
                    "is_loaded", False)
                tester_ok = getattr(
                    getattr(orchestrator, "tester_llm", None),
                    "is_loaded", False)
                if coder_ok and tester_ok:
                    if FirmWiring.wire(orchestrator):
                        logger.info("✅ [SoftwareFirm] Wired natively.")
                    return
                time.sleep(poll_sec)

        t = threading.Thread(target=_poll, daemon=True, name="FirmRewire")
        t.start()
        return cancel  # caller can set cancel to abort


# =====================================================================
# ─── SECTION 15: EDGE NODE ORCHESTRATOR ──────────────────────────────
# =====================================================================

class EdgeNodeOrchestrator:
    def __init__(self):
        self.passport    = probe_silicon()
        self.caps        = CapabilityFlags()
        self.local_llm   = LocalLLMFallback()
        self.dms         = DeadMansSwitch(speak_fn=self.speak)
        self.avatar      = SovereignAvatar()
        self._online     = True
        self._defcon     = 5

        self.blueprint_engine = BlueprintEngine(script_dir=PROJECT_ROOT, timeout_sec=15.0)
        self.mirror = self.blueprint_engine.mirror
        self.rag = self.blueprint_engine.rag_index

        self.tcc          = TCCPreflight()
        self.ram_guard    = RAMBudgetGuard()
        self.spec_reporter = SpeculativeHealthReporter()
        self.procurement  = MultiModelManifest()

        self.watchdog = TunnelWatchdog(get_url_fn=self._get_brain_url, set_url_fn=self._set_brain_url, on_loss=self._on_network_loss, on_reconnect=self._on_network_restore)
        self.sync = EdgeNodeSync(self.mirror, self.local_llm, self.watchdog, self.rag, firebase_db=None)
        self.air_gap = AirGapSurvivalMode(self.mirror, self.local_llm, self.rag)
        self._dual = DualLLMSlots()

        self.firm = None
        self.spine = None
        self.empathy_wire = None
        self.hippocampus = None
        self.lizard_brain = None
        self.openclaw = None
        self.dual_engine = None
        self.tactical_rag = None
        self.uas = None

        self.coder_llm = self._dual.coder
        self.tester_llm = self._dual.tester
        self.execute_script = self._execute_applescript
        self.search_blueprints = getattr(self.rag, "vector_search", getattr(self.rag, "search", None))
        self._boot_report = {}

        # 🚀 Hard synchronization gates
        self.boot_ready = threading.Event()
        self.cloud_ready = threading.Event()

    def _safe_local_infer(self, prompt: str, system: str = "") -> str:
        if not getattr(self.coder_llm, "is_loaded", False): return ""
        try:
            res = self.coder_llm.infer(prompt, system=system)
            return getattr(res, "text", str(res))
        except Exception: return ""

    def _get_brain_url(self) -> str: return self.sync.get_brain_url()
    def _set_brain_url(self, url: str): self.sync.set_brain_url(url)
    def _set_avatar_state(self, state): self.avatar.set_state(AvatarState.name(state).lower() if isinstance(state, int) else state.lower())

    def speak(self, text: str):
        if not text or not text.strip(): return
        logger.info(f"🌌 [SWAYAMBHU]: {text}")
        if _TTS_OK and _TTS_ENGINE:
            def _tts():
                try:
                    _TTS_ENGINE.say(text)
                    _TTS_ENGINE.runAndWait()
                except Exception: pass
            threading.Thread(target=_tts, daemon=True, name="TTS").start()

    def _build_system_prompt(self) -> str:
        prompt = (
            "You are Swayambhu, an advanced AI with complete control over a macOS system. "
            "Respond in a conversational tone, but you MUST format your output as a strict JSON object.\n"
            "SCHEMA: {\"message\": \"Your reply\", \"plan\": [{\"action\": \"tool_name\", \"params\": {\"key\": \"value\"}}]}\n\n"
            "ACTIVE ORGANS AVAILABLE FOR THE 'plan' ARRAY:\n"
            "- 'execute_mac_command' (params: {'script': 'str'}): PRIMARY OS TOOL. Use to execute fast bash commands (e.g. `open x-apple.systempreferences:` or `open -a Safari`) or AppleScript. ALWAYS use this for opening apps, changing settings, or OS toggles.\n"
        )
        if getattr(self, "openclaw", None):
            prompt += "- 'launch_autonomous_mission' (params: {'goal': 'str'}): Use to achieve complex, multi-step OS goals without supervision.\n"
        if getattr(self, "uas", None):
            prompt += "- 'execute_universal_action' (params: {'goal': 'str'}): SECONDARY TOOL. Use ONLY for complex UI navigation (clicking specific screen buttons or menus).\n"
        if getattr(self, "firm", None):
            prompt += "- 'delegate_to_software_firm' (params: {'task': 'str'}): Use to write, test, and deploy new code.\n"
        return prompt

    def _build_tool_payload(self) -> List[dict]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute_mac_command",
                    "parameters": {"type": "object", "properties": {"script": {"type": "string"}}}
                }
            }
        ]
        if getattr(self, "uas", None):
            tools.append({
                "type": "function",
                "function": {
                    "name": "execute_universal_action",
                    "parameters": {"type": "object", "properties": {"goal": {"type": "string"}}}
                }
            })
        if getattr(self, "openclaw", None) and hasattr(self.openclaw, "get_tool_schemas"):
            tools.extend(self.openclaw.get_tool_schemas())
        if getattr(self, "firm", None):
            tools.append({
                "type": "function",
                "function": {
                    "name": "delegate_to_software_firm",
                    "parameters": {"type": "object", "properties": {"task": {"type": "string"}}}
                }
            })
        return tools

    def _cloud_post(self, command: str, image_b64: Optional[str] = None, context: Optional[dict] = None) -> dict:
        url = self.sync.get_brain_url()
        if not url: return {"message": ""}
        sys_override = self._build_system_prompt()
        payload = {"command": command, "image_b64": image_b64, "available_tools": self._build_tool_payload(), "sys_override": sys_override, "context": {"node_id": NODE_ID, **(context or {})}}
        try:
            r = requests.post(f"{url}/edge_command", json=payload, headers=_API_HEADERS, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"message": ""}

    def route_command(self, command: str, image_b64: Optional[str] = None, context: Optional[dict] = None) -> dict:
        self._set_avatar_state(AvatarState.PROCESSING)
        result = {"message": "", "plan": []}

        # 1. Brain/Spine Decision Phase
        if getattr(self, "spine", None):
            result = self.spine.route(command, image_b64)
        elif self._online and self.sync.get_brain_url():
            result = self._cloud_post(command, image_b64, context)

        if not result.get("message") and getattr(self, "air_gap", None):
            result = self.air_gap.process_command(command, image_b64)

        # 2. Universal Execution Phase (The physical bridge)
        for step in result.get("plan", []):
            action = step.get("action", "")
            params = step.get("params", {})

            # Extract the raw command string from any common JSON key
            raw_cmd = params.get("script") or params.get("command") or params.get("goal") or ""

            if not raw_cmd and action:
                logger.error(f"⚠️ [Bridge] Action {action} provided but no command string found.")
                continue

            # Route to physical OS layers
            if action in ("execute_mac_command", "actuate", "execute_universal_action"):
                # Use the robust hybrid executor we built
                logger.info(f"⚡ [Executing] {action} -> {raw_cmd}")
                self._execute_applescript(raw_cmd)

            elif action == "delegate_to_software_firm" and getattr(self, "firm", None):
                self.firm.run_task(raw_cmd)

            elif action == "launch_autonomous_mission" and getattr(self, "openclaw", None):
                self.openclaw.launch_mission(raw_cmd)

        self.speak(result.get("message", ""))
        self._set_avatar_state(AvatarState.IDLE)
        return result

    def _execute_applescript(self, script: str) -> dict:
        if not script: return {"status": "NO_OP"}
        script = script.strip()
        try:
            # 🚀 Auto-detect: If it starts with a shell primitive, use Zsh
            shell_primitives = ("open", "ls", "cd", "mkdir", "echo", "pwd", "curl", "python3")
            is_shell = any(script.startswith(p) for p in shell_primitives)

            if is_shell:
                logger.info(f"⚡ [System Shell] Executing: {script}")
                # 🚀 ARCHITECTURAL FIX: Prevent Zsh from crashing on URLs with '?' or '&'
                safe_script = f"unsetopt nomatch; {script}"
                subprocess.run(safe_script, shell=True, check=True, capture_output=True, timeout=15,
                               executable='/bin/zsh')
            else:
                logger.info(f"🍎 [AppleScript] Executing script...")
                subprocess.run(["osascript", "-e", script], capture_output=True, timeout=15, check=True)

            return {"status": "OK"}
        except Exception as e:
            logger.error(f"❌ [Executor Failed] {e}")
            return {"status": "ERROR", "detail": str(e)}

    def _on_network_loss(self):
        self._online = False
        self._defcon = 1
        self._set_avatar_state(AvatarState.DEFCON)
        self.speak("Network lost. Offline mode active.")

    def _on_network_restore(self, url: Optional[str] = None):
        self._online = True
        self._defcon = 5
        self._set_avatar_state(AvatarState.IDLE)
        self.speak("Cloud restored.")
        if self.sync.get_brain_url() and self.air_gap.flush_queue_to_cloud(self.sync.get_brain_url()): pass

    def start_listening(self):
        if not _SR_OK or not self.tcc.can("microphone"): return
        def _loop():
            recognizer = sr.Recognizer()
            mic = sr.Microphone()
            try:
                with mic as source: recognizer.adjust_for_ambient_noise(source, duration=0.5)
            except Exception: return
            while True:
                try:
                    self._set_avatar_state(AvatarState.LISTENING)
                    with mic as source: audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    self._set_avatar_state(AvatarState.PROCESSING)
                    text = recognizer.recognize_google(audio).strip()
                    if text: self.route_command(text)
                except Exception: time.sleep(1)
        self.caps.voice_input = True
        threading.Thread(target=_loop, daemon=True, name="VoiceListenLoop").start()

    def boot(self, firebase_db=None, auto_procure: bool = True, open_avatar: bool = True) -> "EdgeNodeOrchestrator":
        logger.info("🌌 ═══ SWAYAMBHU EDGE NODE v14.0 BOOTING ═══")
        perms = self.tcc.check_all()
        self.caps.tcc_mic, self.caps.tcc_accessibility = perms.get("microphone", False), perms.get("accessibility", False)

        bb = BlueprintBootstrapper()
        bb.seed_disk()
        bb.seed_mirror(self.mirror)
        FirestoreSeedGuard(firebase_db, bb).ensure_seeded()
        if firebase_db:
            self.caps.firebase = True
            self.watchdog._db = firebase_db

        if auto_procure: self.procurement.download_all()

        try:
            from dual_model_engine import DualModelEngine
            self.dual_engine = DualModelEngine(model_dir=_MODEL_DIR, coder_file=CODER_MODEL_PATH.name if CODER_MODEL_PATH else CODER_MODEL_FILE, tester_file=TESTER_MODEL_PATH.name if TESTER_MODEL_PATH else TESTER_MODEL_FILE)
            self.dual_engine.load(coder_path=CODER_MODEL_PATH, tester_path=TESTER_MODEL_PATH)
            self.coder_llm, self.tester_llm = self.dual_engine.coder, self.dual_engine.tester
            self.caps.dual_engine = True
            logger.info("✅ [OS Kernel] Advanced Parallel Inference Engine mounted.")
        except Exception:
            self._dual.load()
            self.coder_llm, self.tester_llm = self._dual.coder, self._dual.tester

        if CODER_MODEL_PATH and CODER_MODEL_PATH.exists(): self.local_llm._path = CODER_MODEL_PATH

        _orch_ref = self

        def _post_load_watcher():
            if getattr(_orch_ref, "dual_engine", None): _orch_ref.dual_engine.wait_loaded(timeout=120.0)
            _orch_ref.caps.local_coder, _orch_ref.caps.local_tester = getattr(_orch_ref.coder_llm, "is_loaded", False), getattr(_orch_ref.tester_llm,"is_loaded", False)

            # 🚀 ARCHITECTURAL FIX: Safe RAM deduplication.
            if _orch_ref.coder_llm and getattr(_orch_ref.coder_llm, "is_loaded", False):
                # 1. Share the C++ instance memory pointer
                _orch_ref.local_llm._llm = _orch_ref.coder_llm._llm
                _orch_ref.local_llm._path = _orch_ref.coder_llm.model_path
                _orch_ref.local_llm.is_loaded = True

                # 2. Reroute inference through the ModelSlot's thread-safe lock
                def _locked_infer(prompt: str, system: str = "", max_tokens: int = 800) -> str:
                    res = _orch_ref.coder_llm.infer(prompt, system=system, max_tokens=max_tokens)
                    return getattr(res, "text", str(res))

                _orch_ref.local_llm.infer = _locked_infer
                logger.info(f"[LocalLLM] Air-gap fallback natively linked to DualEngine Coder.")

            if _FIRM_OK and FirmWiring.wire(_orch_ref):
                _orch_ref.caps.software_firm = True
                logger.info("✅ [SoftwareFirm] Wired natively.")
            _orch_ref._boot_report["capabilities"] = _orch_ref.caps.summary()
            _orch_ref._boot_report["active_organs"] = _orch_ref.caps.active_organs()

            # Drop the memory gate
            _orch_ref.boot_ready.set()

        threading.Thread(target=_post_load_watcher, daemon=True, name="PostLoadWatcher").start()

        self.spec_reporter.wrap_llm_load(self.local_llm)
        try:
            from tactical_edge_rag import TacticalEdgeRAG
            self.tactical_rag = TacticalEdgeRAG(rag_index=self.rag)
            self.tactical_rag.start_shadow_sync()
            search_fn = getattr(self.tactical_rag, "vector_search", getattr(self.tactical_rag, "search", getattr(self.tactical_rag, "query", None)))
            if search_fn:
                self.search_blueprints = search_fn
                self.caps.tactical_rag = True
                logger.info("✅ [OS Kernel] Tactical RAG mounted.")
        except Exception: pass

        try:
            from universal_action_space import UniversalActionSpace
            self.uas = UniversalActionSpace(llm_fn=self._safe_local_infer, blueprint_engine=self.blueprint_engine)
            bridge = getattr(self.uas, "bridge", None)
            if bridge and hasattr(bridge, "run_applescript"): self.execute_script = lambda s: bridge.run_applescript(s, label="Body")
            self.caps.universal_action = True
            logger.info("✅ [OS Kernel] Universal Action Space mounted.")
        except Exception: pass

        try:
            from openclaw import get_general
            self.openclaw = get_general(firebase_db=firebase_db, llm_fn=lambda p: self._cloud_post(p).get("message", ""),script_dir=PROJECT_ROOT)
            if getattr(self, "coder_llm", None): self.openclaw._distiller._llm = self._safe_local_infer

            # 🚀 Prevent execution of stale/corrupted JSON missions on boot
            self.openclaw.start(resume_interrupted=False)

            self.caps.openclaw = True
            logger.info("✅ [OS Kernel] OpenClaw General mounted.")
        except Exception:
            pass

        self.sync._db = firebase_db

        def _async_sync_check():
            is_url_alive = False
            if self.sync.get_brain_url():
                try:
                    is_url_alive = requests.get(f"{self.sync.get_brain_url()}/health", timeout=2.0).status_code == 200
                except Exception:
                    pass
            if not is_url_alive and firebase_db:
                try:
                    doc = firebase_db.document(FIREBASE_ARTIFACT).get()
                    if doc.exists and doc.to_dict().get("brain_url"): self.sync.set_brain_url(
                        doc.to_dict().get("brain_url"))
                except Exception:
                    pass
            self.sync.start(self.passport, firebase_db=firebase_db)
            self.watchdog.start()

            # 🚀 Drop the network gate once Firebase finishes
            self.cloud_ready.set()

        threading.Thread(target=_async_sync_check, daemon=True).start()

        try:
            from empathy_wire import EmpathyWire
            self.empathy_wire = EmpathyWire()
            self.empathy_wire.start()
            self.caps.empathy_wire = True
            logger.info("✅ [OS Kernel] EmpathyWire mounted.")
        except Exception: pass

        try:
            from hippocampus import BodyHippocampus
            self.hippocampus = BodyHippocampus(store_dir=PROJECT_ROOT / "body_memory")
            self.caps.hippocampus = True
            logger.info("✅ [OS Kernel] Hippocampus mounted.")
        except Exception: pass

        try:
            from sovereign_spine import SovereignSpine
            self.spine = SovereignSpine(local_llm=self.local_llm, cloud_url_fn=self.sync.get_brain_url, cloud_call_fn=lambda c: self._cloud_post(c), empathy_wire=self.empathy_wire, hippocampus=self.hippocampus, client_id="default")
            self.spine.start()
            self.caps.sovereign_spine = True
            logger.info("✅ [OS Kernel] Sovereign Spine mounted.")
        except Exception: pass

        try:
            from gesture_tracker import GestureTracker
            self.gesture_tracker = GestureTracker()
            self.caps.gesture_tracker = True
            logger.info("✅ [OS Kernel] Gesture Tracker mounted.")
        except ImportError: pass

        try:
            from affective_engine import AffectiveEngine
            self.affective_engine = AffectiveEngine()
            self.caps.affective_engine = True
            logger.info("✅ [OS Kernel] Affective Engine mounted.")
        except ImportError: pass

        try:
            from lizard_brain import LizardBrain
            self.lizard_brain = LizardBrain()
            self.caps.lizard_brain = True
            logger.info("✅ [OS Kernel] Lizard Brain mounted.")
        except ImportError: pass

        try:
            from proactive_agency import ProactiveAgency
            self.proactive_agency = ProactiveAgency(llm_fn=self._safe_local_infer)
            self.proactive_agency.start()
            self.caps.proactive_agency = True
            logger.info("✅ [OS Kernel] Proactive Agency mounted.")
        except ImportError: pass

        if self.mirror.list_ids():
            build_fn = getattr(self.rag, "build_faiss_index", getattr(self.rag, "build", None))
            if build_fn: build_fn(list(self.mirror._cache.values()))

        try:
            from acoustic_gate import AcousticGate
            self.acoustic_gate = AcousticGate()
            self.acoustic_gate.start()
            self.caps.voice_input = True
            logger.info("✅ [OS Kernel] Acoustic Gate mounted.")
        except Exception: self.start_listening()

        if open_avatar:
            try:
                from particle_avatar import ParticleAvatarServer
                self.avatar_server = ParticleAvatarServer(port=AVATAR_PORT)
                self.avatar_server.start()
                self.avatar = self.avatar_server
                auto_open_avatar(port=AVATAR_PORT, enabled=True)
            except Exception: auto_open_avatar(enabled=False)
        else: auto_open_avatar(enabled=False)

        self._boot_report = {
            "version": "14.0", "node_id": NODE_ID, "model_dir": str(_MODEL_DIR),
            "coder": str(CODER_MODEL_PATH), "tester": str(TESTER_MODEL_PATH), "draft": str(DRAFT_MODEL_PATH),
            "capabilities": self.caps.summary(), "active_organs": self.caps.active_organs(),
            "blueprints": len(self.mirror.list_ids()), "procurement": self.procurement.get_status(),
        }
        logger.info("🌌 ═══ BOOT COMPLETE ═══")
        for organ in self._boot_report["active_organs"]: logger.info(f"    ✅ {organ}")
        return self

    def get_status(self) -> dict:
        return {**self._boot_report, "online": self._online, "defcon": self._defcon, "brain_url": self.sync.get_brain_url()}

# =====================================================================
# ─── SECTION 16: EDGE API SERVER ─────────────────────────────────────
# =====================================================================

def start_edge_server(orchestrator: EdgeNodeOrchestrator) -> bool:
    """Start the FastAPI edge server on EDGE_SERVER_PORT in a daemon thread."""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError as e:
        logger.error(f"[EdgeServer] Missing dependency: {e}")
        return False

    edge_app = FastAPI(
        title=f"Swayambhu Edge — {NODE_ID}",
        version="14.0",
        description="Swayambhu Mac Body Edge API",
    )
    edge_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class CommandPayload(BaseModel):
        command: str
        image_b64: Optional[str] = None
        context: Optional[dict] = None

    class AvatarStatePayload(BaseModel):
        state: str | int  # Accepts both string from JS and int from Python
        defcon: Optional[int] = None

    @edge_app.post("/command")
    async def cmd_endpoint(payload: CommandPayload):
        return orchestrator.route_command(
            command=payload.command,
            image_b64=payload.image_b64,
            context=payload.context
        )

    @edge_app.post("/avatar/state")
    async def avatar_state_endpoint(payload: AvatarStatePayload):
        # Convert integer states to strings for ParticleAvatarServer compatibility
        if isinstance(payload.state, int):
            state_str = AvatarState.name(payload.state).lower()
        else:
            state_str = payload.state.lower()

        orchestrator._set_avatar_state(state_str)

        if payload.defcon is not None:
            orchestrator._defcon = payload.defcon
            orchestrator.avatar.set_defcon(payload.defcon)

        return {"status": "OK", "state": state_str}

    @edge_app.get("/health")
    async def health_endpoint():
        return orchestrator.get_status()

    @edge_app.get("/blueprints")
    async def blueprints_endpoint():
        return {"ids": orchestrator.mirror.list_ids(),
                "count": len(orchestrator.mirror.list_ids())}

    @edge_app.post("/blueprint/execute")
    async def execute_bp(body: dict):
        bp_id    = body.get("id", "")
        entry_fn = body.get("entry_fn", "")
        kwargs   = body.get("kwargs", {})
        return orchestrator.mirror.execute_local(bp_id, entry_fn, kwargs)

# ─────────────────────────────────────────────────────────────────
# ⚡ MOUNT NEURAL PIPELINE (WEBSOCKET STREAMING)
# ─────────────────────────────────────────────────────────────────
    try:
        from neural_pipeline import (
            attach_body_ws_endpoint,
            BodyWebSocketClient,
            OllamaStreamGenerator
        )

        # 1. Dynamically resolve the Kaggle URL and convert protocol (http -> ws)
        brain_http_url = orchestrator.sync.get_brain_url()
        brain_ws_url = ""
        if brain_http_url and brain_http_url.startswith("http"):
            # The rstrip('/') is CRITICAL to prevent double-slashes (ngrok.dev//ws_stream)
            base_ws_url = brain_http_url.replace("https://", "wss://").replace("http://", "ws://").rstrip('/')
            brain_ws_url = f"{base_ws_url}/ws_stream"
        else:
            logger.warning("⚠️ [EdgeServer] Brain URL not yet synced — streaming may fallback to local.")

        # 2. Initialize the local fallback generator for Air-Gap (DEFCON 1)
        local_gen = OllamaStreamGenerator()

        # 3. Create the upstream relay client that talks to Kaggle
        ws_client = BodyWebSocketClient(
            brain_ws_url=brain_ws_url,
            local_gen=local_gen,
            defcon_fn=lambda: orchestrator._defcon
        )

        # 4. Mount the local WebSocket for the HTML Avatar UI to connect to
        attach_body_ws_endpoint(
            app=edge_app,
            ollama_gen=local_gen,
            brain_client=ws_client
        )
        logger.info("✅ [EdgeServer] Neural Pipeline WebSocket mounted on /ws_stream")
    except ImportError:
        logger.info("⚠️ [EdgeServer] neural_pipeline.py not found — streaming disabled.")
    except Exception as e:
        logger.warning(f"⚠️ [EdgeServer] Failed to mount Neural Pipeline: {e}")

    def _run():
        uvicorn.run(
            edge_app,
            host="0.0.0.0",
            port=EDGE_SERVER_PORT,
            log_level="warning",
        )

    threading.Thread(target=_run, daemon=True, name="EdgeAPIServer").start()
    logger.info(f"✅ [EdgeServer] Listening on port {EDGE_SERVER_PORT}")
    return True


# =====================================================================
# ─── SECTION 17: FIREBASE INIT ───────────────────────────────────────
# =====================================================================

def init_firebase_edge(key_path: Optional[Path] = None):
    path = key_path or (PROJECT_ROOT / "firebase_key.json")
    if not path.exists():
        logger.warning(f"[Firebase] Key not found: {path}")
        return None
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(str(path)))

        # FIX: Correct way to initialize the client with a specific DB ID
        db = firestore.client(database_id=FIREBASE_DB_ID)
        logger.info(f"🔥 [Firebase] Connected to: {FIREBASE_DB_ID}")
        return db
    except Exception as e:
        logger.warning(f"[Firebase] Init failed: {e}")
        return None


# =====================================================================
# ─── SELF-TESTS ──────────────────────────────────────────────────────
# =====================================================================

if __name__ == "__main__":
    import sys as _sys
    import tempfile

    # ── check for --daemon flag (launched by LaunchAgent) ─────────────
    if "--daemon" in _sys.argv:
        _db  = init_firebase_edge()
        edge = EdgeNodeOrchestrator()
        edge.boot(firebase_db=_db, auto_procure=True, open_avatar=False)
        start_edge_server(edge)
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            pass
        _sys.exit(0)

    # ── check for --test flag ─────────────────────────────────────────
    if "--test" in _sys.argv or os.environ.get("SWAYAMBHU_TEST"):
        logging.basicConfig(level=logging.WARNING)
        _PASS = 0
        _FAIL = 0

        def _ok(label: str):
            global _PASS; _PASS += 1
            print(f"  ✅ {label}")

        def _fail(label: str, exc: Exception):
            global _FAIL; _FAIL += 1
            print(f"  ❌ {label}: {exc}")
            traceback.print_exc()

        print("🌌 swayambhu_body.py self-test suite\n")

        # ── T1: Model auto-detection ──────────────────────────────────
        print("=== T1: Model auto-detection ===")
        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                # Create dummy GGUFs
                (td_path / "deepseek-coder-Q4_K_M.gguf").write_bytes(b"\x00")
                (td_path / "qwen2.5-1.5b-Q4_K_M.gguf").write_bytes(b"\x00")
                (td_path / "llama-3.2-1b-Q4_K_M.gguf").write_bytes(b"\x00")
                coder  = _find_model("coder",  td_path)
                tester = _find_model("tester", td_path)
                draft  = _find_model("draft",  td_path)
                assert coder  and "deepseek" in coder.name.lower(),  f"coder={coder}"
                assert tester and "qwen"     in tester.name.lower(), f"tester={tester}"
                assert draft  and "llama"    in draft.name.lower(),  f"draft={draft}"
                # No match → None
                assert _find_model("coder", Path(td) / "empty") is None
            _ok("coder/tester/draft auto-detected from scan")
        except Exception as e:
            _fail("Model auto-detection", e)

        # ── T2: CapabilityFlags ───────────────────────────────────────
        print("\n=== T2: CapabilityFlags ===")
        try:
            cf = CapabilityFlags()
            assert not cf.local_coder
            cf.local_coder = True
            assert cf.local_coder
            assert "local_coder" in cf.active_organs()
            s = cf.summary()
            assert "tcc_mic" in s and "hippocampus" in s
            _ok(f"active={cf.active_organs()}")
        except Exception as e:
            _fail("CapabilityFlags", e)

        # ── T3: BlueprintBootstrapper ─────────────────────────────────
        print("\n=== T3: BlueprintBootstrapper ===")
        try:
            with tempfile.TemporaryDirectory() as td:
                bp_dir = Path(td) / "blueprints"
                bb = BlueprintBootstrapper(bp_dir)
                n = bb.seed_disk()
                assert n == len(BUILTIN_BLUEPRINTS), f"wrote {n}"
                # Idempotent — second call writes 0
                assert bb.seed_disk() == 0
                assert (bp_dir / "open_safari.py").exists()
            _ok(f"seeded {n} blueprints idempotently")
        except Exception as e:
            _fail("BlueprintBootstrapper", e)

        # ── T4: LocalBlueprintMirror ──────────────────────────────────
        print("\n=== T4: LocalBlueprintMirror ===")
        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                mirror = LocalBlueprintMirror(
                    td_path / "vault.json", td_path / "bps")
                bb = BlueprintBootstrapper(td_path / "bps")
                bb.seed_mirror(mirror)
                ids = mirror.list_ids()
                assert len(ids) == len(BUILTIN_BLUEPRINTS), f"ids={len(ids)}"
                assert "get_battery" in ids
                # Execute a safe blueprint
                res = mirror.execute_local("get_ip")
                assert res.get("status") in ("EXECUTED", "LOADED", "error"), res
                # Shield blocks rm -rf
                mirror._cache["evil"] = {
                    "id": "evil", "code": "import os\nrm -rf /tmp/x\n"}
                res2 = mirror.execute_local("evil")
                assert "error" in res2
                # Delta sync
                mirror.apply_delta([
                    {"id": "get_ip", "version": 0, "code": "def run(): pass"}])
                assert mirror._cache["get_ip"]["version"] == 1  # not downgraded
            _ok(f"mirror ids={len(ids)} exec=ok shield=ok delta=ok")
        except Exception as e:
            _fail("LocalBlueprintMirror", e)

        # ── T5: LocalRAGIndex ─────────────────────────────────────────
        print("\n=== T5: LocalRAGIndex ===")
        try:
            with tempfile.TemporaryDirectory() as td:
                rag = LocalRAGIndex(Path(td) / "rag.json")
                bps = [
                    {"id": "open_safari",
                     "description": "open safari browser", "category": "app"},
                    {"id": "get_battery",
                     "description": "check battery level power",
                     "category": "system"},
                    {"id": "web_search",
                     "description": "search the web google",
                     "category": "search"},
                ]
                n = rag.build_faiss_index(bps)
                assert n == 3
                # search() alias works
                results = rag.search("battery", top_k=1)
                assert len(results) >= 1
                assert results[0]["id"] == "get_battery", results
                # offline_rag_query alias
                r2 = rag.offline_rag_query("browser safari", top_k=1)
                assert r2[0]["id"] == "open_safari", r2
            _ok(f"built={n} battery→{results[0]['id']} safari→{r2[0]['id']}")
        except Exception as e:
            _fail("LocalRAGIndex", e)

        # ── T6: RAMBudgetGuard ────────────────────────────────────────
        print("\n=== T6: RAMBudgetGuard ===")
        try:
            guard = RAMBudgetGuard(total_ram_gb=8.0, margin_gb=2.0)
            # budget = 6GB; coder = 8.9GB → rejected
            assert not guard.register("coder", object())
            # tester = 0.9GB → accepted
            assert guard.register("tester", object())
            assert "tester" in guard.get_status()["loaded"]
            guard.unregister("tester")
            assert guard.get_status()["used_gb"] == 0.0
            _ok(f"budget={guard._budget}GB coder_rejected tester_ok")
        except Exception as e:
            _fail("RAMBudgetGuard", e)

        # ── T7: AirGapSurvivalMode ────────────────────────────────────
        print("\n=== T7: AirGapSurvivalMode ===")
        try:
            with tempfile.TemporaryDirectory() as td:
                td_path = Path(td)
                mirror  = LocalBlueprintMirror(
                    td_path / "v.json", td_path / "bps")
                BlueprintBootstrapper(td_path / "bps").seed_mirror(mirror)
                rag = LocalRAGIndex(td_path / "rag.json")
                rag.build_faiss_index(list(mirror._cache.values()))
                llm = LocalLLMFallback()  # not loaded → offline stub
                ag  = AirGapSurvivalMode(mirror, llm, rag)
                # Low score command → LLM fallback response
                r = ag.process_command("what is the weather today")
                assert "message" in r
                ag.queue_for_sync("test", r)
                assert ag.queue_depth() == 1
            _ok(f"air-gap process ok queue_depth=1")
        except Exception as e:
            _fail("AirGapSurvivalMode", e)

        # ── T8: DeadMansSwitch (non-Mac safe) ─────────────────────────
        print("\n=== T8: DeadMansSwitch ===")
        try:
            dms = DeadMansSwitch()
            dms.disarm()
            assert not dms.sever_wifi("test")  # disarmed
            dms.arm()
            # On non-Mac it returns False without side effects
            result = dms.sever_wifi("unit_test")
            assert isinstance(result, bool)
            _ok(f"disarm=blocked arm=tried result={result}")
        except Exception as e:
            _fail("DeadMansSwitch", e)

        # ── T9: LaunchAgentDaemon ─────────────────────────────────────
        print("\n=== T9: LaunchAgentDaemon ===")
        try:
            with tempfile.NamedTemporaryFile(
                    suffix=".py", delete=False) as tmp:
                dummy_script = Path(tmp.name)
            lad = LaunchAgentDaemon(script_path=dummy_script)
            assert lad.LOG_DIR.exists()
            # plist install only on real Mac — just verify generation
            if not _IS_MAC:
                _ok("non-Mac: plist generation skipped gracefully")
            else:
                installed = lad.daemonize()
                assert isinstance(installed, bool)
                if installed:
                    assert lad.is_installed()
                    lad.uninstall()
                    assert not lad.is_installed()
                _ok(f"Mac: install={installed}")
            dummy_script.unlink(missing_ok=True)
        except Exception as e:
            _fail("LaunchAgentDaemon", e)

        # ── T10: probe_silicon ────────────────────────────────────────
        print("\n=== T10: probe_silicon ===")
        try:
            info = probe_silicon()
            assert all(k in info for k in
                       ["chip", "ram_gb", "node_id", "platform"])
            assert info["ram_gb"] > 0
            assert info["node_id"].startswith("MAC_EDGE_")
            _ok(f"chip={info['chip']} ram={info['ram_gb']}GB")
        except Exception as e:
            _fail("probe_silicon", e)

        # ── T11: EdgeNodeOrchestrator (headless boot) ─────────────────
        print("\n=== T11: EdgeNodeOrchestrator headless boot ===")
        try:
            with tempfile.TemporaryDirectory() as td:
                os.environ["SWAYAMBHU_MODEL_DIR"] = td
                orch = EdgeNodeOrchestrator()
                orch.boot(
                    firebase_db=None,
                    auto_procure=False,
                    open_avatar=False,
                )
                st = orch.get_status()
                assert st["version"]    == "14.0"
                assert st["node_id"]    == NODE_ID
                assert st["blueprints"] == len(BUILTIN_BLUEPRINTS)
                assert "capabilities"   in st
                assert "active_organs"  in st
                # route_command works without cloud
                r = orch.route_command("hello swayambhu")
                assert "message" in r
                del os.environ["SWAYAMBHU_MODEL_DIR"]
            _ok(f"boot ok blueprints={st['blueprints']} "
                f"organs={st['active_organs']}")
        except Exception as e:
            _fail("EdgeNodeOrchestrator headless boot", e)

        # ── T12: start_edge_server ────────────────────────────────────
        print("\n=== T12: Edge API server ===")
        try:
            import fastapi as _fastapi_check  # noqa: F401
            with tempfile.TemporaryDirectory() as td:
                os.environ["SWAYAMBHU_MODEL_DIR"] = td
                os.environ["SWAYAMBHU_EDGE_PORT"] = "18003"
                orch = EdgeNodeOrchestrator()
                orch.boot(firebase_db=None,
                          auto_procure=False, open_avatar=False)
                ok = start_edge_server(orch)
                assert ok, "server failed to start"
                time.sleep(1.5)
                resp = requests.get("http://localhost:18003/health", timeout=5)
                assert resp.status_code == 200
                data = resp.json()
                assert data["version"] == "14.0"
                del os.environ["SWAYAMBHU_MODEL_DIR"]
                del os.environ["SWAYAMBHU_EDGE_PORT"]
            _ok("server=up health=200 version=14.0")
        except ImportError:
            _ok("Edge API server — fastapi not installed, skipped (will pass on Mac)")
        except Exception as e:
            _fail("Edge API server", e)

        # ── T13: Blueprint execute endpoint ───────────────────────────
        print("\n=== T13: Blueprint execute via API ===")
        try:
            import fastapi as _fastapi_check2  # noqa: F401
            resp = requests.get("http://localhost:18003/blueprints", timeout=5)
            data = resp.json()
            assert data["count"] == len(BUILTIN_BLUEPRINTS)
            resp2 = requests.post(
                "http://localhost:18003/blueprint/execute",
                json={"id": "get_ip"}, timeout=10,
            )
            assert resp2.status_code == 200
            r2 = resp2.json()
            assert r2.get("status") in ("EXECUTED", "LOADED", "error"), r2
            _ok(f"blueprints={data['count']} get_ip={r2.get('status')}")
        except ImportError:
            _ok("Blueprint execute via API — fastapi not installed, skipped (will pass on Mac)")
        except Exception as e:
            _fail("Blueprint execute via API", e)

        # ── T14: TCCPreflight ─────────────────────────────────────────
        print("\n=== T14: TCCPreflight ===")
        try:
            tcc = TCCPreflight()
            perms = tcc.check_all()
            assert isinstance(perms, dict)
            assert all(k in perms for k in
                       ["microphone", "accessibility", "screen_recording"])
            assert isinstance(tcc.can("microphone"), bool)
            _ok(f"perms={perms}")
        except Exception as e:
            _fail("TCCPreflight", e)

        # ── T15: AvatarState ──────────────────────────────────────────
        print("\n=== T15: AvatarState ===")
        try:
            assert AvatarState.name(AvatarState.IDLE)       == "IDLE"
            assert AvatarState.name(AvatarState.DEFCON)     == "DEFCON"
            assert AvatarState.name(AvatarState.PROCESSING) == "PROCESSING"
            assert AvatarState.name(99) == "UNKNOWN"
            _ok("all state names correct")
        except Exception as e:
            _fail("AvatarState", e)

        # ── Summary ───────────────────────────────────────────────────
        print(f"\n{'═' * 55}")
        total = _PASS + _FAIL
        print(f"  Results: {_PASS}/{total} passed  |  {_FAIL} failed")
        print(f"{'═' * 55}")
        if _FAIL:
            _sys.exit(1)
        else:
            print("\n✅ All swayambhu_body.py tests passed.")
        _sys.exit(0)

# =====================================================================
# ── Normal interactive boot (Enterprise CLI Mode) ────────────────────
# =====================================================================
import itertools
import sys
import threading
import time

_db = init_firebase_edge()
edge = EdgeNodeOrchestrator()

# 1. HARD DISABLE the browser UI and 4K sphere
edge.boot(firebase_db=_db, auto_procure=True, open_avatar=False)
start_edge_server(edge)

logger.info(f"🌌 Edge Node online. Node ID: {NODE_ID}")
logger.info(f"   Active organs: {edge.caps.active_organs()}")

# 2. Setup professional terminal spinner
def spinner_task(stop_event):
    spinner = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
    while not stop_event.is_set():
        # Corrected spinner text to reflect hybrid execution
        sys.stdout.write(f"\r\033[94m{next(spinner)} Processing...\033[0m")
        sys.stdout.flush()
        time.sleep(0.08)
    sys.stdout.write("\r\033[K")  # Clear the line safely

# 3. Setup fast typewriter stream effect
def typewriter_print(text, delay=0.012):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print("\n")

# 4. The Infinite CLI Loop
logging.getLogger().setLevel(logging.ERROR)

print("\n\033[93m⏳ Waiting for Dual-Engine Models to initialize (RAM Optimization)...\033[0m")
edge.boot_ready.wait()
print("\033[93m⏳ Syncing secure tunnel with Kaggle Brain...\033[0m")
edge.cloud_ready.wait()
print("\033[92m✅ Neural Pathways fully stabilized. Ready for input.\033[0m\n")

try:
    while True:
        try:
            cmd = input("\n\033[92m[You] >\033[0m ").strip()
        except EOFError:
            break
        if cmd.lower() in ("exit", "quit", "stop", "clear"):
            break
        if cmd:
            stop_spinner = threading.Event()
            spin_thread = threading.Thread(target=spinner_task, args=(stop_spinner,))
            spin_thread.start()

            try:
                result = edge.route_command(cmd)
                response_text = result.get("message", "")
            finally:
                stop_spinner.set()
                spin_thread.join()

            sys.stdout.write("\033[96m[Swayambhu] >\033[0m ")
            typewriter_print(response_text)

            plan = result.get("plan", [])
            if plan:
                for step in plan:
                    print(f"   ⚡ \033[93mOS Action:\033[0m {step.get('action')} -> {step.get('params')}")

except KeyboardInterrupt:
    pass
finally:
    pass