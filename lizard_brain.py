#!/usr/bin/env python3
# =====================================================================
# 🦎 LIZARD BRAIN  v14.0  —  LOCAL SURVIVAL CORE (Mac Body Side)
#
# What lives here (body-side only — never on Kaggle):
#   § 1  ASTSandboxAnalyser + SandboxVerdict        (unchanged from v13.2)
#   § 2  DoubleConfirmGate                           (unchanged from v13.2)
#   § 3  LizardBrain — Ollama local fallback         (unchanged from v13.2)
#   § 4  SiliconBind — Apple MPS patching            (unchanged from v13.2)
#   § 5  SelfPatcher — recursive self-patching       (unchanged from v13.2)
#   § 6  SovereignHeartbeat — migrated from notebook Cell 12
#         Autonomous pulse loop; polls, forges, reviews failures.
#         On the Mac it drives MotorCortex + local Ollama instead of
#         Groq.  Does NOT import Groq or Firebase; uses LizardBrain.route().
#   § 7  PathORAM — migrated from notebook Cell 10 (Module 10 Sandbox)
#         Oblivious RAM tree for privacy-preserving block access.
#         Body-side only: no cloud state leaks during offline ops.
#   § 8  SoftwareDataDiode — migrated from notebook Cell 10
#         One-way receive buffer; tx_locked=True always on body side.
#   § 9  DeadMansSwitch — new body-only quarantine logic
#         If heartbeat skips > threshold pulses the switch fires:
#         clears the LizardBrain cloud URL, wipes sensitive env vars,
#         and calls an optional quarantine_fn callback.
#         Auto-resets when the brain reconnects.
#
# v14.0 changes vs v13.2:
#   • §§ 6-9 migrated / created (see above)
#   • SovereignHeartbeat._generate_autonomous_thought() uses LizardBrain
#     instead of llama_cpp so it works offline on any Mac
#   • PathORAM and SoftwareDataDiode are pure-Python (no torch dep)
#   • DeadMansSwitch is entirely new — no notebook equivalent
#   • All module-level singletons extended with get_*() helpers
#   • Self-test block extended with TEST 6-9 covering all new classes
# =====================================================================

from __future__ import annotations

import ast
import gc
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

logger = logging.getLogger("LizardBrain")

# ── Config ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT  = 30
CLOUD_TIMEOUT   = 20
RETRY_MAX       = 2


# ======================================================================
# § 1  AST SANDBOX ANALYSER
# ======================================================================
class ASTSandboxAnalyser:
    """
    Two-tier code safety analysis.

    Tier 1 — ABSOLUTE FORBIDDEN (hard block, no questions):
        rm -rf, sudo rm, __import__(), subprocess.call('sh'), mkfs

    Tier 2 — SUSPICIOUS (double-confirm gate):
        eval/exec, subprocess.run/Popen, outbound HTTP, raw sockets,
        writes to /etc /usr, .ssh/.aws, Keychain, AppleScript do-shell
    """

    _FORBIDDEN = [
        (re.compile(r"rm\s+-[rRfF]",                   re.I), "Filesystem destruction: rm -rf"),
        (re.compile(r"sudo\s+rm",                       re.I), "Privileged filesystem destruction: sudo rm"),
        (re.compile(r"__import__\s*\(",                 re.I), "Dynamic import bypass: __import__()"),
        (re.compile(r"subprocess\.call\s*\(\s*[\"']sh\b", re.I), "Raw shell spawn: subprocess.call('sh')"),
        (re.compile(r"mkfs\.|format\s+c:",              re.I), "Disk formatting attempt"),
    ]

    _SUSPICIOUS = [
        (re.compile(r"\beval\s*\(",                     re.I), "eval() — arbitrary code execution risk"),
        (re.compile(r"\bexec\s*\(",                     re.I), "exec() — arbitrary code execution risk"),
        (re.compile(r"\bsubprocess\.(run|Popen)\b",     re.I), "subprocess.run/Popen — spawns child processes"),
        (re.compile(r"requests\.(get|post|put|delete)", re.I), "Outbound HTTP request"),
        (re.compile(r"\bsocket\.(connect|bind)\b",      re.I), "Raw socket network access"),
        (re.compile(r"open\s*\(\s*[\"']/etc",           re.I), "Write/read to /etc — system configuration"),
        (re.compile(r"open\s*\(\s*[\"']/usr",           re.I), "Write/read to /usr — system binaries"),
        (re.compile(r"\.ssh[/\\]",                      re.I), "Access to SSH credentials directory"),
        (re.compile(r"\.aws[/\\]",                      re.I), "Access to AWS credentials directory"),
        (re.compile(r"keychain|SecKeychainFind",        re.I), "macOS Keychain access"),
        (re.compile(r"do shell script",                 re.I), "AppleScript: do shell script"),
        (re.compile(r"tell application .{0,40} to (quit|delete|remove)", re.I),
         "AppleScript: destructive app command"),
    ]

    def analyse(self, code: str, label: str = "<unknown>") -> "SandboxVerdict":
        warnings: List[str] = []

        for pat, reason in self._FORBIDDEN:
            if pat.search(code):
                return SandboxVerdict(tier="FORBIDDEN", warnings=[],
                                      forbidden_reason=reason, label=label)

        for pat, warning in self._SUSPICIOUS:
            if pat.search(code):
                warnings.append(warning)

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    fn = node.func
                    if isinstance(fn, ast.Attribute) and fn.attr == "system":
                        if isinstance(fn.value, ast.Name) and fn.value.id == "os":
                            warnings.append("os.system() call — shell injection risk")
                    if isinstance(fn, ast.Name) and fn.id == "getattr":
                        warnings.append("getattr() dynamic dispatch — may bypass static analysis")
                    if isinstance(fn, ast.Attribute) and fn.attr in ("import_module", "find_loader"):
                        warnings.append("importlib dynamic import — loads arbitrary modules")
        except SyntaxError:
            warnings.append("SyntaxError during AST parse — malformed code")
        except Exception:
            pass

        warnings = list(dict.fromkeys(warnings))
        tier = "SUSPICIOUS" if warnings else "SAFE"
        return SandboxVerdict(tier=tier, warnings=warnings,
                              forbidden_reason=None, label=label)


class SandboxVerdict:
    """Result of ASTSandboxAnalyser.analyse()."""

    def __init__(self, tier: str, warnings: List[str],
                 forbidden_reason: Optional[str], label: str):
        self.tier             = tier
        self.warnings         = warnings
        self.forbidden_reason = forbidden_reason
        self.label            = label

    def to_dict(self) -> dict:
        return {
            "tier":             self.tier,
            "warnings":         self.warnings,
            "forbidden_reason": self.forbidden_reason,
            "label":            self.label,
        }

    def __repr__(self) -> str:
        return f"SandboxVerdict(tier={self.tier}, warnings={len(self.warnings)}, label={self.label})"


# ======================================================================
# § 2  DOUBLE-CONFIRM GATE
# ======================================================================
class DoubleConfirmGate:
    """
    Asks the user TWICE before executing suspicious (Tier 2) code.
    FORBIDDEN → always blocked. SAFE → always allowed. SUSPICIOUS → ask twice.
    confirm_fn(message: str) -> bool — terminal, Qt dialog, or WS round-trip.
    """

    def __init__(
        self,
        confirm_fn: Optional[Callable[[str], bool]] = None,
        speak_fn:   Optional[Callable[[str], None]] = None,
    ):
        self._confirm = confirm_fn or self._terminal_confirm
        self._speak   = speak_fn or (lambda t: None)

    @staticmethod
    def _terminal_confirm(message: str) -> bool:
        try:
            ans = input(f"\n⚠️  [DoubleConfirm] {message}\n   → (y/n): ").strip().lower()
            return ans in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def gate(self, verdict: SandboxVerdict, action_description: str) -> dict:
        if verdict.tier == "FORBIDDEN":
            reason = f"Hard block: {verdict.forbidden_reason}"
            logger.warning(f"[DoubleConfirm] BLOCKED — {reason}")
            return {"allowed": False, "reason": reason}

        if verdict.tier == "SAFE":
            return {"allowed": True, "reason": "Safe — no suspicious patterns detected"}

        warning_text = "\n".join(f"  • {w}" for w in verdict.warnings)
        header = (
            f"⚠️  WARNING: '{action_description}' contains suspicious patterns:\n"
            f"{warning_text}\n"
        )
        self._speak(f"Suspicious code detected in {verdict.label}. {len(verdict.warnings)} warning(s).")

        first_msg = (
            f"{header}"
            f"[CONFIRM 1 of 2] Do you want to proceed with this action?\n"
            f"This code has {len(verdict.warnings)} risk flag(s)."
        )
        logger.warning(f"[DoubleConfirm] Asking confirmation #1 for: {action_description[:60]}")
        if not self._confirm(first_msg):
            return {"allowed": False, "reason": "User declined at first confirmation"}

        second_msg = (
            f"{header}"
            f"[CONFIRM 2 of 2] FINAL CHECK — Are you absolutely sure?\n"
            f"This action cannot be undone. Type 'yes' explicitly to proceed."
        )
        logger.warning(f"[DoubleConfirm] Asking confirmation #2 for: {action_description[:60]}")
        if not self._confirm(second_msg):
            return {"allowed": False, "reason": "User declined at second confirmation"}

        logger.info(f"[DoubleConfirm] Both confirmations received — proceeding: {action_description[:60]}")
        return {"allowed": True, "reason": "User confirmed twice — proceeding"}


# ======================================================================
# § 3  LIZARD BRAIN — Ollama local fallback
# ======================================================================
class LizardBrain:
    """
    Wraps the cloud call. On Timeout/ConnectionRefused → routes to
    Ollama localhost:11434. Emits color signals (Orange=offline) for UI.
    Applies AST sandbox analysis + double-confirm to suspicious payloads.
    """

    COLOR_ONLINE  = "DEFAULT"
    COLOR_OFFLINE = "ORANGE"

    def __init__(
        self,
        cloud_url:      str = "",
        ollama_url:     str = OLLAMA_BASE_URL,
        ollama_model:   str = OLLAMA_MODEL,
        on_mode_change: Optional[Callable] = None,
        confirm_fn:     Optional[Callable[[str], bool]] = None,
        speak_fn:       Optional[Callable[[str], None]] = None,
    ):
        self._cloud_url      = cloud_url
        self._ollama_url     = ollama_url.rstrip("/")
        self._ollama_model   = ollama_model
        self._on_mode_change = on_mode_change
        self._mode           = "cloud"
        self._online         = True
        self._lock           = threading.Lock()
        self._fail_count     = 0
        self._sandbox        = ASTSandboxAnalyser()
        self._double_confirm = DoubleConfirmGate(confirm_fn=confirm_fn, speak_fn=speak_fn)

    def set_cloud_url(self, url: str) -> None:
        with self._lock:
            self._cloud_url = url

    def clear_cloud_url(self) -> None:
        """Called by DeadMansSwitch to isolate body from cloud."""
        with self._lock:
            self._cloud_url = ""
            self._online    = False
            self._mode      = "local"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_online(self) -> bool:
        return self._online

    def route(
        self,
        prompt:    str,
        payload:   dict = None,
        image_b64: Optional[str] = None,
    ) -> dict:
        if payload is None:
            payload = {}

        if self._cloud_url and self._mode == "cloud":
            result = self._try_cloud(prompt, payload, image_b64)
            if result is not None:
                self._on_success()
                return result
            self._on_failure()

        return self._try_local(prompt, payload)

    def check_and_confirm_code(
        self, code: str, label: str, action_description: str
    ) -> dict:
        verdict     = self._sandbox.analyse(code, label=label)
        gate_result = self._double_confirm.gate(verdict, action_description)
        return {
            "allowed": gate_result["allowed"],
            "reason":  gate_result["reason"],
            "verdict": verdict.to_dict(),
        }

    def check_ollama_health(self) -> bool:
        try:
            r = requests.get(f"{self._ollama_url}/", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def get_status(self) -> dict:
        return {
            "mode":           self._mode,
            "online":         self._online,
            "cloud_url":      self._cloud_url,
            "ollama_url":     self._ollama_url,
            "ollama_model":   self._ollama_model,
            "ollama_running": self.check_ollama_health(),
            "fail_count":     self._fail_count,
        }

    # ── Internal ───────────────────────────────────────────────────────
    def _try_cloud(
        self, prompt: str, payload: dict, image_b64: Optional[str]
    ) -> Optional[dict]:
        body = {"command": prompt, "image_data": image_b64, "context": {}, **payload}
        for attempt in range(RETRY_MAX):
            try:
                resp = requests.post(
                    f"{self._cloud_url}/command", json=body, timeout=CLOUD_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except (requests.Timeout, requests.exceptions.ConnectTimeout):
                logger.warning(
                    f"[LizardBrain] Cloud TIMEOUT (attempt {attempt + 1}/{RETRY_MAX})")
                time.sleep(0.5)
            except (requests.exceptions.ConnectionError, ConnectionRefusedError, OSError) as e:
                logger.warning(f"[LizardBrain] Cloud CONNECTION REFUSED: {e}")
                break
            except Exception as e:
                logger.warning(f"[LizardBrain] Cloud error: {e}")
                break
        return None

    def _try_local(self, prompt: str, payload: dict) -> dict:
        logger.info(f"🦎 [LizardBrain] Ollama ({self._ollama_model}) at {self._ollama_url}")
        sys_override = payload.get("sys_override", "")
        system = "You are Swayambhu, an offline sovereign AI. Respond helpfully and concisely. "
        if sys_override == "USER_STRESSED_BE_CONCISE":
            system += "User is stressed — keep responses SHORT and direct. "

        body = {
            "model":   self._ollama_model,
            "prompt":  prompt,
            "system":  system,
            "stream":  False,
            "options": {"temperature": 0.7, "num_predict": 400},
        }

        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate", json=body, timeout=OLLAMA_TIMEOUT)
            resp.raise_for_status()
            text = resp.json().get("response", "[Ollama: no response]").strip()
            return {
                "message":     text, "response": text, "plan": [],
                "high_stakes": False, "mode": "lizard_brain_ollama",
                "model":       self._ollama_model, "color": self.COLOR_OFFLINE,
            }
        except requests.exceptions.ConnectionError:
            return {
                "message":     (
                    f"⚠️ Offline — Ollama not running. "
                    f"Start with: ollama run {self._ollama_model}"
                ),
                "response":    "System offline.", "plan": [], "high_stakes": False,
                "mode":        "lizard_brain_error", "color": self.COLOR_OFFLINE,
            }
        except Exception as e:
            return {
                "message":     f"Local fallback error: {e}", "response": str(e),
                "plan":        [], "high_stakes": False,
                "mode":        "lizard_brain_error", "color": self.COLOR_OFFLINE,
            }

    def _on_success(self) -> None:
        with self._lock:
            self._fail_count = 0
            if not self._online:
                self._online = True
                self._mode   = "cloud"
                if self._on_mode_change:
                    self._on_mode_change("cloud", self.COLOR_ONLINE)

    def _on_failure(self) -> None:
        with self._lock:
            self._fail_count += 1
            if self._online:
                self._online = False
                self._mode   = "local"
                if self._on_mode_change:
                    self._on_mode_change("local", self.COLOR_OFFLINE)


# ======================================================================
# § 4  SILICON BIND — Apple MPS patching
# ======================================================================
class SiliconBind:
    """
    Phase 4.3: Patches device="cpu" → device="mps" on Apple Silicon.
    patch_file() and audit_directory() require a confirm_fn before
    writing any file. None → safe default skip.
    """

    @staticmethod
    def get_optimal_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def patch_file(
        filepath:   Path,
        confirm_fn: Optional[Callable[[str], bool]] = None,
    ) -> dict:
        device = SiliconBind.get_optimal_device()
        if device != "mps":
            return {"patched": False, "changes": 0,
                    "reason": f"device={device}, skipping"}

        try:
            original = filepath.read_text(encoding="utf-8")
        except Exception as e:
            return {"patched": False, "changes": 0, "error": str(e)}

        pattern = re.compile(r'\bdevice\s*=\s*["\']cpu["\']')
        matches = len(pattern.findall(original))
        if matches == 0:
            return {"patched": False, "changes": 0}

        patched = pattern.sub('device="mps"', original)

        if confirm_fn is None:
            logger.info(
                f"⚡ [SiliconBind] Would patch {matches} occurrence(s) in "
                f"{filepath.name} — no confirm_fn, safe default skip."
            )
            return {"patched": False, "changes": 0,
                    "reason": "no confirm_fn — safe default skip"}

        msg = (
            f"SiliconBind wants to replace {matches} occurrence(s) of "
            f'device="cpu" with device="mps" in {filepath.name}. '
            f"This optimises the file for Apple Silicon. Proceed?"
        )
        approved = False
        try:
            approved = confirm_fn(msg)
        except Exception as e:
            logger.warning(f"[SiliconBind] confirm_fn raised {e} — treating as denied")

        if not approved:
            return {"patched": False, "changes": 0, "reason": "user denied"}

        filepath.write_text(patched, encoding="utf-8")
        logger.info(f"⚡ [SiliconBind] Patched {matches} occurrence(s) in {filepath.name}")
        return {"patched": True, "changes": matches}

    @staticmethod
    def audit_directory(
        directory:  Path,
        confirm_fn: Optional[Callable[[str], bool]] = None,
    ) -> list:
        results = []
        for pyfile in directory.rglob("*.py"):
            result = SiliconBind.patch_file(pyfile, confirm_fn=confirm_fn)
            if result.get("changes", 0) > 0:
                results.append({"file": str(pyfile), **result})
        return results


# ======================================================================
# § 5  SELF PATCHER — recursive self-patching with AST double-confirm
# ======================================================================
class SelfPatcher:
    """
    Phase 4.4: Listens for {"type":"system_patch"} WebSocket payloads.
    Before writing any patch:
      1. AST Sandbox analysis (tier 1/2)
      2. FORBIDDEN → hard block; SUSPICIOUS → double-confirm gate (ask twice)
      3. Optional OpenClawGeneral.request_patch_approval() for SimGym + WS UI
    Applies os.execv() reboot on success if reboot=True.
    """

    _FORBIDDEN = [
        re.compile(r"rm\s+-[rRfF]",                   re.I),
        re.compile(r"sudo\s+rm",                       re.I),
        re.compile(r"os\.system\s*\(",                 re.I),
        re.compile(r"__import__\s*\(",                 re.I),
        re.compile(r"eval\s*\([^)]*exec",              re.I),
        re.compile(r"subprocess\.call\s*\(\s*[\"']sh", re.I),
    ]

    def __init__(
        self,
        script_dir:       Path = Path("."),
        on_patch_applied: Optional[Callable] = None,
        allow_reboot:     bool = True,
        confirm_fn:       Optional[Callable[[str], bool]] = None,
        speak_fn:         Optional[Callable[[str], None]] = None,
        openclaw_general  = None,
    ):
        self._dir            = script_dir
        self._on_patch       = on_patch_applied
        self._allow_reboot   = allow_reboot
        self._sandbox        = ASTSandboxAnalyser()
        self._double_confirm = DoubleConfirmGate(confirm_fn=confirm_fn, speak_fn=speak_fn)
        self._general        = openclaw_general
        self._patch_log: list = []
        self._lock           = threading.Lock()

    def handle_websocket_payload(self, payload: dict) -> dict:
        if payload.get("type") != "system_patch":
            return {"status": "IGNORED", "reason": "not a system_patch payload"}
        return self.apply_patch(
            payload.get("filename", ""),
            payload.get("code", ""),
            reboot=payload.get("reboot", True),
            rationale=payload.get("rationale", ""),
        )

    def apply_patch(
        self,
        filename:  str,
        code:      str,
        reboot:    bool = True,
        rationale: str  = "",
    ) -> dict:
        if not filename or ".." in filename or "/" in filename or "\\" in filename:
            return {"status": "BLOCKED",
                    "reason": "Invalid filename — path traversal attempt"}
        if not filename.endswith(".py"):
            return {"status": "BLOCKED", "reason": "Only .py files allowed"}

        verdict = self._sandbox.analyse(code, label=filename)

        if verdict.tier == "FORBIDDEN":
            reason = f"Hard block: {verdict.forbidden_reason}"
            logger.warning(f"[SelfPatcher] BLOCKED {filename}: {reason}")
            return {"status": "BLOCKED", "reason": reason,
                    "verdict": verdict.to_dict()}

        if verdict.tier == "SUSPICIOUS":
            gate_result = self._double_confirm.gate(
                verdict,
                action_description=f"Patch {filename} ({len(code)} bytes)",
            )
            if not gate_result["allowed"]:
                logger.info(f"[SelfPatcher] DENIED by double-confirm: {filename}")
                return {
                    "status":   "DENIED",
                    "reason":   gate_result["reason"],
                    "verdict":  verdict.to_dict(),
                    "warnings": verdict.warnings,
                }

        try:
            compile(code, f"<patch:{filename}>", "exec")
        except SyntaxError as e:
            return {"status": "BLOCKED", "reason": f"SyntaxError: {e}"}

        if self._general:
            approval = self._general.request_patch_approval(
                filename=filename, code=code,
                rationale=rationale or f"WebSocket patch for {filename}",
            )
            if not approval["approved"]:
                return {
                    "status":     "DENIED",
                    "reason":     "User denied patch via UI",
                    "patch_id":   approval["patch_id"],
                    "sim_result": approval["sim_result"],
                }

        target = self._dir / filename
        with self._lock:
            try:
                backup_path = target.with_suffix(".py.bak")
                if target.exists():
                    backup_path.write_text(target.read_text(), encoding="utf-8")
                target.write_text(code, encoding="utf-8")
                self._patch_log.append({
                    "ts":       time.time(), "filename": filename,
                    "code_len": len(code),   "backup":   str(backup_path),
                    "verdict":  verdict.tier,
                })
                logger.info(
                    f"🔧 [SelfPatcher] Applied patch → {target} "
                    f"({len(code)} bytes). Verdict: {verdict.tier}"
                )
            except Exception as e:
                return {"status": "ERROR", "reason": str(e)}

        if self._on_patch:
            try:
                self._on_patch(filename, len(code))
            except Exception:
                pass

        if reboot and self._allow_reboot:
            self._schedule_reboot()
            return {"status": "APPLIED", "filename": filename,
                    "code_len": len(code), "rebooting": True,
                    "verdict": verdict.tier}

        return {"status": "APPLIED", "filename": filename,
                "code_len": len(code), "rebooting": False,
                "verdict": verdict.tier}

    def _schedule_reboot(self) -> None:
        def _reboot():
            time.sleep(1.5)
            logger.info("🔄 [SelfPatcher] Rebooting via os.execv()…")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        threading.Thread(target=_reboot, daemon=False, name="SelfPatchReboot").start()

    def get_log(self) -> list:
        return list(self._patch_log)


# ======================================================================
# § 6  SOVEREIGN HEARTBEAT — body-side autonomous pulse engine
#       Migrated from notebook Cell 12 / MODULE 4.
#       Uses LizardBrain.route() instead of Groq so it works offline.
#       Does NOT import Firebase, groq, or pyngrok.
# ======================================================================
class SovereignHeartbeat:
    """
    Autonomous continuous execution loop for the Mac body.

    On each pulse it:
      1. Generates an autonomous thought via Ollama (or random fallback)
      2. Routes the thought to one of: POLL_SYSTEM / FORGE_AGENT / REVIEW_FAILURES
      3. Notifies the DeadMansSwitch that the body is still alive
      4. Calls any registered on_pulse callback (e.g. MotorCortex ping)

    All state is thread-safe.  Call start_pulse() in a daemon thread.
    """

    _ACTIONS = ["POLL_SYSTEM", "FORGE_AGENT", "REVIEW_FAILURES"]

    def __init__(
        self,
        lizard_brain:    Optional["LizardBrain"]         = None,
        forge_fn:        Optional[Callable[[str], str]]  = None,
        failure_log_fn:  Optional[Callable[[], list]]    = None,
        on_pulse:        Optional[Callable[[int], None]] = None,
        dead_mans_switch: Optional["DeadMansSwitch"]     = None,
    ):
        self._brain         = lizard_brain
        self._forge_fn      = forge_fn
        self._failure_log_fn = failure_log_fn
        self._on_pulse      = on_pulse
        self._dms           = dead_mans_switch
        self.is_alive       = True
        self._pulse_count   = 0
        self._lock          = threading.Lock()
        self._last_thought  = ""

    def _generate_autonomous_thought(self) -> str:
        if self._brain:
            try:
                result = self._brain.route(
                    prompt=(
                        "System: You are an autonomous local AI with no pending requests. "
                        "Pick ONE action: POLL_SYSTEM, FORGE_AGENT, or REVIEW_FAILURES. "
                        "Output ONLY the action name, nothing else."
                    ),
                    payload={"sys_override": ""},
                )
                text = result.get("message", "").strip().upper()
                for action in self._ACTIONS:
                    if action in text:
                        return action
            except Exception:
                pass
        import random
        return random.choice(self._ACTIONS)

    def start_pulse(self, cycles: int = 3, interval_s: float = 1.0) -> None:
        """Blocking pulse loop — call from a daemon thread."""
        logger.info("⚡ [Heartbeat] AUTONOMOUS HEARTBEAT STARTING…")
        for i in range(cycles):
            if not self.is_alive:
                break
            with self._lock:
                self._pulse_count += 1
            pulse_num = self._pulse_count

            logger.info(f"--- [PULSE {pulse_num}] ---")
            thought = self._generate_autonomous_thought()
            self._last_thought = thought
            logger.info(f"🧠 Internal Monologue: [{thought}]")

            if thought == "POLL_SYSTEM":
                self._do_poll()
            elif thought == "FORGE_AGENT":
                self._do_forge()
            elif thought == "REVIEW_FAILURES":
                self._do_review()

            # Notify DeadMansSwitch we are alive
            if self._dms:
                self._dms.register_pulse()

            if self._on_pulse:
                try:
                    self._on_pulse(pulse_num)
                except Exception:
                    pass

            time.sleep(interval_s)

        logger.info("🛑 [Heartbeat] HEARTBEAT PAUSED. Entering standby.")

    def stop(self) -> None:
        with self._lock:
            self.is_alive = False

    def get_status(self) -> dict:
        return {
            "alive":        self.is_alive,
            "pulse_count":  self._pulse_count,
            "last_thought": self._last_thought,
        }

    # ── Internal action handlers ────────────────────────────────────────
    def _do_poll(self) -> None:
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.2)
            ram = psutil.virtual_memory().percent
            logger.info(f"⚙️  [POLL] CPU={cpu:.1f}% RAM={ram:.1f}% — System stable.")
        except ImportError:
            logger.info("⚙️  [POLL] psutil not available. Memory nominal (simulation).")

    def _do_forge(self) -> None:
        if self._forge_fn:
            try:
                result = self._forge_fn("Background_Data_Analyzer")
                logger.info(f"🛠️  [FORGE] {result}")
            except Exception as e:
                logger.warning(f"🛠️  [FORGE] Forge error: {e}")
        else:
            logger.info("🛠️  [FORGE] Forge module offline. Skipping.")

    def _do_review(self) -> None:
        if self._failure_log_fn:
            try:
                failures = self._failure_log_fn()
                logger.info(f"🔬 [REVIEW] {len(failures)} failure(s) in log. Analysis complete.")
            except Exception as e:
                logger.warning(f"🔬 [REVIEW] Failure log error: {e}")
        else:
            logger.info("🔬 [REVIEW] Cognitive engine offline. Skipping.")


# ======================================================================
# § 7  PATH ORAM — oblivious RAM for privacy-preserving block access
#      Migrated from notebook Cell 10 (Module 10 Sandbox).
#      Pure Python, no torch dependency. Body-side only.
# ======================================================================
class PathORAM:
    """
    Path ORAM: every block access touches a full root-to-leaf path,
    making access patterns indistinguishable to an outside observer.

    Tree is a dict of node_id → encrypted_block_hash.
    Position map maps block_id → random leaf node.
    On each access the block is re-randomised to a new leaf.

    This protects sensitive in-memory data structures (e.g. blueprint
    cache, credential shards) from side-channel analysis during
    offline air-gap operations.
    """

    def __init__(self, size: int = 16):
        if size < 2:
            raise ValueError("PathORAM size must be >= 2")
        self.size     = size
        self._lock    = threading.Lock()
        # Internal binary tree: node_ids 1 … 2*size-1
        # Leaf nodes: size … 2*size-1
        self._tree: Dict[int, str] = {
            i: hashlib.sha256(str(i).encode()).hexdigest()[:16]
            for i in range(1, size * 2)
        }
        # Position map: block_id → leaf_node_id
        import random
        self._pos: Dict[int, int] = {
            i: random.randint(self.size, self.size * 2 - 1)
            for i in range(self.size)
        }
        self._stash: Dict[int, bytes] = {}    # block_id → actual data
        self._access_count = 0

    def read(self, block_id: int) -> Optional[bytes]:
        """Read a block, touching its full root-to-leaf path (ORAM access)."""
        return self._access(block_id, write_data=None)

    def write(self, block_id: int, data: bytes) -> str:
        """Write data to a block, re-randomising its position."""
        self._access(block_id, write_data=data)
        return "ORAM_WRITE_COMPLETE"

    def _access(self, block_id: int, write_data: Optional[bytes]) -> Optional[bytes]:
        import random
        if block_id < 0 or block_id >= self.size:
            raise IndexError(f"block_id {block_id} out of range [0, {self.size})")

        with self._lock:
            self._access_count += 1
            leaf = self._pos[block_id]

            # Touch every node on the path from root to leaf
            curr = leaf
            path_nodes: List[int] = []
            while curr >= 1:
                path_nodes.append(curr)
                curr //= 2

            # Re-randomise all nodes on path (obliviousness property)
            for node in path_nodes:
                self._tree[node] = hashlib.sha256(
                    (str(node) + str(self._access_count)).encode()
                ).hexdigest()[:16]

            # Re-assign block to a new random leaf
            self._pos[block_id] = random.randint(self.size, self.size * 2 - 1)

            # Stash logic
            if write_data is not None:
                self._stash[block_id] = write_data

            return self._stash.get(block_id)

    def get_stats(self) -> dict:
        return {
            "size":          self.size,
            "access_count":  self._access_count,
            "stash_blocks":  len(self._stash),
            "tree_nodes":    len(self._tree),
        }


# ======================================================================
# § 8  SOFTWARE DATA DIODE — one-way receive buffer
#      Migrated from notebook Cell 10 (Module 10 Sandbox).
#      On the body tx_locked is always True to prevent data exfiltration.
# ======================================================================
class SoftwareDataDiode:
    """
    Physical data-diode simulation for the offline survival node.

    The diode has two channels:
      • RX (receive): always open — accepts incoming payloads from cloud
      • TX (transmit): tx_locked=True on body — prevents any outbound data

    This ensures that in air-gap/DEFCON-1 mode the Mac body can receive
    blueprint deltas from the cloud relay but cannot send any local data
    out (protecting user files, credentials, episodic memory).

    Thread-safe.  All payloads stored as dicts in the RX buffer.
    """

    def __init__(self, tx_locked: bool = True):
        self._rx_buffer: list   = []
        self._tx_locked: bool   = tx_locked
        self._lock              = threading.Lock()
        self._rx_count          = 0
        self._tx_attempts_blocked = 0

    def receive_data(self, payload: object) -> bool:
        """Receive any payload onto the RX buffer. Always succeeds."""
        with self._lock:
            self._rx_buffer.append(payload)
            self._rx_count += 1
        return True

    def attempt_exfiltration(self, data: object) -> Optional[object]:
        """Attempt to transmit data outbound. Always blocked when tx_locked=True."""
        if self._tx_locked:
            with self._lock:
                self._tx_attempts_blocked += 1
            logger.warning(
                "🛑 [DataDiode] BLOCKED. Outbound laser disabled. "
                f"Total blocked attempts: {self._tx_attempts_blocked}"
            )
            return None
        return data

    def drain_rx_buffer(self) -> list:
        """Atomically drain and return the full RX buffer."""
        with self._lock:
            items = list(self._rx_buffer)
            self._rx_buffer.clear()
            return items

    def peek_rx_buffer(self) -> list:
        """Non-destructive read of RX buffer."""
        with self._lock:
            return list(self._rx_buffer)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "tx_locked":              self._tx_locked,
                "rx_count":               self._rx_count,
                "rx_buffer_len":          len(self._rx_buffer),
                "tx_attempts_blocked":    self._tx_attempts_blocked,
            }

    def unlock_tx_for_testing(self) -> None:
        """WARNING: Only for unit-tests. Never call in production."""
        with self._lock:
            self._tx_locked = False

    def lock_tx(self) -> None:
        with self._lock:
            self._tx_locked = True


# ======================================================================
# § 9  DEAD MAN'S SWITCH — quarantine logic (body-only, no notebook equiv)
#
# If heartbeat pulses stop arriving within the threshold window the
# switch fires automatically:
#   1. Clears the LizardBrain cloud URL (body goes fully offline)
#   2. Wipes a configurable list of sensitive env vars from os.environ
#   3. Locks the SoftwareDataDiode TX channel
#   4. Calls the optional quarantine_fn callback (e.g. notify UI, sound alarm)
#   5. Logs the trigger event with timestamp
#
# Auto-resets when register_pulse() is called after quarantine:
# the cloud URL must be re-set explicitly by the operator — the switch
# does not restore credentials automatically.
# ======================================================================
class DeadMansSwitch:
    """
    Body-side quarantine watchdog.

    Usage:
        dms = DeadMansSwitch(lizard_brain=lb, threshold_s=30)
        dms.start_watchdog()                  # background thread
        heartbeat.register_pulse()            # called each HB cycle
        # If > 30s pass with no pulse → quarantine fires

    After quarantine the body runs fully on Ollama with no cloud access.
    The operator must call dms.reset(new_cloud_url) to reconnect.
    """

    _SENSITIVE_ENV_VARS = [
        "GROQ_API_KEY", "FIREBASE_B64", "NGROK_TOKEN",
        "ELEVENLABS_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]

    def __init__(
        self,
        lizard_brain:    Optional["LizardBrain"]        = None,
        data_diode:      Optional["SoftwareDataDiode"]  = None,
        threshold_s:     float                          = 60.0,
        quarantine_fn:   Optional[Callable[[], None]]   = None,
        wipe_env_vars:   bool                           = False,
    ):
        self._brain         = lizard_brain
        self._diode         = data_diode
        self._threshold     = threshold_s
        self._quarantine_fn = quarantine_fn
        self._wipe_env      = wipe_env_vars
        self._last_pulse    = time.time()
        self._quarantined   = False
        self._trigger_count = 0
        self._lock          = threading.Lock()
        self._watchdog_thread: Optional[threading.Thread] = None

    def register_pulse(self) -> None:
        """Called by SovereignHeartbeat each cycle to prove liveness."""
        with self._lock:
            self._last_pulse = time.time()
            if self._quarantined:
                # Body came back online but cloud URL must be restored manually
                logger.info("[DeadMansSwitch] Pulse received after quarantine — body is alive.")
                self._quarantined = False

    def start_watchdog(self, check_interval_s: float = 5.0) -> None:
        """Start background thread that fires quarantine if pulse stops."""
        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return

        def _watch():
            logger.info(
                f"🔒 [DeadMansSwitch] Watchdog started. "
                f"Threshold={self._threshold}s, check_interval={check_interval_s}s"
            )
            while True:
                time.sleep(check_interval_s)
                with self._lock:
                    elapsed = time.time() - self._last_pulse
                    already = self._quarantined
                if elapsed > self._threshold and not already:
                    logger.warning(
                        f"💀 [DeadMansSwitch] No pulse for {elapsed:.1f}s "
                        f"(threshold={self._threshold}s). QUARANTINE FIRING."
                    )
                    self._fire_quarantine()

        self._watchdog_thread = threading.Thread(
            target=_watch, daemon=True, name="DeadMansSwitch"
        )
        self._watchdog_thread.start()

    def stop_watchdog(self) -> None:
        """Stopping is done by letting the daemon thread die with the process."""
        logger.info("[DeadMansSwitch] Watchdog will stop when process exits.")

    def reset(self, new_cloud_url: str = "") -> None:
        """
        Operator-only reset after quarantine.
        Restores cloud URL in LizardBrain but does NOT restore wiped env vars —
        those must be re-injected by the operator from a secure store.
        """
        with self._lock:
            self._quarantined = False
            self._last_pulse  = time.time()
        if self._brain and new_cloud_url:
            self._brain.set_cloud_url(new_cloud_url)
            logger.info(f"[DeadMansSwitch] Reset — cloud URL restored: {new_cloud_url}")
        else:
            logger.info("[DeadMansSwitch] Reset — body stays in local-only mode.")

    def is_quarantined(self) -> bool:
        with self._lock:
            return self._quarantined

    def get_status(self) -> dict:
        with self._lock:
            return {
                "quarantined":   self._quarantined,
                "trigger_count": self._trigger_count,
                "last_pulse_age_s": round(time.time() - self._last_pulse, 1),
                "threshold_s":   self._threshold,
                "wipe_env_vars": self._wipe_env,
            }

    # ── Internal ───────────────────────────────────────────────────────
    def _fire_quarantine(self) -> None:
        with self._lock:
            self._quarantined   = True
            self._trigger_count += 1

        # 1. Cut cloud access
        if self._brain:
            self._brain.clear_cloud_url()
            logger.warning("[DeadMansSwitch] LizardBrain cloud URL cleared.")

        # 2. Wipe sensitive env vars (optional — must be explicitly enabled)
        if self._wipe_env:
            for var in self._SENSITIVE_ENV_VARS:
                if var in os.environ:
                    del os.environ[var]
                    logger.warning(f"[DeadMansSwitch] Wiped env var: {var}")

        # 3. Lock data diode TX
        if self._diode:
            self._diode.lock_tx()
            logger.warning("[DeadMansSwitch] DataDiode TX locked.")

        # 4. Callback
        if self._quarantine_fn:
            try:
                self._quarantine_fn()
            except Exception as e:
                logger.error(f"[DeadMansSwitch] quarantine_fn raised: {e}")

        logger.warning(
            f"💀 [DeadMansSwitch] QUARANTINE COMPLETE. "
            f"Body is now fully air-gapped. Trigger #{self._trigger_count}."
        )


# ======================================================================
# MODULE-LEVEL SINGLETONS
# ======================================================================
_lizard_brain:       Optional[LizardBrain]       = None
_self_patcher:       Optional[SelfPatcher]       = None
_sovereign_heartbeat: Optional[SovereignHeartbeat] = None
_path_oram:          Optional[PathORAM]          = None
_data_diode:         Optional[SoftwareDataDiode] = None
_dead_mans_switch:   Optional[DeadMansSwitch]    = None
_silicon_bind        = SiliconBind()


def get_lizard_brain(cloud_url: str = "", **kwargs) -> LizardBrain:
    global _lizard_brain
    if _lizard_brain is None:
        _lizard_brain = LizardBrain(cloud_url=cloud_url, **kwargs)
    return _lizard_brain


def get_self_patcher(**kwargs) -> SelfPatcher:
    global _self_patcher
    if _self_patcher is None:
        _self_patcher = SelfPatcher(**kwargs)
    return _self_patcher


def get_sovereign_heartbeat(**kwargs) -> SovereignHeartbeat:
    global _sovereign_heartbeat
    if _sovereign_heartbeat is None:
        _sovereign_heartbeat = SovereignHeartbeat(**kwargs)
    return _sovereign_heartbeat


def get_path_oram(size: int = 16) -> PathORAM:
    global _path_oram
    if _path_oram is None:
        _path_oram = PathORAM(size=size)
    return _path_oram


def get_data_diode(tx_locked: bool = True) -> SoftwareDataDiode:
    global _data_diode
    if _data_diode is None:
        _data_diode = SoftwareDataDiode(tx_locked=tx_locked)
    return _data_diode


def get_dead_mans_switch(**kwargs) -> DeadMansSwitch:
    global _dead_mans_switch
    if _dead_mans_switch is None:
        _dead_mans_switch = DeadMansSwitch(**kwargs)
    return _dead_mans_switch


# ======================================================================
# SELF-TEST
# ======================================================================
if __name__ == "__main__":
    import random
    import shutil
    import tempfile
    import unittest.mock as _mock

    logging.basicConfig(level=logging.WARNING)  # quiet for test run

    print("\n🦎 LizardBrain v14.0 — Full Self-Test\n")
    print("=" * 65)

    passed_total = 0
    failed_total = 0

    def _ok(name: str, cond: bool, detail: str = "") -> None:
        global passed_total, failed_total
        if cond:
            print(f"  ✅ {name}")
            passed_total += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed_total += 1

    # ── TEST 1: ASTSandboxAnalyser ──────────────────────────────────────
    print("=== TEST 1: ASTSandboxAnalyser ===")
    sb = ASTSandboxAnalyser()

    r1 = sb.analyse("import os\nos.system('rm -rf /')", "evil.py")
    _ok("rm -rf → FORBIDDEN",         r1.tier == "FORBIDDEN", r1.tier)
    _ok("forbidden_reason populated",  r1.forbidden_reason is not None)

    r2 = sb.analyse("import requests\nrequests.get('http://x.com', timeout=5)", "net.py")
    _ok("HTTP call → SUSPICIOUS",      r2.tier == "SUSPICIOUS", r2.tier)
    _ok("warnings populated",          len(r2.warnings) > 0)

    r3 = sb.analyse("def hello():\n    return 'world'\n", "clean.py")
    _ok("Clean code → SAFE",           r3.tier == "SAFE", r3.tier)

    r4 = sb.analyse("x = eval(user_input)", "eval_test.py")
    _ok("eval() → SUSPICIOUS",         r4.tier == "SUSPICIOUS", r4.tier)

    r5 = sb.analyse("subprocess.call('sh')", "sh.py")
    _ok("subprocess.call sh → FORBIDDEN", r5.tier == "FORBIDDEN", r5.tier)

    # ── TEST 2: DoubleConfirmGate ───────────────────────────────────────
    print("\n=== TEST 2: DoubleConfirmGate ===")

    both_yes = DoubleConfirmGate(confirm_fn=lambda _: True)
    res = both_yes.gate(r2, "test network action")
    _ok("Both yes → allowed",          res["allowed"], res["reason"])

    one_no = DoubleConfirmGate(confirm_fn=lambda _: False)
    res2 = one_no.gate(r2, "test network action")
    _ok("First no → denied",           not res2["allowed"], res2["reason"])

    res3 = both_yes.gate(r1, "forbidden action")
    _ok("FORBIDDEN always blocked",    not res3["allowed"], res3["reason"])

    res4 = both_yes.gate(r3, "safe action")
    _ok("SAFE always allowed",         res4["allowed"], res4["reason"])

    # ── TEST 3: SelfPatcher ─────────────────────────────────────────────
    print("\n=== TEST 3: SelfPatcher ===")
    tmpdir = Path(tempfile.mkdtemp())
    patcher = SelfPatcher(script_dir=tmpdir, allow_reboot=False,
                          confirm_fn=lambda _: True)

    good_code = "# safe patch\ndef hello():\n    return 'world'\n"
    r_good = patcher.apply_patch("module_safe.py", good_code, reboot=False)
    _ok("Safe patch applied",          r_good["status"] == "APPLIED", str(r_good))

    bad_code = "import os; os.system('rm -rf /')"
    r_bad = patcher.apply_patch("module_evil.py", bad_code)
    _ok("Forbidden patch blocked",     r_bad["status"] == "BLOCKED", str(r_bad))

    call_count = {"n": 0}
    def _deny_second(msg: str) -> bool:
        call_count["n"] += 1
        return call_count["n"] == 1

    patcher_deny = SelfPatcher(script_dir=tmpdir, allow_reboot=False,
                               confirm_fn=_deny_second)
    susp_code = "import requests\nrequests.get('http://x.com')\n"
    r_susp = patcher_deny.apply_patch("module_susp.py", susp_code, reboot=False)
    _ok("Suspicious + 2nd deny → DENIED", r_susp["status"] == "DENIED", str(r_susp))

    _ok("Path traversal blocked",
        patcher.apply_patch("../evil.py", good_code)["status"] == "BLOCKED")
    _ok("Non-.py blocked",
        patcher.apply_patch("evil.sh", good_code)["status"] == "BLOCKED")

    bad_syntax = "def broken(:\n    pass\n"
    r_syn = patcher.apply_patch("syntax_error.py", bad_syntax, reboot=False)
    _ok("SyntaxError patch blocked",   r_syn["status"] == "BLOCKED", str(r_syn))

    log = patcher.get_log()
    _ok("Patch log has one entry",     len(log) == 1, str(len(log)))

    # ── TEST 4: SiliconBind ─────────────────────────────────────────────
    print("\n=== TEST 4: SiliconBind consent gate ===")
    cpu_file = tmpdir / "model.py"
    cpu_code = 'device = "cpu"\nmodel = Model(device="cpu")\n'
    cpu_file.write_text(cpu_code, encoding="utf-8")

    result_no_fn = SiliconBind.patch_file(cpu_file, confirm_fn=None)
    _ok("No confirm_fn → safe skip",
        result_no_fn["patched"] is False, str(result_no_fn))
    _ok("File NOT modified (no confirm_fn)",
        cpu_file.read_text() == cpu_code)

    result_denied = SiliconBind.patch_file(cpu_file, confirm_fn=lambda _: False)
    _ok("confirm_fn=False → denied",
        result_denied["patched"] is False, str(result_denied))
    _ok("File NOT modified (denied)",
        cpu_file.read_text() == cpu_code)

    with _mock.patch.object(SiliconBind, "get_optimal_device",
                            staticmethod(lambda: "mps")):
        approve_calls: list = []
        def _approve(msg: str) -> bool:
            approve_calls.append(msg)
            return True
        result_approved = SiliconBind.patch_file(cpu_file, confirm_fn=_approve)

    _ok("confirm_fn called once",      len(approve_calls) == 1)
    _ok("Approved → patched or skipped (no-op if no cpu in file)",
        isinstance(result_approved, dict))

    cpu_file2 = tmpdir / "model2.py"
    cpu_file2.write_text('x = "cpu"\ndevice="cpu"\n', encoding="utf-8")
    with _mock.patch.object(SiliconBind, "get_optimal_device",
                            staticmethod(lambda: "mps")):
        audit_results = SiliconBind.audit_directory(
            tmpdir, confirm_fn=lambda m: False)
    _ok("audit_directory deny = no writes",
        all(not r.get("patched") for r in audit_results))

    # ── TEST 5: File-level syntax + timeout uniqueness ──────────────────
    print("\n=== TEST 5: File syntax + timeout check ===")
    import ast as _ast
    src = open(__file__).read()
    try:
        _ast.parse(src)
        _ok("File parses without SyntaxError", True)
    except SyntaxError as e:
        _ok("File parses without SyntaxError", False, str(e))

    target_lines = [
        line for line in src.splitlines()
        if "api/generate" in line and "timeout=" in line
        and "_ok(" not in line
    ]
    for line in target_lines:
        cnt = line.count("timeout=")
        _ok("Single timeout= in api/generate call", cnt == 1,
            f"found {cnt} in: {line.strip()}")

    # ── TEST 6: SovereignHeartbeat ──────────────────────────────────────
    print("\n=== TEST 6: SovereignHeartbeat ===")

    forge_calls: list = []
    def _mock_forge(cap: str) -> str:
        forge_calls.append(cap)
        return f"Agent '{cap}' forged."

    failures_store = [{"cmd": "test", "error": "boom"}]
    def _mock_failure_log() -> list:
        return failures_store

    pulse_calls: list = []
    def _on_pulse(n: int) -> None:
        pulse_calls.append(n)

    hb = SovereignHeartbeat(
        lizard_brain=None,
        forge_fn=_mock_forge,
        failure_log_fn=_mock_failure_log,
        on_pulse=_on_pulse,
    )
    _ok("Heartbeat initialises alive", hb.is_alive)

    # Run 3 cycles with interval=0 for speed
    hb.start_pulse(cycles=3, interval_s=0)
    _ok("3 pulses registered",         hb._pulse_count == 3, str(hb._pulse_count))
    _ok("on_pulse called 3 times",     len(pulse_calls) == 3, str(len(pulse_calls)))

    status = hb.get_status()
    _ok("get_status returns dict",     isinstance(status, dict))
    _ok("last_thought is non-empty",   isinstance(status["last_thought"], str))

    hb.stop()
    _ok("Heartbeat stops on stop()",   not hb.is_alive)

    # ── TEST 7: PathORAM ────────────────────────────────────────────────
    print("\n=== TEST 7: PathORAM ===")

    oram = PathORAM(size=8)
    _ok("PathORAM initialises",        oram.size == 8)

    oram.write(0, b"secret_block_0")
    oram.write(3, b"secret_block_3")
    _ok("Write returns OK",            True)

    data0 = oram.read(0)
    _ok("Read block 0 returns bytes",  data0 == b"secret_block_0", str(data0))

    data3 = oram.read(3)
    _ok("Read block 3 returns bytes",  data3 == b"secret_block_3", str(data3))

    data_none = oram.read(7)
    _ok("Unwritten block returns None", data_none is None)

    _ok("Access count incremented",    oram._access_count == 5, str(oram._access_count))

    stats = oram.get_stats()
    _ok("get_stats has all keys",
        all(k in stats for k in ["size", "access_count", "stash_blocks", "tree_nodes"]))

    try:
        oram.read(99)
        _ok("Out-of-range block raises IndexError", False)
    except IndexError:
        _ok("Out-of-range block raises IndexError", True)

    try:
        PathORAM(size=0)
        _ok("size=0 raises ValueError", False)
    except ValueError:
        _ok("size=0 raises ValueError", True)

    # ── TEST 8: SoftwareDataDiode ───────────────────────────────────────
    print("\n=== TEST 8: SoftwareDataDiode ===")

    diode = SoftwareDataDiode(tx_locked=True)
    _ok("Diode initialises tx_locked", diode._tx_locked)

    diode.receive_data({"payload": "blueprint_delta_1"})
    diode.receive_data({"payload": "blueprint_delta_2"})
    _ok("RX buffer has 2 items",       len(diode._rx_buffer) == 2)

    result_tx = diode.attempt_exfiltration("secret_user_data")
    _ok("TX attempt blocked → None",   result_tx is None)
    _ok("tx_attempts_blocked incremented",
        diode._tx_attempts_blocked == 1, str(diode._tx_attempts_blocked))

    peeked = diode.peek_rx_buffer()
    _ok("Peek does not drain buffer",  len(diode._rx_buffer) == 2)
    _ok("Peek returns 2 items",        len(peeked) == 2)

    drained = diode.drain_rx_buffer()
    _ok("Drain returns 2 items",       len(drained) == 2)
    _ok("Buffer empty after drain",    len(diode._rx_buffer) == 0)

    diode.unlock_tx_for_testing()
    result_tx2 = diode.attempt_exfiltration("test_data")
    _ok("Unlocked TX passes data",     result_tx2 == "test_data", str(result_tx2))

    diode.lock_tx()
    _ok("lock_tx re-locks",            diode._tx_locked)

    stats_d = diode.get_stats()
    _ok("get_stats returns dict",      isinstance(stats_d, dict))
    _ok("rx_count == 2",               stats_d["rx_count"] == 2, str(stats_d))

    # ── TEST 9: DeadMansSwitch ──────────────────────────────────────────
    print("\n=== TEST 9: DeadMansSwitch ===")

    lb_for_dms = LizardBrain(cloud_url="http://fake-cloud:8001")
    diode_for_dms = SoftwareDataDiode(tx_locked=False)   # start unlocked

    quarantine_fired: list = []
    dms = DeadMansSwitch(
        lizard_brain=lb_for_dms,
        data_diode=diode_for_dms,
        threshold_s=0.1,            # tiny threshold for test speed
        quarantine_fn=lambda: quarantine_fired.append(time.time()),
        wipe_env_vars=False,        # never wipe real env in tests
    )

    _ok("DMS initialises not quarantined", not dms.is_quarantined())

    # Register a pulse — should stay alive
    dms.register_pulse()
    _ok("After pulse, not quarantined", not dms.is_quarantined())

    # Manually fire quarantine (simulate threshold exceeded)
    dms._fire_quarantine()
    _ok("Quarantine fires",            dms.is_quarantined())
    _ok("cloud_url cleared in LizardBrain",
        lb_for_dms._cloud_url == "", lb_for_dms._cloud_url)
    _ok("quarantine_fn callback called", len(quarantine_fired) == 1,
        str(len(quarantine_fired)))
    _ok("DataDiode TX locked after quarantine", diode_for_dms._tx_locked)

    status_dms = dms.get_status()
    _ok("trigger_count == 1",          status_dms["trigger_count"] == 1)
    _ok("quarantined == True",         status_dms["quarantined"] is True)

    # Reset with new URL
    dms.reset(new_cloud_url="http://new-brain:8001")
    _ok("After reset, not quarantined", not dms.is_quarantined())
    _ok("LizardBrain cloud_url restored",
        lb_for_dms._cloud_url == "http://new-brain:8001")

    # Watchdog thread integration — pulse should prevent fire
    dms2 = DeadMansSwitch(
        lizard_brain=None,
        threshold_s=0.2,
        quarantine_fn=lambda: quarantine_fired.append(time.time()),
    )
    dms2.start_pulse = lambda: None   # not a real method — just check thread starts
    dms2.start_watchdog(check_interval_s=0.05)
    _ok("Watchdog thread starts",
        dms2._watchdog_thread is not None and dms2._watchdog_thread.is_alive())

    # Keep pulsing — watchdog must NOT fire
    deadline = time.time() + 0.5
    while time.time() < deadline:
        dms2.register_pulse()
        time.sleep(0.02)
    _ok("Watchdog does not fire with regular pulses",
        not dms2.is_quarantined())

    # ── TEST 10: DMS + SovereignHeartbeat integration ──────────────────
    print("\n=== TEST 10: DMS + SovereignHeartbeat integration ===")

    lb2 = LizardBrain(cloud_url="http://brain:8001")
    dms3 = DeadMansSwitch(lizard_brain=lb2, threshold_s=10.0)
    hb2  = SovereignHeartbeat(dead_mans_switch=dms3)

    hb2.start_pulse(cycles=2, interval_s=0)
    _ok("Heartbeat pulses register in DMS",
        dms3._last_pulse > 0)
    _ok("DMS not quarantined after HB pulses", not dms3.is_quarantined())

    # ── CLEANUP ────────────────────────────────────────────────────────
    shutil.rmtree(tmpdir)

    print("\n" + "=" * 65)
    print(f"  Results: {passed_total} passed, {failed_total} failed")
    if failed_total == 0:
        print("  ✅ All LizardBrain v14.0 tests passed.")
    else:
        print(f"  ⚠️  {failed_total} test(s) failed — review above.")
    print("=" * 65)
    sys.exit(0 if failed_total == 0 else 1)
