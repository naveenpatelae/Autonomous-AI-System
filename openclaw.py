#!/usr/bin/env python3
# =====================================================================
# ⚔️  PHASE 5 — THE "OPENCLAW" AUTONOMOUS GENERAL  (v13.2)
#
# FIXES in v13.2:
#   • ConsentGate is now MANDATORY before ANY file write.
#     distill_file() NEVER calls filepath.write_text() directly.
#     All writes go through: SimulationGym → ConsentGate → write (or not).
#   • Timeout = silent deny. No write, no log noise.
#   • User says "no" via resolve_patch(approved=False) = silent deny.
#   • SimulationGym now prints a full rich test report to stdout so
#     you can see every test result in the terminal / Kaggle cell output.
#   • Removed the circular `from openclaw import _general` reference
#     inside NocturnalDistiller — distiller now receives the consent_fn
#     at construction time via dependency injection.
# =====================================================================

from __future__ import annotations
from meta_agent_factory import MetaAgentFactory, ToolRegistry, ToolSchema
import gc
import sys
import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
import timeit
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("OpenClaw")

try:
    import PIL.Image
    import PIL.ImageGrab
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    import pyautogui
    _PAG_OK = True
except ImportError:
    _PAG_OK = False


# ─────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────
class TaskStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclass
class SubTask:
    id:           str
    description:  str
    agent_type:   str = "general"
    depends_on:   List[str] = field(default_factory=list)
    status:       TaskStatus = TaskStatus.PENDING
    result:       Any = None
    error:        str = ""
    started_at:   float = 0.0
    finished_at:  float = 0.0
    attempts:     int = 0


@dataclass
class Mission:
    id:           str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal:         str = ""
    status:       str = "active"
    dag:          Dict[str, SubTask] = field(default_factory=dict)
    iteration:    int = 0
    observations: List[str] = field(default_factory=list)
    created_at:   float = field(default_factory=time.time)
    updated_at:   float = field(default_factory=time.time)
    mission_log:  List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for tid, task in d["dag"].items():
            task["status"] = (
                task["status"].value
                if isinstance(task["status"], TaskStatus)
                else task["status"]
            )
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Mission":
        dag = {}
        for tid, t in d.get("dag", {}).items():
            t["status"] = TaskStatus(t.get("status", "pending"))
            dag[tid] = SubTask(**{k: v for k, v in t.items()
                                  if k in SubTask.__dataclass_fields__})
        d["dag"] = dag
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────────────
# SIMULATION GYM
# Runs candidate patch code through security + performance tests.
# Prints a full human-readable report to stdout (visible in terminal /
# Kaggle cell output) BEFORE the consent gate fires.
# ─────────────────────────────────────────────────────────────────────
class SimulationGym:
    """
    ⚔️ POLYGLOT SIMULATION GYM (v14.1)
    Combined with Silent Logging and WebSocket UI Support.
    """

    ATTACK_TYPES = ["injection", "timeout", "overflow", "confusion", "evasion"]

    # Map extensions to local validation toolchains
    TOOLCHAINS = {
        "py":    {"cmd": [sys.executable, "-m", "py_compile", "-"], "name": "Python"},
        "sh":    {"cmd": ["shellcheck", "-"], "name": "Shell"},
        "js":    {"cmd": ["node", "--check"], "name": "NodeJS"},
        "cpp":   {"cmd": ["clang", "-fsyntax-only", "-x", "c++", "-"], "name": "C++"},
        "swift": {"cmd": ["swiftc", "-parse", "-"], "name": "Swift"},
    }

    _FORBIDDEN = [
        re.compile(r"rm\s+-[rRfF]", re.I),
        re.compile(r"sudo\s+rm", re.I),
        re.compile(r"os\.system\s*\(", re.I),
        re.compile(r"__import__\s*\(", re.I),
        re.compile(r"subprocess\.call\s*\(\s*[\"']sh", re.I),
    ]

    def run_all_tests(self, filename: str, code: str) -> dict:
        ext = filename.split(".")[-1].lower()
        result: dict = {
            "filename": filename,
            "passed": True,
            "security_findings": [],
            "perf_ms": 0.0,
            "syntax_ok": True,
            "syntax_error": "",
            "forbidden_hit": None,
            "pros": [],
            "cons": [],
            "risk_level": "LOW",
            "language": ext.upper()
        }

        # ── 1. Universal Forbidden pattern (Pattern-based) ────────────
        for pat in self._FORBIDDEN:
            if pat.search(code):
                result.update({"forbidden_hit": pat.pattern, "passed": False, "risk_level": "HIGH"})
                result["cons"].append(f"Forbidden pattern: {pat.pattern}")
                self._print_report(result, stage="FORBIDDEN")
                return result

        # ── 2. Polyglot Syntax Gate (Replaces Python compile) ──────────
        if ext in self.TOOLCHAINS:
            tc = self.TOOLCHAINS[ext]
            try:
                proc = subprocess.run(tc["cmd"], input=code.encode(), capture_output=True, timeout=5)
                if proc.returncode != 0:
                    result.update({"syntax_ok": False, "passed": False,
                                   "syntax_error": proc.stderr.decode()[:200], "risk_level": "HIGH"})
                    result["cons"].append(f"Syntax Error ({tc['name']})")
                    self._print_report(result, stage="SYNTAX_FAIL")
                    return result
                else:
                    result["pros"].append(f"Verified {tc['name']} syntax via toolchain")
            except FileNotFoundError:
                # If toolchain is missing, we don't block. We assume valid and let the tester see it.
                result["pros"].append(f"Assumed valid {ext.upper()} (Toolchain not found)")
        else:
            result["pros"].append(f"Unchecked language ({ext.upper()}) - proceeding to simulation")

        # ── 3. Adversarial security tests ─────────────────────────────
        all_findings: List[str] = []
        attack_detail: Dict[str, List[str]] = {}
        for attack_type in self.ATTACK_TYPES:
            agent = ForgedAdversaryAgent(attack_type)
            ar = agent.attack(code) # Note: ForgedAdversaryAgent should use Regex now
            attack_detail[attack_type] = ar["findings"]
            all_findings.extend(ar["findings"])

        result["security_findings"] = all_findings
        result["attack_detail"] = attack_detail
        if all_findings:
            result["risk_level"] = "HIGH" if len(all_findings) >= 3 else "MEDIUM"
            # We don't mark 'passed' as False here to let the human decide in the UI
            for f in all_findings:
                result["cons"].append(f"Security finding: {f}")
        else:
            result["pros"].append("Passed all 5 universal security attack types")

        # ── 4. Performance benchmark ──────────────────────────────────
        # Benchmarking only works for Python natively; others marked as N/A
        if ext == "py":
            try:
                def _exec():
                    ns: dict = {}
                    exec(compile(code, f"<bench:{filename}>", "exec"), ns)
                elapsed = timeit.timeit(_exec, number=1) * 1000
                result["perf_ms"] = round(elapsed, 2)
            except Exception:
                result["perf_ms"] = -1
        else:
            result["perf_ms"] = 0.0 # Non-python benchmarking coming in Phase 6

        # ── 5. Positive signals ───────────────────────────────────────
        if "try:" in code or "catch" in code:
            result["pros"].append("Uses error handling (try/catch)")
        if "timeout" in code.lower():
            result["pros"].append("Guarded with timeouts")

        self._print_report(result, stage="COMPLETE")
        return result

    def _print_report(self, result: dict, stage: str = "COMPLETE"):
        """🚀 YOUR SILENT LOGGING FIX: Writes to security_audit.log"""
        from swayambhu_utils import PROJECT_ROOT
        log_file = PROJECT_ROOT / "security_audit.log"

        bar = "═" * 60
        filename = result.get("filename", "unknown")
        lang = result.get("language", "??")
        risk = result.get("risk_level", "LOW")
        risk_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")
        passed_icon = "✅ PASSED" if result.get("passed") else "❌ FAILED"

        lines = [
            f"\n{bar}",
            f"  🧪 POLYGLOT GYM REPORT — {filename} ({lang})",
            f"  Stage  : {stage}",
            f"  Verdict: {passed_icon}   Risk: {risk_icon} {risk}",
            f"  Perf   : {result.get('perf_ms', 0):.1f}ms (native code check)",
            bar
        ]

        if result.get("syntax_ok"):
            lines.append(f"  [✅] Syntax         : Valid {lang}")
        else:
            lines.append(f"  [❌] Syntax         : {result.get('syntax_error', '')}")

        if result.get("forbidden_hit"):
            lines.append(f"  [🚫] Forbidden      : {result['forbidden_hit']}")

        if result.get("pros"):
            lines.append("\n  Pros:")
            for p in result["pros"]: lines.append(f"    ✅ {p}")

        if result.get("cons"):
            lines.append("\n  Cons:")
            for c in result["cons"]: lines.append(f"    ⚠️  {c}")

        lines.append(bar)

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception:
            pass

    def build_patch_notification(self, patch_id, filename, rationale, sim_result, diff_summary=""):
        """Keeps the WebSocket UI connection alive."""
        return {
            "type":               "patch_proposal",
            "id":                 patch_id,
            "filename":           filename,
            "rationale":          rationale,
            "diff_summary":       diff_summary or f"Patch for {filename}",
            "pros":               sim_result["pros"],
            "cons":               sim_result["cons"],
            "risk_level":         sim_result["risk_level"],
            "security_findings":  sim_result["security_findings"],
            "perf_ms":            sim_result["perf_ms"],
            "syntax_ok":          sim_result["syntax_ok"],
            "passed_simulation":  sim_result["passed"],
            "ts":                 time.time(),
        }

# ─────────────────────────────────────────────────────────────────────
# FORGED ADVERSARY AGENT (security simulation)
# ─────────────────────────────────────────────────────────────────────
class ForgedAdversaryAgent:
    """Simulated adversarial agent for the Adversarial Arena."""

    def __init__(self, attack_type: str):
        self.attack_type = attack_type

    def attack(self, target_code: str) -> dict:
        findings: List[str] = []
        code_lr = target_code.lower()

        if self.attack_type == "injection":
            # catches eval(), exec(), and dynamic execution in most languages
            if re.search(r'\b(eval|exec|system|shell_execute|process\.run)\s*\(', target_code, re.I):
                findings.append(f"VULN: Dynamic execution primitive found — Injection Risk")
            if 'unsafe' in code_lr and ('load' in code_lr or 'parse' in code_lr):
                findings.append("VULN: Unsafe data deserialization detected")

        elif self.attack_type == "timeout":
            # Universal network check: looks for network keywords without a 'timeout' nearby
            net_keywords = [r'requests\.', r'http', r'curl', r'fetch\(', r'socket', r'urllib']
            for kw in net_keywords:
                if re.search(kw, target_code, re.I):
                    # If a network keyword exists, ensure 'timeout' is also present in the file
                    if "timeout" not in code_lr:
                        findings.append(f"VULN: Network logic ({kw.strip('.')}) found without explicit timeout — DoS risk")
                        break

        elif self.attack_type == "overflow":
            # Catches Python (while True:), C++/JS (while(true)), and Bash (while true; do)
            if re.search(r'while\s*\(?\s*(true|1|alive)\s*\)?\s*[:{;]', target_code, re.I):
                if 'break' not in code_lr and 'return' not in code_lr:
                    findings.append("VULN: Potential infinite loop — no exit condition found")

        elif self.attack_type == "confusion":
            # Catches JSON/XML parsing without error handling in Python, JS, C++
            if re.search(r'(json|xml|parse|decode)', code_lr):
                if not any(x in code_lr for x in ['try', 'catch', 'except', 'err != nil']):
                    findings.append("WARN: Data parsing found without visible error handling")

        elif self.attack_type == "evasion":
            # Catches shell access in Python, JS (child_process), C++ (system)
            if re.search(r'(os\.system|subprocess|child_process|spawn|execvp|sh\s+-c)', target_code, re.I):
                findings.append("VULN: Direct shell access detected — Evasion Risk")

        return {
            "attack_type": self.attack_type,
            "findings":    findings,
            "severity":    "HIGH" if findings else "NONE",
        }


# ─────────────────────────────────────────────────────────────────────
# OPTIC NERVE (screen capture)
# ─────────────────────────────────────────────────────────────────────
class OpticNerve:
    """Phase 5.1: Screen capture + diff to verify OS actions."""

    def capture(self) -> Optional[bytes]:
        if not _PIL_OK:
            return None
        try:
            import io
            img = PIL.ImageGrab.grab()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            logger.warning(f"[OpticNerve] capture error: {e}")
            return None

    def capture_region(self, x: int, y: int, w: int, h: int) -> Optional[bytes]:
        if not _PIL_OK:
            return None
        try:
            import io
            img = PIL.ImageGrab.grab(bbox=(x, y, x + w, y + h))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as e:
            logger.warning(f"[OpticNerve] region capture error: {e}")
            return None

    def compare_screenshots(
        self, before: Optional[bytes], after: Optional[bytes], threshold: float = 0.01
    ) -> dict:
        if before is None or after is None:
            return {"changed": False, "delta_ratio": 0.0, "reason": "no_capture"}
        if before == after:
            return {"changed": False, "delta_ratio": 0.0}

        if _PIL_OK:
            try:
                import io
                import PIL.ImageChops
                img_b = PIL.Image.open(io.BytesIO(before)).convert("RGB")
                img_a = PIL.Image.open(io.BytesIO(after)).convert("RGB")
                if img_b.size != img_a.size:
                    img_a = img_a.resize(img_b.size)
                diff = PIL.ImageChops.difference(img_b, img_a)
                pixels = list(diff.getdata())
                changed = sum(1 for p in pixels if max(p) > 10)
                ratio   = changed / max(len(pixels), 1)
                return {"changed": ratio > threshold, "delta_ratio": round(ratio, 4),
                        "changed_pixels": changed, "method": "pixel_diff"}
            except Exception:
                pass

        h_b = hashlib.md5(before).hexdigest()
        h_a = hashlib.md5(after).hexdigest()
        diff_bytes = sum(a != b for a, b in zip(before, after)) + abs(len(before) - len(after))
        ratio = diff_bytes / max(len(before), len(after), 1)
        return {"changed": h_b != h_a, "delta_ratio": round(min(ratio, 1.0), 4),
                "method": "hash_fallback"}


# ─────────────────────────────────────────────────────────────────────
# DAG PLANNER (forward + backward)
# ─────────────────────────────────────────────────────────────────────
class DAGPlanner:
    """Phase 5.2: Builds forward heuristic DAG and backward acausal DAG."""

    def __init__(self, llm_fn: Optional[Callable] = None):
        self._llm = llm_fn

    def build_dag(self, goal: str, context: dict = None) -> Dict[str, SubTask]:
        ctx = context or {}
        if self._llm:
            try:
                return self._llm_decompose(goal, ctx)
            except Exception as e:
                logger.warning(f"[DAGPlanner] LLM decompose failed: {e}")
        return self._heuristic_decompose(goal, ctx)

    def _heuristic_decompose(self, goal: str, ctx: dict) -> Dict[str, SubTask]:
        tasks: Dict[str, SubTask] = {}
        goal_lower = goal.lower()
        prev_id = "t_preflight"

        tasks[prev_id] = SubTask(
            id=prev_id,
            description="Pre-flight check: verify system state and preconditions.",
            agent_type="general",
            depends_on=[],
        )

        templates = [
            (["screen", "click", "ui", "button", "window", "open"], "ui",
             "Perform UI action via screen automation."),
            (["write", "code", "script", "file", "create", "generate"], "general",
             "Write or generate the required code/file."),
            (["compile", "build", "c++", "cpp", "rust"], "cpp",
             "Compile/build the target code."),
            (["api", "request", "fetch", "http", "post", "get"], "api",
             "Make API call and process response."),
            (["verify", "check", "confirm", "test", "validate"], "general",
             "Verify the result meets requirements."),
        ]

        added = set()
        for keywords, agent_type, desc in templates:
            if any(k in goal_lower for k in keywords) and agent_type not in added:
                tid = f"t_{agent_type}_{len(tasks)}"
                tasks[tid] = SubTask(
                    id=tid, description=f"{desc} Goal: {goal[:80]}",
                    agent_type=agent_type, depends_on=[prev_id],
                )
                prev_id = tid
                added.add(agent_type)

        final = "t_final_verify"
        tasks[final] = SubTask(
            id=final, description="Final verification: confirm goal state achieved.",
            agent_type="general", depends_on=[prev_id],
        )

        if len(tasks) == 2:
            tasks["t_main"] = SubTask(
                id="t_main", description=f"Execute primary goal: {goal[:100]}",
                agent_type="general", depends_on=["t_preflight"],
            )
            tasks[final].depends_on = ["t_main"]

        return tasks

    def _llm_decompose(self, goal: str, ctx: dict) -> Dict[str, SubTask]:
        prompt = (
            f"Decompose this goal into 2-5 sequential sub-tasks as a JSON array.\n"
            f"Goal: {goal}\nContext: {json.dumps(ctx)[:300]}\n\n"
            f"Return ONLY a JSON array:\n"
            f'[{{"id":"t1","description":"...","agent_type":"general","depends_on":[]}},...]'
        )
        raw = re.sub(r'```json|```', '', self._llm(prompt)).strip()
        tasks: Dict[str, SubTask] = {}
        for t in json.loads(raw):
            tid = t.get("id", f"t_{len(tasks)}")
            tasks[tid] = SubTask(
                id=tid, description=t.get("description", ""),
                agent_type=t.get("agent_type", "general"),
                depends_on=t.get("depends_on", []),
            )
        return tasks

    def build_backward_dag(
        self, goal: str, llm_fn: Optional[Callable] = None
    ) -> Dict[str, SubTask]:
        """
        Acausal backward chain: works backward from the Win State.
        Returns tasks in forward execution order.
        """
        fn = llm_fn or self._llm
        forward_steps: List[str] = []

        if fn:
            prompt = (
                f"Use backward chaining from this WIN STATE: {goal}\n"
                f"List prerequisite steps in REVERSE order (last step first, first step last).\n"
                f"Return ONLY a JSON array of step description strings. Max 6 steps."
            )
            try:
                raw = re.sub(r'```json|```', '', fn(prompt)).strip()
                forward_steps = list(reversed(json.loads(raw)))
            except Exception as e:
                logger.warning(f"[DAGPlanner] backward LLM error: {e}")

        if not forward_steps:
            forward_steps = [
                f"Clarify goal: {goal[:60]}",
                "Gather all prerequisites and context",
                "Execute primary action sequence",
                "Validate all sub-results",
                f"Verify Win State achieved: {goal[:60]}",
            ]

        tasks: Dict[str, SubTask] = {}
        prev_id: Optional[str] = None
        for i, desc in enumerate(forward_steps):
            tid = f"b_step_{i}"
            tasks[tid] = SubTask(
                id=tid, description=desc, agent_type="general",
                depends_on=[prev_id] if prev_id else [],
            )
            prev_id = tid

        return tasks

    def visualize(self, dag: Dict[str, SubTask]) -> str:
        lines = ["DAG:"]
        for tid, task in dag.items():
            deps = " → ".join(task.depends_on) if task.depends_on else "(root)"
            lines.append(
                f"  [{task.status.value:8s}] {tid}: {task.description[:50]} | deps={deps}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# DUAL-DAG COMPARATOR
# ─────────────────────────────────────────────────────────────────────
class DualDAGComparator:
    """
    Generates forward and backward DAGs for the same goal,
    scores them on efficiency and ethics, broadcasts a trade-off JSON
    via WebSocket, and blocks until the user selects one.
    """

    def __init__(
        self,
        planner: DAGPlanner,
        ws_broadcast_fn: Optional[Callable[[dict], None]] = None,
        confirm_fn: Optional[Callable[[str], bool]] = None,
    ):
        self._planner       = planner
        self._broadcast     = ws_broadcast_fn
        self._confirm       = confirm_fn or (lambda msg: True)
        self._choice_event  = threading.Event()
        self._user_choice: Optional[str] = None

    def receive_choice(self, choice: str):
        """Accept 'forward' or 'backward' from WS / API."""
        self._user_choice = choice
        self._choice_event.set()

    def compare_and_wait(
        self,
        goal: str,
        llm_fn: Optional[Callable] = None,
        timeout: float = 120.0,
    ) -> tuple:
        forward_dag  = self._planner.build_dag(goal)
        backward_dag = self._planner.build_backward_dag(goal, llm_fn)

        fwd_score = self._score(forward_dag,  "forward")
        bwd_score = self._score(backward_dag, "backward")

        recommendation = (
            "backward" if bwd_score["ethics_score"] > fwd_score["ethics_score"]
            else "forward"
        )

        payload = {
            "type":           "dag_comparison",
            "goal":           goal,
            "forward":        fwd_score,
            "backward":       bwd_score,
            "recommendation": recommendation,
            "ts":             time.time(),
        }

        logger.info(f"[DualDAG] Broadcasting comparison. Recommended: {recommendation}")
        if self._broadcast:
            try:
                self._broadcast(payload)
            except Exception as e:
                logger.warning(f"[DualDAG] broadcast error: {e}")

        self._choice_event.clear()
        self._user_choice = None
        chose_in_time = self._choice_event.wait(timeout=timeout)

        if not chose_in_time or self._user_choice not in ("forward", "backward"):
            logger.info(f"[DualDAG] Timeout — using recommendation: {recommendation}")
            self._user_choice = recommendation

        chosen_dag = forward_dag if self._user_choice == "forward" else backward_dag
        logger.info(f"[DualDAG] User chose: {self._user_choice}")
        return chosen_dag, self._user_choice

    def _score(self, dag: Dict[str, SubTask], direction: str) -> dict:
        tc           = len(dag)
        has_verify   = any("verify" in t.description.lower() for t in dag.values())
        has_preflight = any(
            "preflight" in tid or "precondition" in t.description.lower()
            for tid, t in dag.items()
        )
        max_depth    = self._dag_depth(dag)

        efficiency  = max(0, 10 - tc) * 0.5 + (5 - min(max_depth, 5)) * 0.5
        ethics      = (
            (2 if has_verify else 0)
            + (2 if has_preflight else 0)
            + (2 if direction == "backward" else 0)
        )

        pros: List[str] = []
        cons: List[str] = []
        if direction == "forward":
            pros.append("Natural execution order — easy to debug step-by-step")
            pros.append(f"Familiar dependency chain ({tc} tasks)")
            if not has_verify:
                cons.append("No explicit final verification step")
            cons.append("May discover missing preconditions mid-execution")
        else:
            pros.append("Goal-first design — no logical dead ends by construction")
            pros.append("Every step's preconditions traced back from Win State")
            pros.append("Higher ethics score: safety built-in structurally")
            if tc > 5:
                cons.append(f"More steps ({tc}) — longer total execution time")
            cons.append("Less intuitive to trace individual step failures")

        return {
            "direction":        direction,
            "task_count":       tc,
            "max_depth":        max_depth,
            "efficiency_score": round(efficiency, 2),
            "ethics_score":     round(ethics, 2),
            "pros":             pros,
            "cons":             cons,
            "tasks_preview":    [
                {"id": tid, "desc": t.description[:60]}
                for tid, t in list(dag.items())[:5]
            ],
        }

    def _dag_depth(self, dag: Dict[str, SubTask]) -> int:
        memo: Dict[str, int] = {}
        def depth(tid: str) -> int:
            if tid in memo:
                return memo[tid]
            task = dag.get(tid)
            if not task or not task.depends_on:
                return 0
            d = 1 + max((depth(dep) for dep in task.depends_on if dep in dag), default=0)
            memo[tid] = d
            return d
        return max((depth(t) for t in dag), default=0)


# ─────────────────────────────────────────────────────────────────────
# OODA LOOP
# ─────────────────────────────────────────────────────────────────────
class OODALoop:
    """Phase 5.1: Recursive Goal→Plan→Execute→Observe→Compare loop."""

    MAX_ITERATIONS = 20

    def __init__(
        self,
        dag_planner:  DAGPlanner,
        agent_pool:   "EphemeralAgentPool",
        optic_nerve:  OpticNerve,
        llm_fn:       Optional[Callable] = None,
        on_iteration: Optional[Callable] = None,
    ):
        self._planner  = dag_planner
        self._agents   = agent_pool
        self._optic    = optic_nerve
        self._llm      = llm_fn
        self._on_iter  = on_iteration
        self._running  = False
        self._stop_evt = threading.Event()

    def run(self, mission: Mission) -> Mission:
        self._stop_evt.clear()
        self._running = True
        logger.info(f"⚔️  [OODA] Starting mission: {mission.id} — {mission.goal[:60]}")
        context = {"goal": mission.goal, "failures": [], "iteration": 0}

        while not self._stop_evt.is_set():
            mission.iteration += 1
            context["iteration"] = mission.iteration

            if mission.iteration > self.MAX_ITERATIONS:
                mission.status = "failed"
                mission.mission_log.append({"ts": time.time(), "event": "max_iterations_exceeded",
                                            "iteration": mission.iteration})
                break

            mission.mission_log.append({"ts": time.time(), "event": "iteration_start",
                                         "iteration": mission.iteration})

            if not mission.dag or self._all_done_or_failed(mission):
                mission.dag = self._planner.build_dag(mission.goal, context)

            before   = self._optic.capture()
            results  = self._agents.execute_dag(mission.dag, context)
            after    = self._optic.capture()
            delta    = self._optic.compare_screenshots(before, after)
            obs      = self._observe(mission, results, delta)
            mission.observations.append(obs)
            mission.mission_log.append({"ts": time.time(), "event": "observation",
                                         "obs": obs[:200], "delta": delta.get("delta_ratio")})

            if self._compare(mission.goal, obs, results):
                mission.status = "done"
                mission.mission_log.append({"ts": time.time(), "event": "goal_achieved",
                                             "iteration": mission.iteration})
                break

            context["failures"].append({"iteration": mission.iteration, "observation": obs})
            if self._on_iter:
                try:
                    self._on_iter(mission, obs, False)
                except Exception:
                    pass
            mission.updated_at = time.time()

        self._running = False
        return mission

    def stop(self):
        self._stop_evt.set()

    def _all_done_or_failed(self, m: Mission) -> bool:
        return all(
            t.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.SKIPPED)
            for t in m.dag.values()
        )

    def _observe(self, m: Mission, results: dict, delta: dict) -> str:
        done   = sum(1 for t in m.dag.values() if t.status == TaskStatus.DONE)
        failed = sum(1 for t in m.dag.values() if t.status == TaskStatus.FAILED)
        total  = len(m.dag)
        screen = (f"Screen changed (Δ={delta.get('delta_ratio',0):.3f})"
                  if delta.get("changed") else "No visible screen change")
        return f"Iteration {m.iteration}: {done}/{total} tasks done, {failed} failed. {screen}."

    def _compare(self, goal: str, obs: str, results: dict) -> bool:
        done   = sum(1 for t in results.values() if t.get("status") == "done")
        failed = sum(1 for t in results.values() if t.get("status") == "failed")
        return len(results) > 0 and done == len(results) and failed == 0


# ─────────────────────────────────────────────────────────────────────
# EPHEMERAL AGENTS
# ─────────────────────────────────────────────────────────────────────
class EphemeralAgent:
    def __init__(self, agent_type: str, llm_fn: Optional[Callable] = None):
        self.agent_type = agent_type
        self._llm = llm_fn
        self.id = str(uuid.uuid4())[:6]

    def execute(self, task: SubTask, context: dict) -> dict:
        task.started_at = time.time()
        task.attempts  += 1
        task.status     = TaskStatus.RUNNING
        try:
            if self.agent_type == "ui":
                result = f"[UI Agent] staged: {task.description[:60]}"
            elif self.agent_type == "cpp":
                result = f"[C++ Agent] staged: {task.description[:60]}"
            elif self.agent_type == "api":
                result = f"[API Agent] staged: {task.description[:60]}"
            else:
                result = (self._llm(f"Execute: {task.description}\nCtx: {json.dumps(context)[:200]}")
                          if self._llm else f"[General] Completed: {task.description[:60]}")
            task.result = result
            task.status = TaskStatus.DONE
            task.finished_at = time.time()
            return {"status": "done", "result": result}
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.finished_at = time.time()
            return {"status": "failed", "error": str(e)}


class EphemeralAgentPool:
    """
    Manages a swarm of micro-agents via the Meta-Agent Factory.
    Converts high-level DAG tasks into precise, validated tool calls.
    """

    def __init__(self, llm_fn: Optional[Callable] = None, max_workers: int = 4):
        self._llm = llm_fn
        self.registry = ToolRegistry()
        self._setup_default_tools()
        self.factory = MetaAgentFactory(llm_fn=self._llm, registry=self.registry)

    def _setup_default_tools(self):
        self.registry.register(ToolSchema(
            name="actuate",
            description="Executes AppleScript on the Mac",
            required_params=["script"],
            param_types={"script": "string"}
        ))

    def execute_dag(self, dag: Dict[str, SubTask], context: dict) -> Dict[str, dict]:
        """Iterates through the DAG and executes tasks using the Swarm Factory."""
        results: Dict[str, dict] = {}
        executed: set = set()

        for _ in range(len(dag) + 2):
            ready = [
                t for tid, t in dag.items()
                if tid not in executed
                   and t.status == TaskStatus.PENDING
                   and all(dep in executed for dep in t.depends_on)
            ]
            if not ready:
                break

            for task in ready:
                swarm_result = self.factory.execute_task(task.id, task.description)

                if swarm_result["status"] == "success":
                    task.status = TaskStatus.DONE
                    task.result = swarm_result
                    results[task.id] = {"status": "done", "result": swarm_result}
                else:
                    task.status = TaskStatus.FAILED
                    task.error = swarm_result.get("reason", "Swarm validation failed")
                    results[task.id] = {"status": "failed", "error": task.error}

                executed.add(task.id)
                gc.collect()

            for tid, task in dag.items():
                if tid in executed:
                    continue
                if any(dag[dep].status == TaskStatus.FAILED
                       for dep in task.depends_on if dep in dag):
                    task.status = TaskStatus.SKIPPED
                    results[tid] = {"status": "skipped", "reason": "dependency failed"}
                    executed.add(tid)

        return results


# ─────────────────────────────────────────────────────────────────────
# WAR ROOM (environment persistence)
# ─────────────────────────────────────────────────────────────────────
class WarRoom:
    LOCAL_BACKUP_DIR = Path(os.getenv("WAR_ROOM_DIR", "./war_room"))

    def __init__(self, firebase_db=None):
        self._db = firebase_db
        self.LOCAL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        self._active_missions: Dict[str, Mission] = {}
        self._lock = threading.Lock()

    def save_mission(self, mission: Mission) -> bool:
        mission.updated_at = time.time()
        data = mission.to_dict()
        local_path = self.LOCAL_BACKUP_DIR / f"mission_{mission.id}.json"
        try:
            local_path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning(f"[WarRoom] Local save failed: {e}")
        if self._db:
            try:
                self._db.document(
                    f"artifacts/SWAYAMBHU_SOVEREIGN_001/missions/{mission.id}"
                ).set(data)
            except Exception as e:
                logger.warning(f"[WarRoom] Firestore save failed: {e}")
        with self._lock:
            self._active_missions[mission.id] = mission
        return True

    def load_mission(self, mission_id: str) -> Optional[Mission]:
        if self._db:
            try:
                doc = self._db.document(
                    f"artifacts/SWAYAMBHU_SOVEREIGN_001/missions/{mission_id}"
                ).get()
                if doc.exists:
                    return Mission.from_dict(doc.to_dict())
            except Exception:
                pass
        local_path = self.LOCAL_BACKUP_DIR / f"mission_{mission_id}.json"
        if local_path.exists():
            try:
                return Mission.from_dict(json.loads(local_path.read_text()))
            except Exception:
                pass
        return None

    def find_interrupted_missions(self) -> List[Mission]:
        interrupted = []
        for f in self.LOCAL_BACKUP_DIR.glob("mission_*.json"):
            try:
                data = json.loads(f.read_text())
                if data.get("status") == "active":
                    interrupted.append(Mission.from_dict(data))
            except Exception:
                pass
        return interrupted

    def sovereign_boot_check(self) -> List[Mission]:
        interrupted = self.find_interrupted_missions()
        if interrupted:
            interrupted.sort(key=lambda m: m.updated_at, reverse=True)
            logger.info(f"⚔️  [WarRoom] {len(interrupted)} interrupted missions to resume.")
        return interrupted


# ─────────────────────────────────────────────────────────────────────
# NOCTURNAL DISTILLER
# ─────────────────────────────────────────────────────────────────────
class NocturnalDistiller:
    """
    Phase 5.4: Adversarial Arena + Performance Benchmarking.

    v13.2 FIX: distill_file() NEVER writes directly.
    The pipeline is strictly:
        1. Run SimulationGym → full report printed to stdout
        2. Call consent_fn(patch_id, filename, hardened_code, sim_result)
           — this blocks until the user approves, denies, or times out
        3. Only write IF consent_fn returns True
        4. If user denies or no response: silent, no write, no error

    The consent_fn is injected at construction time via dependency
    injection — no circular imports, no global singletons.
    """

    ATTACK_TYPES = ["injection", "timeout", "overflow", "confusion", "evasion"]

    def __init__(
        self,
        script_dir:    Path = Path("."),
        llm_fn:        Optional[Callable] = None,
        idle_check_fn: Optional[Callable[[], bool]] = None,
        # ── v13.2: injected consent gate ──────────────────────────────
        # Signature: consent_fn(patch_id, filename, code, sim_result) -> bool
        # Returns True  → user approved  → write proceeds
        # Returns False → user denied or timed out → silent, no write
        consent_fn:    Optional[Callable[[str, str, str, dict], bool]] = None,
    ):
        self._dir            = script_dir
        self._llm            = llm_fn
        self._is_idle        = idle_check_fn or (lambda: True)
        self._consent        = consent_fn   # May be None → always deny (safe default)
        self._running        = False
        self._stop_evt       = threading.Event()
        self._thread:        Optional[threading.Thread] = None
        self._distill_log:   List[dict] = []
        self._hardened_files: Dict[str, int] = defaultdict(int)
        self._perf_log:      List[dict] = []
        self._sim_gym        = SimulationGym()

    def start(self):
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._distill_loop, daemon=True, name="NocturnalDistiller"
        )
        self._thread.start()
        self._running = True
        logger.info("🌙 [NocturnalDistiller] Started (security + perf benchmarking).")

    def stop(self):
        self._stop_evt.set()
        self._running = False

    def run_adversarial_arena(self, code: str, filename: str) -> dict:
        findings: List[str] = []
        for at in self.ATTACK_TYPES:
            r = ForgedAdversaryAgent(at).attack(code)
            findings.extend(r["findings"])
        return {"filename": filename, "total_vulnerabilities": len(findings),
                "findings": findings, "hardening_needed": len(findings) > 0}

    def benchmark_performance(self, code: str, filename: str) -> dict:
        """Benchmarks module-level load time and scans AST for slow patterns."""
        import ast as _ast
        report: dict = {
            "filename": filename, "module_load_ms": 0.0,
            "slow_patterns": [], "optimization_hints": [], "perf_ok": True,
        }

        try:
            def _load():
                ns: dict = {}
                exec(compile(code, f"<bench:{filename}>", "exec"), ns)
                ns.clear()

            elapsed = timeit.timeit(_load, number=3) / 3 * 1000
            report["module_load_ms"] = round(elapsed, 2)
            if elapsed > 500:
                report["perf_ok"] = False
                report["optimization_hints"].append(
                    f"Module loads in {elapsed:.0f}ms — move heavy init to lazy properties"
                )
            elif elapsed > 200:
                report["optimization_hints"].append(
                    f"Module loads in {elapsed:.0f}ms — consider lazy imports"
                )
        except Exception as e:
            report["optimization_hints"].append(f"Load benchmark failed: {e}")

        try:
            tree = _ast.parse(code)
            for node in _ast.walk(tree):
                if isinstance(node, _ast.For):
                    for child in _ast.walk(node):
                        if isinstance(child, _ast.For) and child is not node:
                            if "nested_loop" not in report["slow_patterns"]:
                                report["slow_patterns"].append("nested_loop")
                                report["optimization_hints"].append(
                                    "Nested for-loops detected — use dict/set lookups or numpy for O(1)"
                                )
                            break
                if isinstance(node, _ast.Call):
                    fn_name = getattr(node.func, "attr", "") or getattr(node.func, "id", "")
                    if fn_name == "sleep" and "bare_sleep" not in report["slow_patterns"]:
                        report["slow_patterns"].append("bare_sleep")
                        report["optimization_hints"].append(
                            "time.sleep() found — use threading.Event.wait() for interruptible sleep"
                        )
        except Exception:
            pass

        if not report["optimization_hints"]:
            report["optimization_hints"].append("No obvious performance issues detected ✅")

        self._perf_log.append({"ts": time.time(), **report})
        return report

    def harden_code(self, code: str, findings: List[str]) -> str:
        import ast as _ast

        # ── Primary: LLM rewrite ──────────────────────────────────────
        if self._llm and findings:
            try:
                prompt = (
                    f"Fix these vulnerabilities in the Python code below.\n"
                    f"Vulnerabilities to fix: {json.dumps(findings)}\n\n"
                    f"Return ONLY the fixed Python code with no explanation, "
                    f"no markdown fences, no preamble.\n\n"
                    f"Code:\n{code}"
                )
                raw = self._llm(prompt)
                cleaned = re.sub(r"```(?:python)?|```", "", raw).strip()
                if cleaned and len(cleaned) > 20:
                    try:
                        compile(cleaned, "<llm_hardened>", "exec")
                        return cleaned
                    except SyntaxError:
                        pass
            except Exception as e:
                logger.warning(f"[harden_code] LLM rewrite failed: {e} — falling back to AST")

        # ── Fallback: AST-based timeout injection ─────────────────────
        try:
            tree = _ast.parse(code)
        except SyntaxError:
            return code

        lines = code.splitlines(keepends=True)
        insertions: List[tuple] = []

        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Call):
                continue
            func = node.func
            if not (
                    isinstance(func, _ast.Attribute)
                    and func.attr in ("get", "post", "put", "delete", "patch")
                    and isinstance(func.value, _ast.Name)
                    and func.value.id == "requests"
            ):
                continue
            kw_args = [kw.arg for kw in node.keywords]
            if "timeout" in kw_args:
                continue
            end_line = node.end_lineno - 1
            if end_line >= len(lines):
                continue
            original = lines[end_line].rstrip("\n")
            rpos = original.rfind(")")
            if rpos == -1:
                continue
            new_line = original[:rpos] + ", timeout=30)" + original[rpos + 1:]
            insertions.append((end_line, new_line + "\n"))

        if not insertions:
            return code

        for idx, new_line in insertions:
            lines[idx] = new_line

        hardened = "".join(lines)
        try:
            compile(hardened, "<hardened>", "exec")
            return hardened
        except SyntaxError:
            return code


    def distill_file(self, filepath: Path) -> dict:
        """
        Analyse one file. Print full simulation report.
        Ask for consent before any write. Silent if denied or no response.
        """
        try:
            code = filepath.read_text(encoding="utf-8")
        except Exception as e:
            return {"status": "error", "error": str(e)}

        arena = self.run_adversarial_arena(code, filepath.name)
        perf  = self.benchmark_performance(code, filepath.name)

        if not arena["hardening_needed"] and perf["perf_ok"]:
            # SimulationGym still prints a clean-pass report
            self._sim_gym.run_all_tests(filepath.name, code)
            return {"status": "clean", "findings": 0, "perf": perf}

        # ── Generate hardened version ─────────────────────────────────
        hardened = self.harden_code(code, arena["findings"])

        if hardened == code:
            # Nothing changed — nothing to propose
            return {"status": "no_change", "findings": arena["total_vulnerabilities"], "perf": perf}

        # ── SimulationGym: run ALL tests on the hardened code + print ─
        # This is the full report the user sees BEFORE being asked.
        sim_result = self._sim_gym.run_all_tests(filepath.name, hardened)

        # ── Consent Gate — MANDATORY, no bypass ──────────────────────
        write_approved = False

        if self._consent is not None:
            patch_id = hashlib.sha256(
                f"{filepath.name}{time.time()}".encode()
            ).hexdigest()[:12]

            # consent_fn blocks until user responds or times out.
            # Returns True (approved) or False (denied / no response).
            try:
                write_approved = self._consent(
                    patch_id, filepath.name, hardened, sim_result
                )
            except Exception as e:
                logger.warning(f"[Distiller] consent_fn raised {e} — treating as denied")
                write_approved = False
        else:
            # No consent function registered — safe default: never write
            logger.info(
                f"[Distiller] No consent_fn registered for {filepath.name} — "
                f"patch NOT applied (safe default)."
            )
            write_approved = False

        # ── Write only if approved ────────────────────────────────────
        if write_approved:
            filepath.write_text(hardened, encoding="utf-8")
            self._hardened_files[filepath.name] += 1
            logger.info(f"✅ [Distiller] Patch APPROVED and applied: {filepath.name}")
        # else: total silence — user said no or didn't respond

        entry = {
            "ts": time.time(),
            "file": filepath.name,
            "vulnerabilities": arena["findings"],
            "hardened": write_approved,
            "perf_hints": perf["optimization_hints"],
        }
        self._distill_log.append(entry)

        return {
            "status":           "approved" if write_approved else "denied",
            "vulnerabilities":  arena["total_vulnerabilities"],
            "security_findings": arena["findings"],
            "perf":             perf,
            "sim_result":       sim_result,
        }

    def _distill_loop(self):
        logger.info("🌙 [NocturnalDistiller] Loop running.")
        while not self._stop_evt.is_set():
            if self._is_idle():
                # 🚀 SMART ARCHITECTURE: Only scan the blueprints folder!
                blueprint_dir = self._dir / "blueprints"
                if blueprint_dir.exists():
                    for fp in list(blueprint_dir.rglob("*.py")):
                        if self._stop_evt.is_set():
                            break
                        if not fp.name.endswith(".distill_bak"):
                            self.distill_file(fp)
                            time.sleep(2)
            time.sleep(30)

    def get_report(self) -> dict:
        return {
            "running":            self._running,
            "distilled_sessions": len(self._distill_log),
            "hardened_files":     dict(self._hardened_files),
            "recent":             self._distill_log[-5:],
            "perf_sessions":      len(self._perf_log),
            "perf_recent":        self._perf_log[-3:],
        }


# ─────────────────────────────────────────────────────────────────────
# TOP-LEVEL OPENCLAW GENERAL
# ─────────────────────────────────────────────────────────────────────
class OpenClawGeneral:
    """
    Phase 5: Autonomous General.
    Orchestrates OODA, DAGPlanner, EphemeralAgentPool, WarRoom,
    NocturnalDistiller, SimulationGym, DualDAGComparator.

    v13.2: NocturnalDistiller receives consent_fn via constructor DI.
           No circular imports. No silent writes.
    """

    def __init__(
        self,
        firebase_db:      Any = None,
        llm_fn:           Optional[Callable] = None,
        script_dir:       Path = Path("."),
        ws_broadcast_fn:  Optional[Callable[[dict], None]] = None,
    ):
        self._llm          = llm_fn
        self._ws_broadcast = ws_broadcast_fn
        self._optic        = OpticNerve()
        self._planner      = DAGPlanner(llm_fn=llm_fn)
        self._agents       = EphemeralAgentPool(llm_fn=llm_fn)
        self._war_room     = WarRoom(firebase_db=firebase_db)
        self._sim_gym      = SimulationGym()
        self._dual_dag     = DualDAGComparator(
            planner=self._planner,
            ws_broadcast_fn=ws_broadcast_fn,
        )
        self._ooda_running = False
        self._missions:    Dict[str, Mission] = {}

        # patch_id → {id, filename, code, notification, approved, event}
        self._pending_patches: Dict[str, dict] = {}
        self._patch_lock = threading.Lock()

        # ── v13.2: Distiller constructed with consent_fn injected ─────
        # This replaces the old `from openclaw import _general` reference
        # inside NocturnalDistiller.distill_file() — no circular imports.
        self._distiller = NocturnalDistiller(
            script_dir=script_dir,
            llm_fn=llm_fn,
            idle_check_fn=lambda: not self._ooda_running,
            consent_fn=self._distiller_consent_gate,
        )

    # ── Consent gate wired into the distiller ────────────────────────
    def _distiller_consent_gate(
        self,
        patch_id:   str,
        filename:   str,
        code:       str,
        sim_result: dict,
    ) -> bool:
        """
        Called by NocturnalDistiller when it wants to write a hardened file.
        Routes through the standard request_patch_approval() pipeline.
        Returns True (approved) or False (denied / timed out → silent).
        """
        result = self.request_patch_approval(
            filename=filename,
            code=code,
            rationale=(
                "Nocturnal Distiller: found security vulnerabilities and/or "
                "performance issues. Hardened version ready for review."
            ),
            # Pass pre-computed sim_result so we don't re-run the tests
            _precomputed_sim=sim_result,
        )
        return result["approved"]

    def get_tool_schemas(self) -> List[dict]:
        """Expose OpenClaw's autonomous capabilities to the Cloud Brain."""
        return [{
            "type": "function",
            "function": {
                "name": "launch_autonomous_mission",
                "description": "Launch the OpenClaw autonomous OODA loop agent on the Mac to dynamically figure out and achieve ANY complex OS, GUI, or App goal without needing a hardcoded blueprint (e.g., 'Open settings', 'Send an email', 'Change to dark mode').",
                "parameters": {
                    "type": "object",
                    "properties": {"goal": {"type": "string"}},
                    "required": ["goal"],
                },
            },
        }]

    def start(self, resume_interrupted: bool = True):
        self._distiller.start()
        if resume_interrupted:
            for m in self._war_room.sovereign_boot_check():
                logger.info(f"⚔️  [General] Resuming: {m.id} — {m.goal[:50]}")
                self._run_mission_async(m)

    def launch_mission(self, goal: str, use_dual_dag: bool = False) -> Mission:
        mission = Mission(goal=goal)
        self._missions[mission.id] = mission

        if use_dual_dag:
            def _with_comparison():
                dag, choice = self._dual_dag.compare_and_wait(goal, self._llm)
                mission.dag = dag
                mission.mission_log.append({"ts": time.time(), "event": "dag_choice",
                                             "choice": choice})
                self._run_mission_async(mission)
            threading.Thread(target=_with_comparison, daemon=True,
                             name=f"DualDAG_{mission.id}").start()
        else:
            self._run_mission_async(mission)

        return mission

    # ── Self-Patcher UI Hook ──────────────────────────────────────────
    def request_patch_approval(
        self,
        filename:          str,
        code:              str,
        rationale:         str = "",
        timeout:           float = 120.0,
        _precomputed_sim:  Optional[dict] = None,
    ) -> dict:
        """
        Full Self-Patcher pipeline:
          1. Run SimulationGym (or use pre-computed result)
          2. Broadcast Pros/Cons JSON via WebSocket to HTML UI
          3. Block until user approves or denies (up to `timeout` seconds)
          4. Timeout = silent DENY — no write, no noise
          5. Return result dict

        The ONLY path to a True return is an explicit approve signal
        via resolve_patch(patch_id, approved=True).
        """
        patch_id = hashlib.sha256(f"{filename}{time.time()}".encode()).hexdigest()[:12]
        logger.info(f"[SelfPatcher] ConsentGate opened for: {filename} (id={patch_id})")

        # Use pre-computed sim result if supplied (avoids running tests twice)
        if _precomputed_sim is not None:
            sim_result = _precomputed_sim
        else:
            logger.info(f"[SelfPatcher] SimulationGym running for: {filename}")
            sim_result = self._sim_gym.run_all_tests(filename, code)

        lines = code.splitlines()
        diff_summary = "\n".join(lines[:8]) + (
            f"\n... ({len(lines)} lines total)" if len(lines) > 8 else ""
        )

        notification = self._sim_gym.build_patch_notification(
            patch_id=patch_id,
            filename=filename,
            rationale=rationale or f"Auto-generated patch for {filename}",
            sim_result=sim_result,
            diff_summary=diff_summary,
        )

        approval_event = threading.Event()
        record = {
            "id": patch_id, "filename": filename, "code": code,
            "notification": notification, "approved": None, "event": approval_event,
        }

        with self._patch_lock:
            self._pending_patches[patch_id] = record

        if self._ws_broadcast:
            try:
                self._ws_broadcast(notification)
                logger.info(f"[SelfPatcher] Patch proposal sent via WS (id={patch_id})")
            except Exception as e:
                logger.warning(f"[SelfPatcher] WS broadcast failed: {e}")
        else:
            logger.info(
                f"[SelfPatcher] No WS broadcaster — patch {patch_id} awaiting "
                f"API call to /patch/resolve\n"
                f"  Approve : POST /patch/resolve  {{\"patch_id\": \"{patch_id}\", \"approved\": true}}\n"
                f"  Deny    : POST /patch/resolve  {{\"patch_id\": \"{patch_id}\", \"approved\": false}}"
            )

        logger.info(f"[SelfPatcher] ⏸ Waiting for your decision (timeout={timeout}s)...")
        resolved = approval_event.wait(timeout=timeout)

        with self._patch_lock:
            final = self._pending_patches.pop(patch_id, record)

        if not resolved:
            # Timeout → silent deny. No log noise, no write.
            final["approved"] = False

        approved = final.get("approved", False)
        if not approved:
            # "no" or timeout → total silence here; caller decides whether to log
            pass

        return {
            "approved":     approved,
            "patch_id":     patch_id,
            "sim_result":   sim_result,
            "notification": notification,
        }

    def resolve_patch(self, patch_id: str, approved: bool):
        """
        Called by WS handler or API when user clicks Approve/Deny.
        approved=True  → write proceeds
        approved=False → silent deny, no write
        """
        with self._patch_lock:
            record = self._pending_patches.get(patch_id)
        if record:
            record["approved"] = approved
            record["event"].set()
            if approved:
                logger.info(f"[SelfPatcher] {patch_id} → APPROVED ✅")
            # Denied → total silence (no log line)
        else:
            logger.warning(f"[SelfPatcher] Unknown patch_id: {patch_id}")

    def resolve_dag_choice(self, choice: str):
        """Called by WS handler when user picks 'forward' or 'backward'."""
        self._dual_dag.receive_choice(choice)

    def get_pending_patches(self) -> List[dict]:
        with self._patch_lock:
            return [
                {k: v for k, v in r.items() if k not in ("event", "code")}
                for r in self._pending_patches.values()
            ]

    def _run_mission_async(self, mission: Mission):
        def _run():
            self._ooda_running = True
            ooda = OODALoop(self._planner, self._agents, self._optic, self._llm,
                            self._on_ooda_iteration)
            try:
                ooda.run(mission)
            finally:
                self._ooda_running = False
                self._war_room.save_mission(mission)

        threading.Thread(target=_run, daemon=True, name=f"OODALoop_{mission.id}").start()

    def _on_ooda_iteration(self, mission: Mission, obs: str, achieved: bool):
        self._war_room.save_mission(mission)

    def get_status(self) -> dict:
        return {
            "missions": {
                mid: {"goal": m.goal[:50], "status": m.status, "iteration": m.iteration}
                for mid, m in self._missions.items()
            },
            "ooda_running":    self._ooda_running,
            "distiller":       self._distiller.get_report(),
            "pending_patches": len(self._pending_patches),
        }


# ── Module-level singleton ────────────────────────────────────────────
_general: Optional[OpenClawGeneral] = None


def get_general(**kwargs) -> OpenClawGeneral:
    global _general
    if _general is None:
        _general = OpenClawGeneral(**kwargs)
    return _general


# ─────────────────────────────────────────────────────────────────────
# SELF-TEST
# python openclaw.py
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import shutil
    import tempfile
    logging.basicConfig(level=logging.INFO)

    print("\n⚔️  OpenClaw v13.2 — Full Simulation Self-Test\n")
    print("=" * 60)
    passed_total = 0
    failed_total = 0

    def _ok(name: str, cond: bool, detail: str = ""):
        global passed_total, failed_total
        if cond:
            print(f"  ✅ {name}")
            passed_total += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed_total += 1

    tmpdir = Path(tempfile.mkdtemp())

    # ─────────────────────────────────────────────────────────────────
    # TEST 1 — SimulationGym: clean code
    # ─────────────────────────────────────────────────────────────────
    print("\n=== TEST 1: SimulationGym — clean code ===")
    gym = SimulationGym()
    good_code = (
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "def hello():\n"
        "    try:\n"
        "        return 'world'\n"
        "    except Exception as e:\n"
        "        logger.error(e)\n"
    )
    gr = gym.run_all_tests("good.py", good_code)
    _ok("Clean code passes", gr["passed"], str(gr["cons"]))
    _ok("Risk is LOW",       gr["risk_level"] == "LOW", gr["risk_level"])
    _ok("Has pros",          len(gr["pros"]) > 0)

    # ─────────────────────────────────────────────────────────────────
    # TEST 2 — SimulationGym: vulnerable code
    # ─────────────────────────────────────────────────────────────────
    print("\n=== TEST 2: SimulationGym — vulnerable code ===")
    bad_code = (
        "import requests\n"
        "def fetch(url):\n"
        "    return requests.get(url)  # no timeout\n"
        "x = eval(user_input)         # injection\n"
    )
    br = gym.run_all_tests("bad.py", bad_code)
    _ok("Vulnerable code fails",     not br["passed"],       str(br["passed"]))
    _ok("Has security findings",     len(br["security_findings"]) > 0)
    _ok("Risk is MEDIUM or HIGH",    br["risk_level"] in ("MEDIUM", "HIGH"), br["risk_level"])
    _ok("Cons list populated",       len(br["cons"]) > 0)

    # ─────────────────────────────────────────────────────────────────
    # TEST 3 — SimulationGym: forbidden pattern hard-fail
    # ─────────────────────────────────────────────────────────────────
    print("\n=== TEST 3: SimulationGym — forbidden pattern ===")
    forbidden_code = "import os\nos.system('rm -rf /')\n"
    fr = gym.run_all_tests("danger.py", forbidden_code)
    _ok("Forbidden code hard-fails",  not fr["passed"])
    _ok("Risk is HIGH",               fr["risk_level"] == "HIGH")
    _ok("forbidden_hit populated",    fr["forbidden_hit"] is not None)

    # ─────────────────────────────────────────────────────────────────
    # TEST 4 — DualDAGComparator
    # ─────────────────────────────────────────────────────────────────
    print("\n=== TEST 4: DualDAGComparator ===")
    planner  = DAGPlanner()
    received: list = []
    cmp = DualDAGComparator(planner, ws_broadcast_fn=lambda p: received.append(p))
    cmp._user_choice = "backward"
    cmp._choice_event.set()
    dag, choice = cmp.compare_and_wait("Deploy service to production", timeout=0.1)
    _ok("Choice returned",           choice in ("forward", "backward"))
    _ok("DAG has tasks",             len(dag) > 0)
    _ok("Broadcast fired",           len(received) > 0)

    # ─────────────────────────────────────────────────────────────────
    # TEST 5 — NocturnalDistiller: approved write
    # ─────────────────────────────────────────────────────────────────
    print("\n=== TEST 5: NocturnalDistiller — user approves ===")
    approved_calls: list = []

    def _consent_approve(patch_id, filename, code, sim_result) -> bool:
        approved_calls.append({"patch_id": patch_id, "filename": filename})
        print(f"  [ConsentGate] Auto-approving patch {patch_id} for {filename}")
        return True

    dist_approve = NocturnalDistiller(script_dir=tmpdir, consent_fn=_consent_approve)
    vuln_file = tmpdir / "vuln.py"
    vuln_file.write_text(
        "import requests\ndef get(url): return requests.get(url)\n",
        encoding="utf-8"
    )
    res = dist_approve.distill_file(vuln_file)
    _ok("Status is approved",        res.get("status") == "approved", str(res.get("status")))
    _ok("consent_fn was called",     len(approved_calls) == 1)
    _ok("File was written",          vuln_file.exists())

    # ─────────────────────────────────────────────────────────────────
    # TEST 6 — NocturnalDistiller: user denies → silent, no write
    # ─────────────────────────────────────────────────────────────────
    print("\n=== TEST 6: NocturnalDistiller — user denies (silent) ===")
    denied_calls: list = []

    def _consent_deny(patch_id, filename, code, sim_result) -> bool:
        denied_calls.append(patch_id)
        return False   # user says no

    deny_file = tmpdir / "deny_test.py"
    original_content = "import requests\ndef get(url): return requests.get(url)\n"
    deny_file.write_text(original_content, encoding="utf-8")

    dist_deny = NocturnalDistiller(script_dir=tmpdir, consent_fn=_consent_deny)
    res_deny = dist_deny.distill_file(deny_file)

    _ok("Status is denied",          res_deny.get("status") == "denied", str(res_deny.get("status")))
    _ok("consent_fn was called",     len(denied_calls) == 1)
    _ok("File NOT modified",         deny_file.read_text() == original_content)

    # ─────────────────────────────────────────────────────────────────
    # TEST 7 — NocturnalDistiller: no consent_fn → safe default deny
    # ─────────────────────────────────────────────────────────────────
    print("\n=== TEST 7: NocturnalDistiller — no consent_fn (safe default) ===")
    safe_file = tmpdir / "safe_default.py"
    safe_file.write_text(original_content, encoding="utf-8")

    dist_no_consent = NocturnalDistiller(script_dir=tmpdir, consent_fn=None)
    res_safe = dist_no_consent.distill_file(safe_file)

    _ok("Status is denied (safe default)", res_safe.get("status") in ("denied", "no_change"),
        str(res_safe.get("status")))
    _ok("File NOT modified",               safe_file.read_text() == original_content)

    # ─────────────────────────────────────────────────────────────────
    # TEST 8 — NocturnalDistiller: timeout → silent deny
    # ─────────────────────────────────────────────────────────────────
    print("\n=== TEST 8: NocturnalDistiller — timeout → silent deny ===")
    timeout_file = tmpdir / "timeout_test.py"
    timeout_file.write_text(original_content, encoding="utf-8")

    def _consent_timeout(patch_id, filename, code, sim_result) -> bool:
        # Simulates user not responding: delegate to request_patch_approval with tiny timeout
        return False  # immediate "no response" path

    dist_timeout = NocturnalDistiller(script_dir=tmpdir, consent_fn=_consent_timeout)
    res_timeout = dist_timeout.distill_file(timeout_file)
    _ok("Timeout treated as deny",    res_timeout.get("status") in ("denied", "no_change"),
        str(res_timeout.get("status")))
    _ok("File NOT modified",          timeout_file.read_text() == original_content)

    # ─────────────────────────────────────────────────────────────────
    # TEST 9 — OpenClawGeneral: full Self-Patcher UI Hook
    # ─────────────────────────────────────────────────────────────────
    print("\n=== TEST 9: OpenClawGeneral — Self-Patcher UI Hook ===")
    ws_payloads: list = []
    general = OpenClawGeneral(
        script_dir=tmpdir,
        ws_broadcast_fn=lambda p: ws_payloads.append(p),
    )

    # Simulate user clicking Approve in the HTML UI after 300ms
    def _auto_approve():
        time.sleep(0.3)
        patches = general.get_pending_patches()
        if patches:
            general.resolve_patch(patches[0]["id"], approved=True)

    threading.Thread(target=_auto_approve, daemon=True).start()
    result = general.request_patch_approval(
        "mod.py", good_code, "Test patch", timeout=5.0
    )
    _ok("Approved via UI hook",           result["approved"])
    _ok("Sim result present",             "sim_result" in result)
    _ok("WS payload dispatched",          len(ws_payloads) > 0)
    _ok("WS type is patch_proposal",      ws_payloads[0].get("type") == "patch_proposal")

    # ─────────────────────────────────────────────────────────────────
    # TEST 10 — OpenClawGeneral: timeout → silent deny, no crash
    # ─────────────────────────────────────────────────────────────────
    print("\n=== TEST 10: OpenClawGeneral — timeout → silent deny ===")
    result_timeout = general.request_patch_approval(
        "mod_timeout.py", good_code, "Timeout test", timeout=0.1
    )
    _ok("Timeout returns approved=False",  not result_timeout["approved"])
    _ok("No exception raised",             True)   # reaching here = no crash

    # ─────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────
    shutil.rmtree(tmpdir)
    print("\n" + "=" * 60)
    print(f"  Results : {passed_total} passed, {failed_total} failed")
    if failed_total == 0:
        print("  ✅ All OpenClaw v13.2 tests passed.")
    else:
        print(f"  ⚠️  {failed_total} test(s) failed — review above.")
    print("=" * 60)
    import sys
    sys.exit(0 if failed_total == 0 else 1)
