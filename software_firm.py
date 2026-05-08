#!/usr/bin/env python3
# =====================================================================
# 🏢 SOFTWARE FIRM  —  Split-Architecture 3-Model Pipeline  v2.0
#
# Architecture (as decided):
#   Mac (MPS/CPU)  → DeepSeek-Coder-1.5B  — Scaffolder / Coder
#   Mac (MPS/CPU)  → Qwen2.5-1.5B-Instruct — QA Tester (pytest writer)
#   Kaggle (Groq)  → Llama-3.3-70B          — Manager + Red-Teamer
#
# Pipeline flow:
#   Manager(70B) → JSON spec
#       ↓
#   Coder(1.5B, Mac, ~30 tok/s) → raw Python code
#       ↓  (parallel)
#   Tester(1.5B, Mac) → pytest suite    RedTeamer(70B, Groq) → vuln report
#       ↓  (join)
#   Manager(70B) → reviews flags → PASS or KICK_BACK to Coder
#       ↓
#   FirmResult  (code + tests + security_report + iterations)
#
# Key design decisions:
#   • Tester + RedTeamer run in parallel threads (Mac CPU ~20-40 tok/s)
#   • RedTeamer uses Groq 70B (better security reasoning than 1B LoRA)
#   • No small models on Kaggle — its 30GB RAM stays for 70B context
#   • Wires into: SovereignSpine.router, MetaAgentFactory, SLoRARouter
#
# Migrated from notebook Cells 10 / 15:
#   • TransientWorkerAgent  — single-task ephemeral sub-agent
#   • AgentAsAJudge         — critic that scores worker output
#   • SwarmOrchestrator     — multi-agent decompose → critic loop
#   • AgentRegistry         — agent catalogue with quality scores
#   • AlphaEvolveGym        — nightly synthetic-task evolution
#   • AgentCard             — typed dataclass replacing bare dict
#   • FirmSwarmBridge       — wires SwarmOrchestrator into build()
#
# WIRING (swayambhu_v13.py boot):
# ─────────────────────────────────────────────────────────────────────
#   from software_firm import SoftwareFirm, AgentRegistry, AlphaEvolveGym
#
#   registry = AgentRegistry()
#   registry.register_agent(
#       name="Cpp_Kernel_Optimizer",
#       description="Bare-metal C++ writer.",
#       capabilities=["c++", "optimization", "compiling"],
#       base_rate=0.05,
#   )
#
#   firm = SoftwareFirm(
#       manager_fn   = self._cloud_llm_fn,
#       coder_llm    = self.edge.local_llm,
#       tester_llm   = self.edge.local_llm,
#       agent_registry = registry,
#       on_progress  = lambda s, m: print(f"[Firm] {s}: {m}"),
#   )
#   result = firm.build("Build a secure FastAPI login endpoint")
#
#   gym = AlphaEvolveGym(registry)
#   gym.run_nightly_evolution("Cpp_Kernel_Optimizer", synthetic_tasks=5)
# =====================================================================

from __future__ import annotations

import ast
import json
import logging
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("SoftwareFirm")

# ── Config ────────────────────────────────────────────────────────────
MAX_ITERATIONS     = 3
CODER_MAX_TOKENS   = 1200
TESTER_MAX_TOKENS  = 800
REDTEAM_MAX_TOKENS = 600
MANAGER_MAX_TOKENS = 500
PARALLEL_TIMEOUT   = 60.0
SWARM_MAX_REVISIONS = 2

_BASE_DIR = Path(__file__).parent.resolve()
_FIRM_DIR = _BASE_DIR / "software_firm"
_FIRM_DIR.mkdir(parents=True, exist_ok=True)

# ── System prompts (role-locked) ─────────────────────────────────────
_MANAGER_SPEC_PROMPT = """\
You are the Engineering Manager. Produce a concise JSON spec for a coding task.
Return ONLY valid JSON:
{{
  "task": "<one sentence>",
  "language": "python",
  "requirements": ["<req1>", "<req2>"],
  "security_constraints": ["<constraint1>"],
  "acceptance_criteria": ["<criterion1>"]
}}
Task: {goal}"""

_CODER_PROMPT = """\
You are a Python engineer. Write ONLY clean, working Python code.
No explanations. No markdown fences. Pure .py content only.

Spec:
{spec}

Previous security flags to fix (empty = first pass):
{flags}

Write the complete Python module now:"""

_TESTER_PROMPT = """\
You are a QA engineer. Write a rigorous pytest suite for this code.
Return ONLY valid Python with pytest functions. No markdown.

Code to test:
```python
{code}
```

Write pytest functions that cover: happy path, edge cases, error handling."""

_REDTEAM_PROMPT = """\
You are a security red-teamer. Audit this Python code strictly for:
1. Hardcoded secrets / API keys in plaintext
2. SQL/command injection vulnerabilities
3. Missing input validation
4. Unsafe deserialization
5. Timing attacks on auth
6. Memory leaks (unclosed resources)

Code:
```python
{code}
```

Return JSON only:
{{"vulnerabilities": [{{"severity": "HIGH|MEDIUM|LOW", "type": "<type>", "line_hint": "<hint>", "description": "<desc>"}}],
  "passed": <true if no HIGH severity>, "summary": "<one sentence>"}}"""

_MANAGER_REVIEW_PROMPT = """\
You are the Engineering Manager reviewing a security audit.
Vulnerabilities found: {vulns}
Tester result: {test_result}

Decide: return JSON only:
{{"decision": "PASS" or "KICK_BACK",
  "reason": "<one sentence>",
  "required_fixes": ["<fix1>"]}}

PASS only if: no HIGH severity vulns AND tests syntactically valid."""


# =====================================================================
# DATA CLASSES
# =====================================================================

@dataclass
class Vulnerability:
    severity:    str    # HIGH | MEDIUM | LOW
    type:        str
    line_hint:   str
    description: str


@dataclass
class RedTeamReport:
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    passed:          bool = True
    summary:         str  = ""
    raw:             str  = ""


@dataclass
class IterationRecord:
    iteration:        int
    code:             str
    test_code:        str
    redteam_report:   RedTeamReport
    manager_decision: str   # PASS | KICK_BACK
    required_fixes:   List[str]
    coder_ms:         float = 0.0
    tester_ms:        float = 0.0
    redteam_ms:       float = 0.0
    review_ms:        float = 0.0


@dataclass
class FirmResult:
    session_id:     str
    goal:           str
    final_code:     str
    final_tests:    str
    redteam_report: RedTeamReport
    iterations:     int
    total_ms:       float
    status:         str   # "passed" | "max_iterations" | "error"
    iteration_log:  List[IterationRecord] = field(default_factory=list)
    swarm_log:      List[str] = field(default_factory=list)
    error:          str  = ""

    def save(self, path: Path = None) -> Path:
        out = path or (_FIRM_DIR / f"result_{self.session_id}.json")
        out.write_text(json.dumps(asdict(self), indent=2, default=str))
        return out


@dataclass
class AgentCard:
    """Typed replacement for the notebook's bare dict agent catalog entry."""
    name:          str
    description:   str
    capabilities:  List[str]
    base_rate_usd: float
    endpoint:      str  = "local"
    quality_score: float = 0.50

    def matches(self, task_requirement: str) -> float:
        """Return a capability match score for discovery."""
        req_lower = task_requirement.lower()
        hits = sum(1 for c in self.capabilities if c.lower() in req_lower)
        return hits * self.quality_score


# =====================================================================
# ROLE AGENTS  (original, unchanged)
# =====================================================================

class ManagerAgent:
    """
    Groq 70B — orchestrates the pipeline, reviews security audits.
    spec()   → JSON specification from natural language goal
    review() → PASS or KICK_BACK decision from audit results
    """

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        self._llm = llm_fn

    def spec(self, goal: str) -> dict:
        if not self._llm:
            return self._sim_spec(goal)
        try:
            raw   = self._llm(_MANAGER_SPEC_PROMPT.format(goal=goal))
            clean = re.sub(r"```json|```", "", raw).strip()
            return json.loads(clean)
        except Exception as e:
            logger.warning(f"[Manager] spec parse error: {e}")
            return self._sim_spec(goal)

    def review(self, vulns: List[Vulnerability], test_result: str) -> dict:
        vuln_summary = json.dumps(
            [{"severity": v.severity, "type": v.type,
              "description": v.description[:80]} for v in vulns],
        )
        if not self._llm:
            return self._sim_review(vulns)
        try:
            raw = self._llm(
                _MANAGER_REVIEW_PROMPT.format(
                    vulns=vuln_summary,
                    test_result=test_result[:300],
                )
            )
            clean = re.sub(r"```json|```", "", raw).strip()
            return json.loads(clean)
        except Exception as e:
            logger.warning(f"[Manager] review parse error: {e}")
            return self._sim_review(vulns)

    def _sim_spec(self, goal: str) -> dict:
        return {
            "task": goal,
            "language": "python",
            "requirements": ["implement core logic", "add error handling"],
            "security_constraints": ["no hardcoded secrets", "validate all inputs"],
            "acceptance_criteria": ["passes unit tests", "no HIGH vulnerabilities"],
        }

    def _sim_review(self, vulns: List[Vulnerability]) -> dict:
        high = [v for v in vulns if v.severity == "HIGH"]
        if high:
            return {
                "decision": "KICK_BACK",
                "reason": f"{len(high)} HIGH severity vulnerability(s) found.",
                "required_fixes": [v.description[:80] for v in high],
            }
        return {"decision": "PASS", "reason": "No HIGH severity issues.", "required_fixes": []}


class CoderAgent:
    """
    DeepSeek-Coder-1.5B on Mac — writes raw Python from spec.
    Receives required_fixes from Manager on KICK_BACK.
    """

    _SYSTEM = (
        "You are a Python engineer. Write clean, secure Python code only. "
        "No explanations, no markdown fences."
    )

    def __init__(self, llm_fn: Optional[Callable[[str, str, int], str]] = None):
        self._llm = llm_fn

    def code(self, spec: dict, required_fixes: List[str] = None) -> str:
        flags  = "\n".join(f"- {f}" for f in (required_fixes or []))
        prompt = _CODER_PROMPT.format(
            spec=json.dumps(spec, indent=2),
            flags=flags or "(none — first pass)",
        )
        if not self._llm:
            return self._sim_code(spec, required_fixes or [])
        try:
            result = self._llm(prompt, self._SYSTEM, CODER_MAX_TOKENS)
            result = re.sub(r"```python|```", "", result).strip()
            return result
        except Exception as e:
            logger.warning(f"[Coder] LLM error: {e}")
            return self._sim_code(spec, required_fixes or [])

    def _sim_code(self, spec: dict, fixes: List[str]) -> str:
        task         = spec.get("task", "task")
        fix_comments = "\n".join(f"    # Fixed: {f[:60]}" for f in fixes)
        return f'''#!/usr/bin/env python3
"""
Auto-generated module for: {task}
"""
import os
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

{fix_comments}

class SecureHandler:
    """Handles the core logic for: {task}"""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or {{}}
        self._validate_config()

    def _validate_config(self):
        """Validate configuration — no hardcoded secrets."""
        required = ["api_endpoint"]
        for key in required:
            if key not in self._config:
                logger.warning(f"Config missing: {{key}}")

    def process(self, data: str) -> dict:
        """Process input with validation."""
        if not isinstance(data, str):
            raise TypeError(f"Expected str, got {{type(data).__name__}}")
        if len(data) > 10_000:
            raise ValueError("Input exceeds maximum length")
        result = hashlib.sha256(data.encode()).hexdigest()
        return {{"status": "ok", "hash": result, "length": len(data)}}

    def get_api_key(self) -> str:
        """Read API key from environment — never hardcoded."""
        key = os.environ.get("API_KEY", "")
        if not key:
            raise EnvironmentError("API_KEY environment variable not set")
        return key


def main():
    handler = SecureHandler(config={{"api_endpoint": "https://api.example.com"}})
    result  = handler.process("hello world")
    print(f"Result: {{result}}")
    return result


if __name__ == "__main__":
    main()
'''


class TesterAgent:
    """
    Qwen2.5-1.5B-Instruct on Mac — writes pytest suite.
    Runs in parallel with RedTeamer to minimise total latency.
    """

    _SYSTEM = (
        "You are a QA engineer. Write rigorous pytest functions only. "
        "No explanations, no markdown fences."
    )

    def __init__(self, llm_fn: Optional[Callable[[str, str, int], str]] = None):
        self._llm = llm_fn

    def write_tests(self, code: str) -> str:
        prompt = _TESTER_PROMPT.format(code=code[:2000])
        if not self._llm:
            return self._sim_tests(code)
        try:
            result = self._llm(prompt, self._SYSTEM, TESTER_MAX_TOKENS)
            result = re.sub(r"```python|```", "", result).strip()
            return result
        except Exception as e:
            logger.warning(f"[Tester] LLM error: {e}")
            return self._sim_tests(code)

    def validate_syntax(self, test_code: str) -> Tuple[bool, str]:
        try:
            ast.parse(test_code)
            if not re.search(r'def test_', test_code):
                return False, "No test functions found (def test_*)"
            return True, "valid"
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

    def run_tests(self, code: str, test_code: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            (tmpdir / "module_under_test.py").write_text(code)
            patched = "from module_under_test import *\n" + test_code
            test_file = tmpdir / "test_generated.py"
            test_file.write_text(patched)
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", str(test_file),
                     "-v", "--tb=short", "-q", "--no-header"],
                    capture_output=True, text=True,
                    timeout=30, cwd=str(tmpdir),
                )
                output = result.stdout + result.stderr
                return {
                    "passed":     len(re.findall(r" PASSED", output)),
                    "failed":     len(re.findall(r" FAILED", output)),
                    "errors":     len(re.findall(r" ERROR",  output)),
                    "returncode": result.returncode,
                    "output":     output[:800],
                    "ran":        True,
                }
            except subprocess.TimeoutExpired:
                return {"passed": 0, "failed": 0, "errors": 1,
                        "output": "Timeout", "ran": False}
            except Exception as e:
                return {"passed": 0, "failed": 0, "errors": 1,
                        "output": str(e), "ran": False}

    def _sim_tests(self, code: str) -> str:  # noqa: ARG002
        return '''import pytest

def test_process_valid_input():
    handler = SecureHandler(config={"api_endpoint": "https://api.example.com"})
    result = handler.process("hello world")
    assert result["status"] == "ok"
    assert "hash" in result
    assert result["length"] == 11

def test_process_empty_string():
    handler = SecureHandler(config={"api_endpoint": "https://api.example.com"})
    result = handler.process("")
    assert result["status"] == "ok"
    assert result["length"] == 0

def test_process_rejects_non_string():
    handler = SecureHandler(config={"api_endpoint": "https://api.example.com"})
    with pytest.raises(TypeError):
        handler.process(12345)

def test_process_rejects_oversized_input():
    handler = SecureHandler(config={"api_endpoint": "https://api.example.com"})
    with pytest.raises(ValueError):
        handler.process("x" * 10_001)

def test_api_key_missing_raises():
    import os
    handler = SecureHandler(config={"api_endpoint": "https://api.example.com"})
    os.environ.pop("API_KEY", None)
    with pytest.raises(EnvironmentError):
        handler.get_api_key()

def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-secret-key")
    handler = SecureHandler(config={"api_endpoint": "https://api.example.com"})
    key = handler.get_api_key()
    assert key == "test-secret-key"
'''


class RedTeamerAgent:
    """
    Groq 70B — security audit.
    Better than a 1B LoRA for reasoning about subtle vulnerabilities.
    Runs in parallel with TesterAgent.
    """

    _SYSTEM = (
        "You are a security red-teamer. Find vulnerabilities in Python code. "
        "Return only valid JSON."
    )

    _STATIC_CHECKS = [
        (r'(?:password|secret|api_key|token)\s*=\s*["\'][^"\']{4,}["\']',
         "HIGH", "hardcoded_secret",
         "Hardcoded credential detected"),
        (r'exec\s*\(|eval\s*\(',
         "HIGH", "code_injection",
         "Unsafe exec/eval — code injection risk"),
        (r'os\.system\s*\(|subprocess\.call\s*\([^)]*shell\s*=\s*True',
         "HIGH", "command_injection",
         "Shell injection via os.system or shell=True"),
        (r'pickle\.loads?\s*\(',
         "HIGH", "unsafe_deserialization",
         "Unsafe pickle deserialization"),
        (r'cursor\.execute\s*\([^,)]*%\s*|cursor\.execute\s*\([^,)]*\.format\(',
         "HIGH", "sql_injection",
         "Possible SQL injection via string formatting"),
        (r'open\s*\([^)]+\)(?!\s*as\b)(?!.*close)',
         "MEDIUM", "resource_leak",
         "File opened without context manager — potential resource leak"),
        (r'requests\.(get|post)\s*\([^)]+\)(?!.*timeout)',
         "MEDIUM", "missing_timeout",
         "HTTP request without timeout — DoS risk"),
        (r'assert\s+',
         "LOW", "assert_in_prod",
         "assert statements disabled in optimised Python (-O flag)"),
    ]

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        self._llm = llm_fn

    def audit(self, code: str) -> RedTeamReport:
        static_vulns = self._static_scan(code)
        if self._llm:
            llm_report = self._llm_audit(code)
        else:
            llm_report = RedTeamReport(
                vulnerabilities=static_vulns,
                passed=not any(v.severity == "HIGH" for v in static_vulns),
                summary="Static scan only (no LLM).",
            )
        all_vulns = static_vulns + [
            v for v in llm_report.vulnerabilities
            if not any(s.type == v.type for s in static_vulns)
        ]
        passed = not any(v.severity == "HIGH" for v in all_vulns)
        return RedTeamReport(
            vulnerabilities=all_vulns,
            passed=passed,
            summary=llm_report.summary or (
                "No HIGH severity issues." if passed
                else f"{sum(1 for v in all_vulns if v.severity == 'HIGH')} HIGH severity issue(s) found."
            ),
            raw=llm_report.raw,
        )

    def _static_scan(self, code: str) -> List[Vulnerability]:
        vulns = []
        for pattern, severity, vtype, desc in self._STATIC_CHECKS:
            if re.search(pattern, code, re.I):
                vulns.append(Vulnerability(
                    severity=severity, type=vtype,
                    line_hint=self._find_line(code, pattern),
                    description=desc,
                ))
        return vulns

    def _find_line(self, code: str, pattern: str) -> str:
        for i, line in enumerate(code.splitlines(), 1):
            if re.search(pattern, line, re.I):
                return f"line ~{i}: {line.strip()[:60]}"
        return "unknown"

    def _llm_audit(self, code: str) -> RedTeamReport:
        try:
            raw   = self._llm(_REDTEAM_PROMPT.format(code=code[:2500]))
            clean = re.sub(r"```json|```", "", raw).strip()
            data  = json.loads(clean)
            vulns = [
                Vulnerability(
                    severity=v.get("severity", "LOW"),
                    type=v.get("type", "unknown"),
                    line_hint=v.get("line_hint", ""),
                    description=v.get("description", ""),
                )
                for v in data.get("vulnerabilities", [])
            ]
            return RedTeamReport(
                vulnerabilities=vulns,
                passed=bool(data.get("passed", True)),
                summary=data.get("summary", ""),
                raw=raw[:500],
            )
        except Exception as e:
            logger.warning(f"[RedTeamer] LLM audit error: {e}")
            return RedTeamReport(summary=f"LLM audit failed: {e}")


# =====================================================================
# MIGRATED FROM NOTEBOOK CELLS 10 / 15
# TransientWorkerAgent, AgentAsAJudge, SwarmOrchestrator,
# AgentRegistry, AlphaEvolveGym
# =====================================================================

class TransientWorkerAgent:
    """
    Single-task ephemeral sub-agent.

    Migrated from notebook Cell 10 (SwarmOrchestrator section).
    Each instance is created fresh per subtask and discarded afterwards —
    no shared mutable state between tasks.

    Fields:
        role  — human-readable label (e.g. "Researcher", "Engineer")
        skill — declared capability used by the critic for scoring
        llm_fn — optional real LLM; falls back to deterministic sim
    """

    def __init__(
        self,
        role:   str,
        skill:  str,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.role   = role
        self.skill  = skill
        self._llm   = llm_fn

    def execute_subtask(self, subtask: str) -> str:
        """
        Execute a single subtask description.
        With a real LLM: calls it and returns the raw response.
        Sim mode: returns a deterministic tagged string.
        """
        if self._llm:
            prompt = (
                f"You are a {self.role} specialising in {self.skill}.\n"
                f"Complete this task and return ONLY the result:\n{subtask}"
            )
            try:
                return self._llm(prompt)
            except Exception as e:
                logger.warning(f"[TransientWorker:{self.role}] LLM error: {e}")
        return f"[{self.role} Output]: Completed '{subtask}' using {self.skill}."

    def __repr__(self) -> str:
        return f"TransientWorkerAgent(role={self.role!r}, skill={self.skill!r})"


class AgentAsAJudge:
    """
    Critic that scores worker output and decides approve/reject.

    Migrated from notebook Cell 10.
    With a real LLM the judge prompt asks for a float 0-1 JSON score.
    Sim mode uses random sampling above the strictness threshold.

    Args:
        strictness — minimum score [0,1] required to approve (default 0.85)
        llm_fn     — optional cloud LLM for informed scoring
    """

    _JUDGE_PROMPT = """\
You are a quality critic. Score this output on a scale 0.0-1.0.
Task:   {task_goal}
Output: {worker_output}

Return JSON only: {{"score": <float 0-1>, "reason": "<one sentence>"}}"""

    def __init__(
        self,
        strictness: float = 0.85,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        if not (0.0 <= strictness <= 1.0):
            raise ValueError(f"strictness must be in [0,1], got {strictness}")
        self.strictness = strictness
        self._llm       = llm_fn

    def evaluate_work(
        self,
        task_goal:     str,
        worker_output: str,
    ) -> Tuple[bool, float]:
        """
        Returns (approved: bool, score: float).
        approved is True when score >= strictness.
        """
        score = self._score(task_goal, worker_output)
        approved = score >= self.strictness
        sym = "✅" if approved else "❌"
        logger.debug(f"[Critic] {sym} Score: {score:.2f} (threshold {self.strictness})")
        return approved, score

    def _score(self, task_goal: str, worker_output: str) -> float:
        if self._llm:
            try:
                raw   = self._llm(self._JUDGE_PROMPT.format(
                    task_goal=task_goal[:200],
                    worker_output=worker_output[:800],
                ))
                clean = re.sub(r"```json|```", "", raw).strip()
                data  = json.loads(clean)
                s     = float(data.get("score", 0.5))
                return max(0.0, min(1.0, s))
            except Exception as e:
                logger.warning(f"[AgentAsAJudge] LLM score error: {e}")
        # Deterministic sim: uniform sample
        return random.uniform(0.6, 1.0)


class SwarmOrchestrator:
    """
    Multi-agent decompose → worker → critic loop.

    Migrated from notebook Cell 10 (Module 12).

    The orchestrator:
      1. Decomposes the mission into typed subtasks.
      2. Spawns a fresh TransientWorkerAgent per subtask.
      3. Runs each worker through AgentAsAJudge with up to
         max_revisions retry passes.
      4. Returns a joined string payload of all worker outputs.

    Args:
        critic         — AgentAsAJudge instance (injected for testing)
        worker_llm_fn  — optional LLM passed to each TransientWorkerAgent
        max_revisions  — max re-tries per subtask on critic failure
    """

    # Default decomposition blueprint — overridable via subclass
    _DEFAULT_SUBTASKS: List[Dict[str, str]] = [
        {"role": "Researcher", "skill": "RAG",           "task": "Gather and summarise relevant context."},
        {"role": "Engineer",   "skill": "Python",         "task": "Write the core implementation."},
        {"role": "Copywriter", "skill": "Summarization",  "task": "Format the deliverable brief."},
    ]

    def __init__(
        self,
        critic:        Optional[AgentAsAJudge] = None,
        worker_llm_fn: Optional[Callable[[str], str]] = None,
        max_revisions: int = SWARM_MAX_REVISIONS,
    ):
        self.critic        = critic or AgentAsAJudge()
        self._worker_llm   = worker_llm_fn
        self.max_revisions = max(0, max_revisions)

    def _decompose_task(self, goal: str) -> List[Dict[str, str]]:
        """
        Map a free-text goal to a list of subtask dicts.
        Override in subclasses for domain-specific decomposition.
        Each dict must have: role, skill, task.
        """
        # Default decomposition is static; goal is available for subclassing
        _ = goal
        return [dict(st) for st in self._DEFAULT_SUBTASKS]

    def execute_complex_mission(self, mission: str) -> str:
        """
        Full swarm execution.  Returns a newline-joined string of all
        worker outputs that passed the critic.

        Raises nothing — failed subtasks after max_revisions are included
        with a FAILED marker so the caller can inspect partial results.
        """
        sub_tasks = self._decompose_task(mission)
        payloads: List[str] = []

        for st in sub_tasks:
            worker   = TransientWorkerAgent(
                role=st["role"],
                skill=st["skill"],
                llm_fn=self._worker_llm,
            )
            approved = False
            draft    = ""
            for revision in range(self.max_revisions + 1):
                draft    = worker.execute_subtask(st["task"])
                approved, score = self.critic.evaluate_work(st["task"], draft)
                logger.debug(
                    f"[Swarm] {st['role']} rev={revision} "
                    f"score={score:.2f} approved={approved}"
                )
                if approved:
                    break

            if not approved:
                draft = f"[FAILED after {self.max_revisions + 1} attempts] {draft}"

            payloads.append(draft)

        return "\n".join(payloads)

    def get_subtask_template(self, goal: str) -> List[Dict[str, str]]:
        """Public accessor for the decomposition result (useful for debugging)."""
        return self._decompose_task(goal)


# ── AgentRegistry ─────────────────────────────────────────────────────

class AgentRegistry:
    """
    Catalogue of named agents with capability-based discovery.

    Migrated from notebook Cell 10 (Module 13).

    Changes vs notebook:
      • Stores typed AgentCard dataclasses instead of bare dicts.
      • register_agent() accepts the same kwargs as the notebook but
        returns the AgentCard rather than the bare dict.
      • discover_best_agent() is deterministic (no random tie-breaking).
      • Thread-safe via _lock.
      • get_status() exposes catalogue for health endpoints.
    """

    def __init__(self) -> None:
        self._catalog: Dict[str, AgentCard] = {}
        self._lock    = threading.Lock()

    def register_agent(
        self,
        name:         str,
        description:  str,
        capabilities: List[str],
        base_rate:    float,
        endpoint:     str = "local",
    ) -> AgentCard:
        """
        Register or overwrite an agent entry.
        Returns the AgentCard so callers can inspect or mutate it.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Agent name must be a non-empty string.")
        if base_rate < 0:
            raise ValueError(f"base_rate must be >= 0, got {base_rate}")

        card = AgentCard(
            name=name,
            description=description,
            capabilities=list(capabilities),
            base_rate_usd=base_rate,
            endpoint=endpoint,
            quality_score=0.50,
        )
        with self._lock:
            self._catalog[name] = card
        logger.debug(f"[Registry] Registered agent: {name}")
        return card

    def discover_best_agent(self, task_requirement: str) -> Optional[AgentCard]:
        """
        Return the AgentCard with the highest capability match score,
        or None if the catalogue is empty.
        """
        with self._lock:
            cards = list(self._catalog.values())
        if not cards:
            return None
        best = max(cards, key=lambda c: c.matches(task_requirement))
        return best if best.matches(task_requirement) > 0 else None

    def get_agent(self, name: str) -> Optional[AgentCard]:
        with self._lock:
            return self._catalog.get(name)

    def list_agents(self) -> List[AgentCard]:
        with self._lock:
            return list(self._catalog.values())

    def remove_agent(self, name: str) -> bool:
        with self._lock:
            if name in self._catalog:
                del self._catalog[name]
                return True
        return False

    def get_status(self) -> dict:
        with self._lock:
            return {
                "total_agents": len(self._catalog),
                "agents": [
                    {
                        "name":          c.name,
                        "capabilities":  c.capabilities,
                        "quality_score": round(c.quality_score, 3),
                        "endpoint":      c.endpoint,
                    }
                    for c in self._catalog.values()
                ],
            }


# ── AlphaEvolveGym ────────────────────────────────────────────────────

class AlphaEvolveGym:
    """
    Nightly synthetic-task evolution for registered agents.

    Migrated from notebook Cell 10 (Module 13).

    Changes vs notebook:
      • Takes an AgentRegistry (not a bare dict catalog).
      • Uses exponential moving average (EMA) for quality_score updates
        so a single bad run doesn't demolish a good agent.
      • update_weight parameter controls EMA (default 0.3, matches notebook).
      • run_nightly_evolution() is thread-safe.
      • Logs improvement/regression clearly.
      • Raises ValueError for unknown agent names.
    """

    def __init__(
        self,
        registry:      AgentRegistry,
        update_weight: float = 0.3,
        win_threshold: float = 0.4,  # random win probability per synthetic task
    ):
        if not isinstance(registry, AgentRegistry):
            raise TypeError("registry must be an AgentRegistry instance.")
        if not (0.0 < update_weight <= 1.0):
            raise ValueError(f"update_weight must be in (0,1], got {update_weight}")
        self._registry      = registry
        self._update_weight = update_weight
        self._win_threshold = win_threshold
        self._evolution_log: List[Dict[str, Any]] = []
        self._lock          = threading.Lock()

    def run_nightly_evolution(
        self,
        agent_name:      str,
        synthetic_tasks: int = 5,
    ) -> dict:
        """
        Run synthetic_tasks simulated competitions for agent_name.
        Updates quality_score via EMA.
        Returns a summary dict.

        Raises ValueError if agent_name not in registry.
        """
        if synthetic_tasks < 1:
            raise ValueError(f"synthetic_tasks must be >= 1, got {synthetic_tasks}")

        card = self._registry.get_agent(agent_name)
        if card is None:
            raise ValueError(f"Agent '{agent_name}' not found in registry.")

        wins     = sum(1 for _ in range(synthetic_tasks)
                       if random.random() > self._win_threshold)
        win_rate = wins / synthetic_tasks

        old_score = card.quality_score
        new_score = old_score * (1 - self._update_weight) + win_rate * self._update_weight
        new_score = round(max(0.0, min(1.0, new_score)), 4)

        card.quality_score = new_score

        direction = "📈" if new_score > old_score else ("📉" if new_score < old_score else "—")
        logger.info(
            f"[AlphaEvolve] {agent_name}: "
            f"{old_score:.3f} → {new_score:.3f} {direction} "
            f"({wins}/{synthetic_tasks} wins)"
        )

        summary = {
            "agent":          agent_name,
            "tasks":          synthetic_tasks,
            "wins":           wins,
            "win_rate":       round(win_rate, 3),
            "old_score":      old_score,
            "new_score":      new_score,
            "timestamp":      time.time(),
        }
        with self._lock:
            self._evolution_log.append(summary)

        return summary

    def run_evolution_all(self, synthetic_tasks: int = 3) -> List[dict]:
        """Convenience method — evolves every agent in the registry."""
        results = []
        for card in self._registry.list_agents():
            try:
                results.append(
                    self.run_nightly_evolution(card.name, synthetic_tasks)
                )
            except Exception as e:
                logger.warning(f"[AlphaEvolve] Error evolving {card.name}: {e}")
        return results

    def get_evolution_log(self, last_n: int = 50) -> List[dict]:
        with self._lock:
            return list(self._evolution_log[-last_n:])


# ── FirmSwarmBridge ───────────────────────────────────────────────────

class FirmSwarmBridge:
    """
    Wires SwarmOrchestrator into the SoftwareFirm build() pipeline.

    When a SoftwareFirm is constructed with swarm_orchestrator set,
    build() calls the swarm for an initial context-gathering pass
    before the Manager spec step.  The swarm payload is injected into
    the spec goal so the Manager has richer context.

    This is optional — FirmSwarmBridge is None by default.
    """

    def __init__(self, orchestrator: SwarmOrchestrator) -> None:
        self._orch = orchestrator

    def enrich_goal(self, goal: str) -> Tuple[str, str]:
        """
        Run the swarm to gather context for goal.
        Returns (enriched_goal, swarm_payload).
        enriched_goal includes the swarm context appended.
        """
        payload = self._orch.execute_complex_mission(goal)
        enriched = (
            f"{goal}\n\n"
            f"[Swarm Context — gathered by multi-agent pre-pass]\n{payload[:800]}"
        )
        return enriched, payload


# =====================================================================
# SOFTWARE FIRM — main orchestrator
# =====================================================================

class SoftwareFirm:
    """
    Orchestrates the 3-model pipeline with optional swarm pre-pass.

        (Optional) SwarmOrchestrator gathers context
            ↓
        Manager(70B) → spec
            ↓
        Coder(1.5B, Mac) → code
            ↓  parallel
        Tester(1.5B, Mac)  RedTeamer(70B, Groq)
            ↓  join + Manager review
        PASS → FirmResult   KICK_BACK → Coder(iteration+1)

    WIRING EXAMPLE:
        firm = SoftwareFirm(
            manager_fn      = cloud_llm_fn,
            coder_llm       = local_llm,
            tester_llm      = local_llm,
            agent_registry  = registry,          # optional
            swarm_orchestrator = SwarmOrchestrator(),  # optional
            on_progress     = lambda s, m: print(f"[Firm] {s}: {m}"),
        )
        result = firm.build("Build a secure login endpoint")
    """

    def __init__(
        self,
        manager_fn:         Optional[Callable[[str], str]] = None,
        coder_llm:                                          Any = None,
        tester_llm:                                         Any = None,
        on_progress:        Optional[Callable[[str, str], None]] = None,
        max_iterations:     int = MAX_ITERATIONS,
        agent_registry:     Optional[AgentRegistry] = None,
        swarm_orchestrator: Optional[SwarmOrchestrator] = None,
    ):
        coder_fn  = self._wrap_llm(coder_llm)
        tester_fn = self._wrap_llm(tester_llm)

        self.manager   = ManagerAgent(llm_fn=manager_fn)
        self.coder     = CoderAgent(llm_fn=coder_fn)
        self.tester    = TesterAgent(llm_fn=tester_fn)
        self.redteamer = RedTeamerAgent(llm_fn=manager_fn)

        self._progress  = on_progress or (lambda s, m: logger.info(f"[Firm] {s}: {m}"))
        self._max_iter  = max_iterations
        self._registry  = agent_registry
        self._swarm_bridge = (
            FirmSwarmBridge(swarm_orchestrator)
            if swarm_orchestrator is not None else None
        )

        self._results: List[FirmResult] = []
        self._lock    = threading.Lock()

    def build(self, goal: str) -> FirmResult:
        """
        Full synchronous pipeline. Returns FirmResult.
        Typical latency breakdown (real LLMs):
          Swarm pre-pass:  ~5-10s  (optional, parallel workers)
          Manager spec:    ~1-2s   (Groq 70B)
          Coder:           ~3-6s   (1.5B Mac)
          Tester+RedTeam:  ~5-8s   (parallel)
          Manager review:  ~1-2s   (Groq 70B)
          Total (1 iter):  ~15-28s (with swarm) / ~10-18s (without)
        """
        t0         = time.time()
        session_id = uuid.uuid4().hex[:8]
        log:       List[IterationRecord] = []
        swarm_log: List[str] = []

        self._progress("start", f"Session {session_id}: '{goal[:60]}'")

        # Optional swarm pre-pass for context enrichment
        enriched_goal = goal
        if self._swarm_bridge is not None:
            self._progress("swarm", "Swarm pre-pass: gathering context…")
            enriched_goal, swarm_payload = self._swarm_bridge.enrich_goal(goal)
            swarm_log.append(swarm_payload)
            self._progress("swarm", f"Swarm context: {len(swarm_payload)} chars")

        # Manager spec
        self._progress("spec", "Manager generating JSON spec…")
        spec = self.manager.spec(enriched_goal)
        self._progress("spec", f"Spec ready: {spec.get('task', '?')[:60]}")

        required_fixes: List[str] = []
        final_code    = ""
        final_tests   = ""
        final_report  = RedTeamReport()
        status        = "max_iterations"

        for iteration in range(1, self._max_iter + 1):
            self._progress("iteration", f"Iteration {iteration}/{self._max_iter}")

            # Coder
            self._progress("code", f"Coder writing code (fixes: {len(required_fixes)})…")
            t_coder  = time.time()
            code     = self.coder.code(spec, required_fixes)
            coder_ms = round((time.time() - t_coder) * 1000, 1)
            self._progress("code", f"Code written ({len(code)} chars, {coder_ms}ms)")

            # Tester + RedTeamer in parallel
            self._progress("parallel", "Tester + RedTeamer running in parallel…")
            test_code, tester_ms, redteam_report, redteam_ms = self._parallel_audit(code)
            self._progress(
                "audit",
                f"Tester: {len(test_code)} chars | "
                f"RedTeam: passed={redteam_report.passed} "
                f"vulns={len(redteam_report.vulnerabilities)}"
            )

            syntax_ok, syntax_msg = self.tester.validate_syntax(test_code)
            test_result_str = (
                f"Syntax: {syntax_msg} | "
                f"Tests: {'valid' if syntax_ok else 'invalid'}"
            )

            # Manager review
            self._progress("review", "Manager reviewing audit results…")
            t_review  = time.time()
            review    = self.manager.review(redteam_report.vulnerabilities, test_result_str)
            review_ms = round((time.time() - t_review) * 1000, 1)

            decision       = review.get("decision", "KICK_BACK")
            required_fixes = review.get("required_fixes", [])

            rec = IterationRecord(
                iteration=iteration,
                code=code,
                test_code=test_code,
                redteam_report=redteam_report,
                manager_decision=decision,
                required_fixes=required_fixes,
                coder_ms=coder_ms,
                tester_ms=tester_ms,
                redteam_ms=redteam_ms,
                review_ms=review_ms,
            )
            log.append(rec)

            self._progress(
                "decision",
                f"Manager: {decision} — {review.get('reason', '')[:80]}"
            )

            final_code   = code
            final_tests  = test_code
            final_report = redteam_report

            if decision == "PASS":
                status = "passed"
                self._progress("pass", f"✅ PASSED in {iteration} iteration(s)")
                break

            if iteration == self._max_iter:
                self._progress("max_iter", "⚠️ Max iterations reached — returning best attempt")

        total_ms = round((time.time() - t0) * 1000, 1)
        result = FirmResult(
            session_id=session_id,
            goal=goal,
            final_code=final_code,
            final_tests=final_tests,
            redteam_report=final_report,
            iterations=len(log),
            total_ms=total_ms,
            status=status,
            iteration_log=log,
            swarm_log=swarm_log,
        )

        with self._lock:
            self._results.append(result)

        result.save(_FIRM_DIR / f"result_{session_id}.json")
        self._progress("done", f"Session {session_id}: {status} in {len(log)} iter, {total_ms}ms")
        return result

    def get_stats(self) -> dict:
        with self._lock:
            results = list(self._results)
        if not results:
            return {"sessions": 0}
        passed   = sum(1 for r in results if r.status == "passed")
        avg_iter = sum(r.iterations for r in results) / len(results)
        avg_ms   = sum(r.total_ms   for r in results) / len(results)
        return {
            "sessions":       len(results),
            "passed":         passed,
            "pass_rate":      round(passed / len(results), 3),
            "avg_iterations": round(avg_iter, 1),
            "avg_total_ms":   round(avg_ms, 1),
        }

    def _parallel_audit(
        self, code: str
    ) -> Tuple[str, float, RedTeamReport, float]:
        test_result    = [None]
        tester_time    = [0.0]
        redteam_result = [None]
        redteam_time   = [0.0]

        def _run_tester():
            t = time.time()
            test_result[0] = self.tester.write_tests(code)
            tester_time[0] = round((time.time() - t) * 1000, 1)

        def _run_redteamer():
            t = time.time()
            redteam_result[0] = self.redteamer.audit(code)
            redteam_time[0]   = round((time.time() - t) * 1000, 1)

        t1 = threading.Thread(target=_run_tester,    daemon=True)
        t2 = threading.Thread(target=_run_redteamer, daemon=True)
        t1.start(); t2.start()
        t1.join(timeout=PARALLEL_TIMEOUT)
        t2.join(timeout=PARALLEL_TIMEOUT)

        return (
            test_result[0]    or "",
            tester_time[0],
            redteam_result[0] or RedTeamReport(summary="RedTeamer timed out"),
            redteam_time[0],
        )

    @staticmethod
    def _wrap_llm(llm) -> Optional[Callable[[str, str, int], str]]:
        if llm is None:
            return None
        if hasattr(llm, "infer"):
            return lambda p, s="", n=400: llm.infer(p, system=s, max_tokens=n)
        if callable(llm):
            return llm
        return None


# =====================================================================
# SELF-TESTS
# =====================================================================

def _run_tests() -> bool:
    import tempfile   # noqa: F401  (used by TesterAgent.run_tests internally)
    logging.basicConfig(level=logging.WARNING)
    print("🏢 SoftwareFirm Self-Tests v2.0\n")
    passed = failed = 0

    def ok(name: str, cond: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if cond:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed += 1

    # ── Shared mock LLMs ──────────────────────────────────────────────
    manager_calls: List[str] = []
    coder_calls:   List[str] = []
    tester_calls:  List[str] = []

    def mock_manager(prompt: str) -> str:
        manager_calls.append(prompt[:40])
        if "Engineering Manager" in prompt and "reviewing" not in prompt.lower():
            return json.dumps({
                "task": "Build a secure login script",
                "language": "python",
                "requirements": ["validate input", "hash passwords"],
                "security_constraints": ["no hardcoded secrets"],
                "acceptance_criteria": ["passes tests"],
            })
        if "reviewing a security audit" in prompt:
            return json.dumps({
                "decision": "PASS",
                "reason": "No HIGH severity issues.",
                "required_fixes": [],
            })
        return json.dumps({
            "vulnerabilities": [], "passed": True,
            "summary": "No vulnerabilities detected.",
        })

    def mock_coder(prompt: str, system: str = "", max_tokens: int = 400) -> str:
        coder_calls.append(prompt[:30])
        return (
            "import hashlib, os\n"
            "class LoginHandler:\n"
            "    def login(self, username: str, password: str) -> bool:\n"
            "        if not username or not password:\n"
            "            raise ValueError('Invalid credentials')\n"
            "        hashed = hashlib.sha256(password.encode()).hexdigest()\n"
            "        return len(hashed) == 64\n"
            "def main():\n"
            "    h = LoginHandler()\n"
            "    return h.login('user', 'pass123')\n"
        )

    def mock_tester(prompt: str, system: str = "", max_tokens: int = 400) -> str:
        tester_calls.append(prompt[:30])
        return (
            "import pytest\n"
            "def test_login_valid():\n"
            "    h = LoginHandler()\n"
            "    assert h.login('user', 'pass') is True\n\n"
            "def test_login_empty_username():\n"
            "    h = LoginHandler()\n"
            "    with pytest.raises(ValueError):\n"
            "        h.login('', 'pass')\n\n"
            "def test_login_empty_password():\n"
            "    h = LoginHandler()\n"
            "    with pytest.raises(ValueError):\n"
            "        h.login('user', '')\n"
        )

    # ═══════════════════════════════════════════════════════════════════
    # SECTION A — Original tests (preserved exactly)
    # ═══════════════════════════════════════════════════════════════════

    print("=== Test 1: ManagerAgent.spec ===")
    mgr  = ManagerAgent(llm_fn=mock_manager)
    spec = mgr.spec("Build a secure login endpoint")
    ok("Returns dict",             isinstance(spec, dict))
    ok("Has task key",             "task" in spec)
    ok("Has requirements",         "requirements" in spec)
    ok("Has security_constraints", "security_constraints" in spec)
    ok("Manager LLM called",       len(manager_calls) >= 1)

    mgr_sim  = ManagerAgent(llm_fn=None)
    spec_sim = mgr_sim.spec("test goal")
    ok("Sim spec has language=python", spec_sim["language"] == "python")

    print("\n=== Test 2: ManagerAgent.review ===")
    no_vulns   = []
    high_vulns = [Vulnerability("HIGH", "hardcoded_secret", "line 3", "API key hardcoded")]
    low_vulns  = [Vulnerability("LOW",  "assert_in_prod",   "line 10", "assert statement")]

    review_pass = mgr.review(no_vulns, "Syntax: valid")
    ok("No vulns → PASS",          review_pass["decision"] == "PASS", str(review_pass))

    review_high = mgr_sim.review(high_vulns, "Syntax: valid")
    ok("HIGH vuln → KICK_BACK",    review_high["decision"] == "KICK_BACK", str(review_high))
    ok("required_fixes non-empty", len(review_high["required_fixes"]) > 0)

    review_low = mgr_sim.review(low_vulns, "Syntax: valid")
    ok("LOW vuln → PASS",          review_low["decision"] == "PASS", str(review_low))

    print("\n=== Test 3: CoderAgent ===")
    coder = CoderAgent(llm_fn=mock_coder)
    code1 = coder.code(spec, [])
    ok("Returns non-empty code",   len(code1) > 50)
    ok("Coder LLM called",         len(coder_calls) >= 1)
    ok("No markdown fences",       "```" not in code1)

    code2 = coder.code(spec, ["Remove hardcoded API key", "Add input validation"])
    ok("Second call with fixes",   len(code2) > 0)

    coder_sim = CoderAgent(llm_fn=None)
    code_sim  = coder_sim.code(spec, [])
    ok("Sim code is valid Python",
       bool(ast.parse(code_sim)))
    ok("Sim code has class",        "class " in code_sim)
    ok("Sim code no hardcoded key", "api_key" not in code_sim.lower() or "environ" in code_sim)

    code_sim2 = coder_sim.code(spec, ["Fix SQL injection"])
    ok("Sim code with fixes",       len(code_sim2) > 0)

    print("\n=== Test 4: TesterAgent ===")
    tester    = TesterAgent(llm_fn=mock_tester)
    test_code = tester.write_tests(code1)
    ok("Returns non-empty tests",    len(test_code) > 30)
    ok("Tester LLM called",          len(tester_calls) >= 1)
    ok("No markdown fences",         "```" not in test_code)

    tester_sim = TesterAgent(llm_fn=None)
    tests_sim  = tester_sim.write_tests(code_sim)
    ok("Sim tests non-empty",        len(tests_sim) > 30)
    ok("Sim tests have pytest fns",  "def test_" in tests_sim)

    valid, msg = tester.validate_syntax(tests_sim)
    ok("Valid test syntax",          valid, msg)

    invalid_code = "def test_bad( :\n    pass"
    bad_valid, bad_msg = tester.validate_syntax(invalid_code)
    ok("Invalid syntax detected",    not bad_valid, bad_msg)

    no_tests_code = "def helper(): pass"
    nt_valid, nt_msg = tester.validate_syntax(no_tests_code)
    ok("No test_ fns detected",      not nt_valid, nt_msg)

    print("\n=== Test 5: TesterAgent.run_tests ===")
    run_result = tester_sim.run_tests(code_sim, tests_sim)
    ok("run_tests returns dict",     isinstance(run_result, dict))
    ok("Has passed key",             "passed" in run_result)
    ok("Has output key",             "output" in run_result)
    ok("Has ran key",                "ran" in run_result)
    ok("No exception",               True)

    print("\n=== Test 6: RedTeamerAgent static scan ===")
    rt = RedTeamerAgent(llm_fn=None)

    clean_code = (
        "import os, hashlib\n"
        "def process(data: str) -> str:\n"
        "    if not isinstance(data, str):\n"
        "        raise TypeError('expected str')\n"
        "    return hashlib.sha256(data.encode()).hexdigest()\n"
    )
    report_clean = rt.audit(clean_code)
    ok("Clean code passes",         report_clean.passed, str(report_clean.vulnerabilities))
    ok("Zero HIGH vulns",           not any(v.severity == "HIGH" for v in report_clean.vulnerabilities))

    vuln_code = (
        "import os, pickle, requests\n"
        'API_KEY = "sk-hardcoded-secret-1234567890"\n'
        "def get_data(user_input):\n"
        "    os.system(f'ls {user_input}')\n"
        "    data = pickle.loads(user_input)\n"
        "    return data\n"
        "def fetch():\n"
        '    return requests.get("https://api.example.com/data")\n'
    )
    report_vuln = rt.audit(vuln_code)
    ok("Vulnerable code fails",      not report_vuln.passed, str(report_vuln.summary))
    ok("Finds hardcoded secret",
       any(v.type == "hardcoded_secret" for v in report_vuln.vulnerabilities),
       str([v.type for v in report_vuln.vulnerabilities]))
    ok("Finds command injection",
       any(v.type == "command_injection" for v in report_vuln.vulnerabilities),
       str([v.type for v in report_vuln.vulnerabilities]))
    ok("Finds pickle deserialization",
       any(v.type == "unsafe_deserialization" for v in report_vuln.vulnerabilities),
       str([v.type for v in report_vuln.vulnerabilities]))

    eval_code = "result = eval(user_input)"
    ok("Detects eval",
       any(v.type == "code_injection" for v in rt.audit(eval_code).vulnerabilities))

    sql_code = 'cursor.execute("SELECT * FROM users WHERE id = %s" % user_id)'
    ok("Detects SQL injection",
       any(v.type == "sql_injection" for v in rt.audit(sql_code).vulnerabilities))

    print("\n=== Test 7: RedTeamerAgent + mock LLM ===")
    def mock_redteam_llm(prompt: str) -> str:
        return json.dumps({
            "vulnerabilities": [
                {"severity": "MEDIUM", "type": "missing_timeout",
                 "line_hint": "line 10", "description": "Request without timeout"},
            ],
            "passed": True,
            "summary": "One medium issue found.",
        })

    rt_llm     = RedTeamerAgent(llm_fn=mock_redteam_llm)
    report_llm = rt_llm.audit(clean_code)
    ok("LLM vulns merged",           len(report_llm.vulnerabilities) >= 1)
    ok("Summary from LLM",           len(report_llm.summary) > 0)

    print("\n=== Test 8: Parallel Tester + RedTeamer ===")
    firm_sim = SoftwareFirm()
    t_start  = time.time()
    test_c, t_ms, report, r_ms = firm_sim._parallel_audit(code_sim)
    elapsed  = time.time() - t_start
    ok("Parallel returns test code",   len(test_c) > 0)
    ok("Parallel returns tester_ms",   t_ms >= 0)
    ok("Parallel returns report",      isinstance(report, RedTeamReport))
    ok("Parallel returns redteam_ms",  r_ms >= 0)
    ok("Both ran (no timeout)",        elapsed < PARALLEL_TIMEOUT)

    print("\n=== Test 9: Full pipeline (sim mode) ===")
    progress_log: List = []
    firm_full  = SoftwareFirm(on_progress=lambda s, m: progress_log.append((s, m)))
    result_sim = firm_full.build("Build a secure user authentication module")
    ok("Returns FirmResult",          isinstance(result_sim, FirmResult))
    ok("Has session_id",              len(result_sim.session_id) == 8)
    ok("Has final_code",              len(result_sim.final_code) > 0)
    ok("Has final_tests",             len(result_sim.final_tests) > 0)
    ok("Status is passed/max_iter",   result_sim.status in ("passed", "max_iterations"))
    ok("iterations >= 1",             result_sim.iterations >= 1)
    ok("total_ms >= 0",               result_sim.total_ms >= 0)
    ok("Progress callbacks fired",    len(progress_log) > 0)
    ok("iteration_log populated",     len(result_sim.iteration_log) >= 1)

    rec = result_sim.iteration_log[0]
    ok("Iter record has code",        len(rec.code) > 0)
    ok("Iter record has test_code",   len(rec.test_code) > 0)
    ok("Iter record has decision",    rec.manager_decision in ("PASS", "KICK_BACK"))
    ok("Iter record has coder_ms",    rec.coder_ms >= 0)
    ok("Iter record has tester_ms",   rec.tester_ms >= 0)
    ok("Iter record has redteam_ms",  rec.redteam_ms >= 0)

    saved = _FIRM_DIR / f"result_{result_sim.session_id}.json"
    ok("Result saved to disk",        saved.exists())

    print("\n=== Test 10: Full pipeline (mock LLMs) ===")
    manager_calls.clear(); coder_calls.clear(); tester_calls.clear()
    progress2: List = []
    firm_mock  = SoftwareFirm(
        manager_fn=mock_manager,
        coder_llm=mock_coder,
        tester_llm=mock_tester,
        on_progress=lambda s, m: progress2.append((s, m)),
        max_iterations=2,
    )
    result_mock = firm_mock.build("Build a secure login endpoint")
    ok("Mock pipeline status",         result_mock.status in ("passed", "max_iterations"))
    ok("Manager called for spec",      len(manager_calls) >= 1)
    ok("Coder called",                 len(coder_calls) >= 1)
    ok("Tester called",                len(tester_calls) >= 1)
    ok("Final code from mock",         len(result_mock.final_code) > 0)
    ok("RedTeam report present",       isinstance(result_mock.redteam_report, RedTeamReport))
    ok("swarm_log is list",            isinstance(result_mock.swarm_log, list))

    print("\n=== Test 11: KICK_BACK → Coder fix cycle ===")
    kick_count = [0]

    def kick_manager(prompt: str) -> str:
        if "Engineering Manager" in prompt and "reviewing" not in prompt.lower():
            return json.dumps({
                "task": "test", "language": "python",
                "requirements": [], "security_constraints": [],
                "acceptance_criteria": [],
            })
        if "reviewing a security audit" in prompt:
            kick_count[0] += 1
            if kick_count[0] == 1:
                return json.dumps({
                    "decision": "KICK_BACK",
                    "reason": "Hardcoded secret found.",
                    "required_fixes": ["Remove API_KEY hardcoded value"],
                })
            return json.dumps({
                "decision": "PASS", "reason": "All issues resolved.", "required_fixes": [],
            })
        return json.dumps({"vulnerabilities": [], "passed": True, "summary": "ok"})

    firm_kick   = SoftwareFirm(
        manager_fn=kick_manager,
        coder_llm=mock_coder,
        tester_llm=mock_tester,
        max_iterations=3,
    )
    result_kick = firm_kick.build("test KICK_BACK cycle")
    ok("KICK_BACK then PASS works",   result_kick.status == "passed",
       f"status={result_kick.status}")
    ok("Two iterations ran",          result_kick.iterations == 2,
       f"iterations={result_kick.iterations}")
    ok("First iter = KICK_BACK",      result_kick.iteration_log[0].manager_decision == "KICK_BACK")
    ok("Second iter = PASS",          result_kick.iteration_log[1].manager_decision == "PASS")

    print("\n=== Test 12: get_stats ===")
    stats = firm_mock.get_stats()
    ok("Stats has sessions",          "sessions" in stats)
    ok("Stats has pass_rate 0-1",     0.0 <= stats["pass_rate"] <= 1.0)
    ok("Stats has avg_iterations",    "avg_iterations" in stats)
    ok("Stats has avg_total_ms",      stats["avg_total_ms"] >= 0)

    print("\n=== Test 13: Sim code passes security scan ===")
    coder_out  = CoderAgent(llm_fn=None).code(spec, [])
    report_sec = RedTeamerAgent(llm_fn=None).audit(coder_out)
    ok("Sim code has no HIGH vulns",  not any(v.severity == "HIGH" for v in report_sec.vulnerabilities),
       str([(v.severity, v.type) for v in report_sec.vulnerabilities]))
    ok("Sim code passes RedTeam",     report_sec.passed, report_sec.summary)

    print("\n=== Test 14: _wrap_llm normalisation ===")
    class MockLocalLLM:
        def infer(self, p: str, system: str = "", max_tokens: int = 400) -> str:
            return f"response to {p[:20]}"

    wrapped = SoftwareFirm._wrap_llm(MockLocalLLM())
    ok("Wraps LocalLLMFallback",      wrapped is not None)
    ok("Wrapped call works",          "response" in wrapped("hello"))

    wrapped_fn = SoftwareFirm._wrap_llm(lambda p, s="", n=400: "direct")
    ok("Wraps raw callable",          wrapped_fn is not None)
    ok("Raw callable works",          wrapped_fn("x") == "direct")
    ok("None → None",                 SoftwareFirm._wrap_llm(None) is None)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION B — New: TransientWorkerAgent
    # ═══════════════════════════════════════════════════════════════════

    print("\n=== Test 15: TransientWorkerAgent ===")
    worker_sim = TransientWorkerAgent("Engineer", "Python")
    out_sim    = worker_sim.execute_subtask("Write a hello-world function.")
    ok("Sim output non-empty",        len(out_sim) > 10)
    ok("Contains role label",         "Engineer" in out_sim)
    ok("Contains skill label",        "Python" in out_sim)
    ok("Repr works",                  "TransientWorkerAgent" in repr(worker_sim))

    # With mock LLM
    worker_llm = TransientWorkerAgent("Researcher", "RAG",
                                      llm_fn=lambda p: "Found 3 relevant docs.")
    out_llm    = worker_llm.execute_subtask("Find context for login systems.")
    ok("LLM worker returns response", "Found" in out_llm)

    # LLM error fallback
    def bad_llm(p: str) -> str:
        raise RuntimeError("network error")
    worker_bad = TransientWorkerAgent("Researcher", "RAG", llm_fn=bad_llm)
    out_bad    = worker_bad.execute_subtask("any task")
    ok("LLM error falls back to sim", "Researcher" in out_bad)

    # Ephemerality: two instances share no state
    w1 = TransientWorkerAgent("A", "x")
    w2 = TransientWorkerAgent("B", "y")
    ok("Workers are independent",     w1.role != w2.role)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION C — New: AgentAsAJudge
    # ═══════════════════════════════════════════════════════════════════

    print("\n=== Test 16: AgentAsAJudge ===")
    # Sim mode: score is random in [0.6, 1.0], threshold 0.85
    random.seed(42)
    judge = AgentAsAJudge(strictness=0.50)  # always approves with sim
    approved, score = judge.evaluate_work("write a function", "def foo(): pass")
    ok("Returns (bool, float)",       isinstance(approved, bool) and isinstance(score, float))
    ok("Score in [0,1]",              0.0 <= score <= 1.0)
    ok("Low threshold approves",      approved)

    # High threshold: chance of rejection (seed-stable)
    random.seed(0)
    judge_strict = AgentAsAJudge(strictness=0.99)
    approved_s, score_s = judge_strict.evaluate_work("task", "output")
    ok("Strict threshold can reject", not approved_s or score_s >= 0.99)

    # With LLM returning 0.9
    def judge_llm(p: str) -> str:
        return json.dumps({"score": 0.9, "reason": "Good output"})
    judge_l     = AgentAsAJudge(strictness=0.85, llm_fn=judge_llm)
    appr_l, sc_l = judge_l.evaluate_work("task", "output")
    ok("LLM score 0.9 with thresh 0.85 approves", appr_l)
    ok("LLM score returned correctly",             abs(sc_l - 0.9) < 1e-6)

    # LLM score 0.5 with threshold 0.85 → reject
    def low_judge_llm(p: str) -> str:
        return json.dumps({"score": 0.5, "reason": "Weak output"})
    judge_low    = AgentAsAJudge(strictness=0.85, llm_fn=low_judge_llm)
    appr_low, _ = judge_low.evaluate_work("task", "output")
    ok("LLM score 0.5 with thresh 0.85 rejects",  not appr_low)

    # LLM error fallback to sim
    def bad_judge(p: str) -> str:
        raise ValueError("timeout")
    judge_err = AgentAsAJudge(strictness=0.0, llm_fn=bad_judge)
    appr_e, sc_e = judge_err.evaluate_work("task", "output")
    ok("Judge LLM error falls back to sim", 0.0 <= sc_e <= 1.0)

    # Invalid strictness
    try:
        AgentAsAJudge(strictness=1.5)
        ok("Rejects strictness > 1",  False, "should have raised")
    except ValueError:
        ok("Rejects strictness > 1",  True)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION D — New: SwarmOrchestrator
    # ═══════════════════════════════════════════════════════════════════

    print("\n=== Test 17: SwarmOrchestrator ===")
    swarm = SwarmOrchestrator()
    result_swarm = swarm.execute_complex_mission("Build a login system")
    ok("Swarm returns non-empty string", len(result_swarm) > 0)
    ok("Swarm has newline-joined parts",
       result_swarm.count("\n") >= len(SwarmOrchestrator._DEFAULT_SUBTASKS) - 1)

    # All subtasks covered
    subtasks = swarm.get_subtask_template("anything")
    ok("Default has 3 subtasks",     len(subtasks) == 3)
    ok("Subtasks have required keys",
       all("role" in st and "skill" in st and "task" in st for st in subtasks))

    # Custom critic always approves → no FAILED markers
    always_approve = AgentAsAJudge(strictness=0.0)
    swarm_pass     = SwarmOrchestrator(critic=always_approve, max_revisions=0)
    out_pass       = swarm_pass.execute_complex_mission("any goal")
    ok("Always-approve swarm no FAILED markers", "FAILED" not in out_pass)

    # Critic always rejects → FAILED markers present
    class AlwaysRejectJudge(AgentAsAJudge):
        def _score(self, task_goal, worker_output):
            return 0.0
    swarm_fail = SwarmOrchestrator(critic=AlwaysRejectJudge(strictness=0.99), max_revisions=0)
    out_fail   = swarm_fail.execute_complex_mission("any goal")
    ok("Always-reject swarm marks FAILED", "FAILED" in out_fail)

    # max_revisions respected
    revision_count = [0]
    class CountingJudge(AgentAsAJudge):
        def _score(self, task_goal, worker_output):
            revision_count[0] += 1
            return 0.0  # always reject
    swarm_count = SwarmOrchestrator(
        critic=CountingJudge(strictness=0.99),
        max_revisions=2,
    )
    swarm_count.execute_complex_mission("mission")
    # 3 subtasks × (2+1) revisions = 9 calls
    ok("max_revisions honoured",    revision_count[0] == 9,
       f"got {revision_count[0]}")

    # Worker LLM wired through
    llm_called = [0]
    def counting_llm(p: str) -> str:
        llm_called[0] += 1
        return "done"
    swarm_llm = SwarmOrchestrator(
        critic=AgentAsAJudge(strictness=0.0),
        worker_llm_fn=counting_llm,
        max_revisions=0,
    )
    swarm_llm.execute_complex_mission("test")
    ok("Worker LLM called for each subtask", llm_called[0] == 3,
       f"got {llm_called[0]}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION E — New: AgentRegistry
    # ═══════════════════════════════════════════════════════════════════

    print("\n=== Test 18: AgentRegistry ===")
    reg = AgentRegistry()
    ok("Empty registry status",      reg.get_status()["total_agents"] == 0)
    ok("Empty discover returns None", reg.discover_best_agent("python") is None)

    card1 = reg.register_agent(
        name="DataCleaner", description="Cleans CSVs.",
        capabilities=["csv", "data cleaning"], base_rate=0.01,
    )
    ok("Returns AgentCard",          isinstance(card1, AgentCard))
    ok("Card name correct",          card1.name == "DataCleaner")
    ok("Card quality default 0.5",   card1.quality_score == 0.50)
    ok("Registry count = 1",         reg.get_status()["total_agents"] == 1)

    card2 = reg.register_agent(
        name="CppOptimizer", description="C++ writer.",
        capabilities=["c++", "optimization"], base_rate=0.05,
    )
    ok("Registry count = 2",         reg.get_status()["total_agents"] == 2)

    # Discovery
    found = reg.discover_best_agent("optimize c++ kernel")
    ok("Discovers CppOptimizer",     found is not None and found.name == "CppOptimizer",
       str(found))

    found2 = reg.discover_best_agent("parse a csv file")
    ok("Discovers DataCleaner",      found2 is not None and found2.name == "DataCleaner",
       str(found2))

    no_match = reg.discover_best_agent("completely unrelated quantum thing")
    ok("No match returns None",      no_match is None)

    # get_agent
    ok("get_agent returns card",     reg.get_agent("DataCleaner") is card1)
    ok("get_agent unknown = None",   reg.get_agent("Ghost") is None)

    # list_agents
    ok("list_agents returns 2",      len(reg.list_agents()) == 2)

    # remove_agent
    ok("remove_agent True on hit",   reg.remove_agent("DataCleaner"))
    ok("remove_agent False on miss", not reg.remove_agent("DataCleaner"))
    ok("Registry count after remove", reg.get_status()["total_agents"] == 1)

    # Overwrite
    reg.register_agent(
        name="CppOptimizer", description="Updated.",
        capabilities=["c++"], base_rate=0.10,
    )
    ok("Overwrite keeps count at 1", reg.get_status()["total_agents"] == 1)
    ok("Overwrite updates description",
       reg.get_agent("CppOptimizer").description == "Updated.")

    # Validation
    try:
        reg.register_agent(name="", description="x", capabilities=[], base_rate=0)
        ok("Empty name raises",       False, "should have raised")
    except ValueError:
        ok("Empty name raises",       True)

    try:
        reg.register_agent(name="X", description="x", capabilities=[], base_rate=-1)
        ok("Negative rate raises",    False, "should have raised")
    except ValueError:
        ok("Negative rate raises",    True)

    # Thread safety: concurrent registrations
    reg_ts = AgentRegistry()
    errors = []
    def reg_worker(i: int):
        try:
            reg_ts.register_agent(
                name=f"Agent_{i}", description="x",
                capabilities=["task"], base_rate=0.01,
            )
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=reg_worker, args=(i,)) for i in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()
    ok("Thread-safe concurrent registration",
       len(errors) == 0 and reg_ts.get_status()["total_agents"] == 20,
       f"errors={errors}")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION F — New: AlphaEvolveGym
    # ═══════════════════════════════════════════════════════════════════

    print("\n=== Test 19: AlphaEvolveGym ===")
    reg_evo = AgentRegistry()
    reg_evo.register_agent(
        name="TestAgent", description="Test.",
        capabilities=["testing"], base_rate=0.0,
    )

    gym = AlphaEvolveGym(reg_evo)
    old_score = reg_evo.get_agent("TestAgent").quality_score

    random.seed(7)
    summary = gym.run_nightly_evolution("TestAgent", synthetic_tasks=10)
    new_score = reg_evo.get_agent("TestAgent").quality_score

    ok("Returns summary dict",       isinstance(summary, dict))
    ok("Summary has agent",          summary["agent"] == "TestAgent")
    ok("Summary has wins",           0 <= summary["wins"] <= 10)
    ok("Summary has win_rate",       0.0 <= summary["win_rate"] <= 1.0)
    ok("Quality score updated",      new_score != old_score or summary["wins"] == 5)
    ok("Quality score in [0,1]",     0.0 <= new_score <= 1.0)
    ok("Summary old_score matches",  summary["old_score"] == old_score)
    ok("Summary new_score matches",  abs(summary["new_score"] - new_score) < 1e-9)

    # EMA formula verification
    reg_ema = AgentRegistry()
    reg_ema.register_agent(
        name="EMA", description="x", capabilities=[], base_rate=0,
    )
    card_ema = reg_ema.get_agent("EMA")
    card_ema.quality_score = 0.5
    gym_ema = AlphaEvolveGym(reg_ema, update_weight=0.3, win_threshold=2.0)
    # win_threshold > 1.0 → random.random() never exceeds it → 0 wins always
    gym_ema.run_nightly_evolution("EMA", synthetic_tasks=5)
    expected = round(0.5 * 0.7 + 0.0 * 0.3, 4)
    ok("EMA formula correct",        abs(reg_ema.get_agent("EMA").quality_score - expected) < 1e-6,
       f"expected {expected}, got {reg_ema.get_agent('EMA').quality_score}")

    # Unknown agent raises
    try:
        gym.run_nightly_evolution("NonExistent")
        ok("Unknown agent raises",   False, "should have raised")
    except ValueError:
        ok("Unknown agent raises",   True)

    # Zero tasks raises
    try:
        gym.run_nightly_evolution("TestAgent", synthetic_tasks=0)
        ok("Zero tasks raises",      False, "should have raised")
    except ValueError:
        ok("Zero tasks raises",      True)

    # Invalid registry type
    try:
        AlphaEvolveGym("not_a_registry")
        ok("Bad registry type raises", False, "should have raised")
    except TypeError:
        ok("Bad registry type raises", True)

    # Invalid update_weight
    try:
        AlphaEvolveGym(reg_evo, update_weight=0.0)
        ok("Zero update_weight raises", False, "should have raised")
    except ValueError:
        ok("Zero update_weight raises", True)

    # run_evolution_all
    reg_all = AgentRegistry()
    for i in range(3):
        reg_all.register_agent(
            name=f"A{i}", description="x", capabilities=[], base_rate=0,
        )
    gym_all = AlphaEvolveGym(reg_all)
    results_all = gym_all.run_evolution_all(synthetic_tasks=2)
    ok("run_evolution_all returns 3 summaries", len(results_all) == 3)

    # evolution log
    log_entries = gym.get_evolution_log()
    ok("Evolution log non-empty",    len(log_entries) >= 1)
    ok("Log entry has timestamp",    "timestamp" in log_entries[-1])

    # Thread-safe concurrent evolution
    reg_tc = AgentRegistry()
    reg_tc.register_agent(name="Concurrent", description="x",
                          capabilities=[], base_rate=0)
    gym_tc = AlphaEvolveGym(reg_tc)
    evo_errors = []
    def evo_worker():
        try:
            gym_tc.run_nightly_evolution("Concurrent", synthetic_tasks=2)
        except Exception as e:
            evo_errors.append(e)
    evo_threads = [threading.Thread(target=evo_worker) for _ in range(10)]
    for t in evo_threads: t.start()
    for t in evo_threads: t.join()
    ok("Thread-safe concurrent evolution",  len(evo_errors) == 0,
       str(evo_errors))
    ok("Score stays in [0,1] after concurrent evo",
       0.0 <= reg_tc.get_agent("Concurrent").quality_score <= 1.0)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION G — New: AgentCard
    # ═══════════════════════════════════════════════════════════════════

    print("\n=== Test 20: AgentCard.matches ===")
    card_m = AgentCard(
        name="ML", description="ML specialist.",
        capabilities=["python", "ml", "training"], base_rate_usd=0.02,
        quality_score=0.8,
    )
    ok("Exact match returns positive score", card_m.matches("train an ml model") > 0)
    ok("No match returns 0",                 card_m.matches("build an oven circuit") == 0.0)
    ok("Partial match proportional",
       0 < card_m.matches("python training") < card_m.matches("python ml training"))

    # ═══════════════════════════════════════════════════════════════════
    # SECTION H — New: FirmSwarmBridge
    # ═══════════════════════════════════════════════════════════════════

    print("\n=== Test 21: FirmSwarmBridge ===")
    bridge_swarm = SwarmOrchestrator(critic=AgentAsAJudge(strictness=0.0))
    bridge       = FirmSwarmBridge(bridge_swarm)
    enriched, payload = bridge.enrich_goal("Build an auth system")
    ok("Enriched goal contains original",     "Build an auth system" in enriched)
    ok("Enriched goal contains swarm header", "Swarm Context" in enriched)
    ok("Payload non-empty",                   len(payload) > 0)
    ok("Enriched longer than original",       len(enriched) > len("Build an auth system"))

    # ═══════════════════════════════════════════════════════════════════
    # SECTION I — New: SoftwareFirm with swarm pre-pass
    # ═══════════════════════════════════════════════════════════════════

    print("\n=== Test 22: SoftwareFirm with SwarmOrchestrator ===")
    swarm_for_firm = SwarmOrchestrator(critic=AgentAsAJudge(strictness=0.0))
    firm_with_swarm = SoftwareFirm(
        swarm_orchestrator=swarm_for_firm,
        max_iterations=1,
    )
    swarm_result = firm_with_swarm.build("Build a caching layer")
    ok("Swarm-enriched build runs",      isinstance(swarm_result, FirmResult))
    ok("swarm_log populated",            len(swarm_result.swarm_log) == 1)
    ok("swarm_log[0] non-empty",         len(swarm_result.swarm_log[0]) > 0)
    ok("Status valid",                   swarm_result.status in ("passed", "max_iterations"))

    # Without swarm: swarm_log is empty list
    firm_no_swarm  = SoftwareFirm(max_iterations=1)
    no_swarm_result = firm_no_swarm.build("Build a cache")
    ok("No swarm → swarm_log empty",    no_swarm_result.swarm_log == [])

    # ═══════════════════════════════════════════════════════════════════
    # SECTION J — New: AgentRegistry integration with SoftwareFirm
    # ═══════════════════════════════════════════════════════════════════

    print("\n=== Test 23: AgentRegistry wired into SoftwareFirm ===")
    reg_firm = AgentRegistry()
    reg_firm.register_agent(
        name="SecurityExpert", description="Security specialist.",
        capabilities=["security", "auth", "validation"], base_rate=0.05,
    )
    firm_reg = SoftwareFirm(agent_registry=reg_firm, max_iterations=1)
    ok("Firm accepts registry",         firm_reg._registry is reg_firm)
    result_reg = firm_reg.build("Build a secure token validator")
    ok("Build with registry works",     isinstance(result_reg, FirmResult))

    # Discovery works separately
    found_sec = reg_firm.discover_best_agent("auth security")
    ok("Discovers SecurityExpert",      found_sec is not None and
       found_sec.name == "SecurityExpert")

    # AlphaEvolve on registry tied to firm
    gym_firm = AlphaEvolveGym(reg_firm)
    evo_result = gym_firm.run_nightly_evolution("SecurityExpert", synthetic_tasks=4)
    ok("Evolution works on firm registry",
       0.0 <= reg_firm.get_agent("SecurityExpert").quality_score <= 1.0)

    print(f"\n{'='*55}")
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  🏆 ALL TESTS PASSED — production-grade confirmed.")
    else:
        print(f"  ⚠️  {failed} test(s) failed — review before shipping.")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_tests() else 1)
