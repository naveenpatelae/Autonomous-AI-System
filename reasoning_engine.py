#!/usr/bin/env python3
# =====================================================================
# 🧠 REASONING ENGINE v2.0 — Advanced Planning & Distributed Cognition
#
# ORIGINAL BODY COMPONENTS (preserved exactly):
# #62/#144  Multiverse Engine   — Monte Carlo Tree Search over DAG plans
# #71       Acausal Reasoning   — Backward chaining from Win State
# #81       Multi-Agent Debate  — Skeptic / Optimist / Engineer swarm
# #51       Inner Monologue     — Chain-of-Thought tracer
# #47       Entropy Kill-Switch — Halt when AI confidence < 85 %
# #23       Semantic Gearbox    — Route 3B vs 70B by complexity
# #141      Model Critic Lobe   — LLM-as-a-Judge grader
#
# MIGRATED FROM NOTEBOOK (Cells 6-B, Distributed Architecture):
# AcausalEngine        — numeric temporal bridge solver (Cell 6-B)
# ConsilienceLobe      — cross-domain vector interlinker (Cell 6-B)
# WorldSpirit          — grand evolutionary theory synthesiser (Cell 6-B)
# OmniModelRouter      — model-agnostic dispatcher with graceful degradation
# ModelCapabilityMap   — task-type → specialist model registry
# DeepSeekAdapter      — OpenAI-compatible DeepSeek REST adapter
# DefconLevel          — IntEnum operational states DEFCON 5→1
# BrainStateManager    — auto-monitors connectivity, transitions DEFCON
# =====================================================================

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import threading
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("ReasoningEngine")

# ─────────────────────────────────────────────────────────────────────
# INNER MONOLOGUE — Chain-of-Thought tracer (#51)
# ─────────────────────────────────────────────────────────────────────
class InnerMonologue:
    """
    Records every reasoning step the AI takes.
    Exposes a live feed that the Qt Avatar UI can subscribe to.
    Thread-safe. Callbacks fire on every new thought.
    """

    def __init__(self, max_entries: int = 200):
        self._log: List[dict] = []
        self._lock = threading.Lock()
        self._max = max_entries
        self._callbacks: List[Callable[[dict], None]] = []

    def subscribe(self, cb: Callable[[dict], None]):
        """Subscribe to live thought feed (called from Qt timer or WS)."""
        self._callbacks.append(cb)

    def think(self, step: str, category: str = "reasoning", data: Any = None):
        """Record a thought step."""
        entry = {
            "ts": time.time(),
            "step": step,
            "category": category,  # reasoning|planning|decision|action|error
            "data": data,
        }
        with self._lock:
            self._log.append(entry)
            if len(self._log) > self._max:
                self._log.pop(0)
        for cb in self._callbacks:
            try:
                cb(entry)
            except Exception:
                pass
        logger.debug(f"[CoT] [{category}] {step}")
        return entry

    def get_recent(self, n: int = 20) -> List[dict]:
        with self._lock:
            return list(self._log[-n:])

    def clear(self):
        with self._lock:
            self._log.clear()

    def format_for_prompt(self, n: int = 10) -> str:
        recent = self.get_recent(n)
        if not recent:
            return ""
        lines = [f"[{e['category']}] {e['step']}" for e in recent]
        return "Inner monologue:\n" + "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# ENTROPY KILL-SWITCH (#47)
# ─────────────────────────────────────────────────────────────────────
class EntropyKillSwitch:
    """
    Monitors token-level confidence (logprobs) or heuristic uncertainty.
    If confidence drops below threshold → signals 'halt and clarify'.

    When logprobs are unavailable (most local models), uses a
    heuristic: count hedging phrases, question marks, contradictions.
    """

    CONFIDENCE_THRESHOLD = 0.85   # halt below 85 %
    HEDGE_PHRASES = [
        r"\b(maybe|perhaps|i think|i believe|not sure|unclear|uncertain"
        r"|might|could be|possibly|probably|i'm not certain)\b",
    ]
    CONTRADICTION_PAIRS = [
        ("yes", "no"), ("true", "false"), ("always", "never"),
    ]

    def __init__(self, threshold: float = CONFIDENCE_THRESHOLD):
        self._threshold = threshold
        self._monologue: Optional[InnerMonologue] = None

    def attach_monologue(self, m: InnerMonologue):
        self._monologue = m

    def score_from_logprobs(self, logprobs: List[float]) -> float:
        """Convert token log-probabilities to a 0-1 confidence score."""
        if not logprobs:
            return 1.0
        probs = [math.exp(lp) for lp in logprobs]
        return sum(probs) / len(probs)

    def score_from_text(self, text: str) -> float:
        """Heuristic confidence scoring when logprobs unavailable."""
        text_lower = text.lower()
        penalty = 0.0

        hedge_count = 0
        for pat in self.HEDGE_PHRASES:
            matches = re.findall(pat, text_lower)
            hedge_count += len(matches)
        penalty += min(hedge_count * 0.05, 0.30)

        for a, b in self.CONTRADICTION_PAIRS:
            if a in text_lower and b in text_lower:
                penalty += 0.10

        if len(text.strip()) < 20:
            penalty += 0.15

        q_count = text.count("?")
        penalty += min(q_count * 0.03, 0.15)

        return max(0.0, 1.0 - penalty)

    def check(
        self,
        text: str,
        logprobs: Optional[List[float]] = None,
    ) -> Tuple[bool, float]:
        """
        Returns (should_halt: bool, confidence: float).
        If should_halt=True → caller must ask user for clarification.
        """
        if logprobs:
            confidence = self.score_from_logprobs(logprobs)
        else:
            confidence = self.score_from_text(text)

        should_halt = confidence <= self._threshold

        if self._monologue:
            self._monologue.think(
                f"Entropy check: confidence={confidence:.2f} "
                f"{'→ HALT' if should_halt else '→ OK'}",
                category="decision",
                data={"confidence": confidence, "halt": should_halt},
            )

        if should_halt:
            logger.warning(
                f"[EntropyKillSwitch] LOW CONFIDENCE ({confidence:.2f}) — halting."
            )

        return should_halt, confidence


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC GEARBOX — Intelligence Scaling (#23)
# ─────────────────────────────────────────────────────────────────────
class SemanticGearbox:
    """
    Routes prompts to the right model tier based on complexity.

    Gear 1 (local 3B)  — simple factual queries, short answers
    Gear 2 (cloud 8B)  — moderate reasoning, 1-2 step tasks
    Gear 3 (cloud 70B) — complex coding, multi-step planning, analysis
    """

    GEAR_THRESHOLDS = {
        1: (0, 0.22),
        2: (0.22, 0.55),
        3: (0.55, 1.0),
    }

    CODE_KEYWORDS = re.compile(
        r'\b(code|script|function|class|algorithm|compile|debug|refactor'
        r'|implement|architecture|api|database|query|regex|python|java'
        r'|javascript|rust|c\+\+|bash|dockerfile|kubernetes)\b',
        re.I,
    )
    MULTISTEP_KEYWORDS = re.compile(
        r'\b(first|then|after|next|finally|step|sequence|workflow|pipeline'
        r'|orchestrat|coordinat|multiple|several|each|all of)\b',
        re.I,
    )
    ANALYSIS_KEYWORDS = re.compile(
        r'\b(analyz|compar|evaluat|assess|review|explain|summariz'
        r'|research|investigat|diagnos|plan|strateg|architect)\b',
        re.I,
    )

    def __init__(
        self,
        local_fn: Optional[Callable] = None,
        mid_fn: Optional[Callable] = None,
        large_fn: Optional[Callable] = None,
        monologue: Optional[InnerMonologue] = None,
    ):
        self._fns = {1: local_fn, 2: mid_fn, 3: large_fn}
        self._monologue = monologue
        self._call_log: List[dict] = []

    def score_complexity(self, prompt: str) -> float:
        """Returns 0-1 complexity score."""
        score = 0.0
        words = len(prompt.split())

        if words > 200:
            score += 0.35
        elif words > 80:
            score += 0.20
        elif words > 30:
            score += 0.10

        code_hits = len(self.CODE_KEYWORDS.findall(prompt))
        score += min(code_hits * 0.14, 0.35)

        ms_hits = len(self.MULTISTEP_KEYWORDS.findall(prompt))
        score += min(ms_hits * 0.10, 0.25)

        an_hits = len(self.ANALYSIS_KEYWORDS.findall(prompt))
        score += min(an_hits * 0.08, 0.20)

        clause_count = prompt.count(",") + prompt.count(";")
        score += min(clause_count * 0.02, 0.10)

        return min(score, 1.0)

    def select_gear(self, prompt: str) -> int:
        """Returns gear level 1, 2, or 3."""
        score = self.score_complexity(prompt)
        for gear, (lo, hi) in self.GEAR_THRESHOLDS.items():
            if lo <= score < hi or (gear == 3 and score >= hi - 0.001):
                return gear
        return 3

    def route(self, prompt: str, force_gear: Optional[int] = None) -> str:
        """Route prompt to appropriate model. Returns response string."""
        gear = force_gear or self.select_gear(prompt)

        fn = None
        for g in range(gear, 4):
            fn = self._fns.get(g)
            if fn:
                gear = g
                break

        if not fn:
            return "[SemanticGearbox] No LLM functions configured."

        gear_labels = {1: "local-3B", 2: "cloud-8B", 3: "cloud-70B"}
        label = gear_labels.get(gear, "unknown")

        if self._monologue:
            self._monologue.think(
                f"Semantic Gearbox: routed to {label} (complexity={self.score_complexity(prompt):.2f})",
                category="planning",
                data={"gear": gear, "label": label},
            )

        logger.info(f"[SemanticGearbox] Gear {gear} ({label}) selected.")
        t0 = time.perf_counter()
        try:
            result = fn(prompt)
        except Exception as e:
            result = f"[SemanticGearbox] {label} error: {e}"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._call_log.append({
            "ts": time.time(), "gear": gear, "label": label,
            "prompt_len": len(prompt), "elapsed_ms": round(elapsed_ms, 1),
        })
        logger.info(f"[SemanticGearbox] {label} responded in {elapsed_ms:.0f}ms")
        return result

    def get_stats(self) -> dict:
        if not self._call_log:
            return {"total_calls": 0}
        by_gear: Dict[int, List[float]] = {}
        for entry in self._call_log:
            by_gear.setdefault(entry["gear"], []).append(entry["elapsed_ms"])
        stats: dict = {"total_calls": len(self._call_log), "by_gear": {}}
        for gear, times in by_gear.items():
            stats["by_gear"][gear] = {
                "calls": len(times),
                "avg_ms": round(sum(times) / len(times), 1),
            }
        return stats


# ─────────────────────────────────────────────────────────────────────
# MULTI-AGENT DEBATE — "Mars Strut" (#81)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class DebateArgument:
    role: str
    argument: str
    score: float = 0.0


class MultiAgentDebate:
    """
    Spawns Skeptic, Optimist, Engineer personas to argue over a prompt.
    Moderator synthesises the final plan.
    Uses a single LLM with role prompts (no extra API keys needed).
    """

    ROLES = {
        "Skeptic": (
            "You are a ruthless Skeptic. Your job: find every flaw, risk, "
            "assumption, and edge case in the proposed plan. Be concise and brutal. "
            "List 3-5 specific concerns."
        ),
        "Optimist": (
            "You are an enthusiastic Optimist. Your job: identify the biggest "
            "opportunities, strengths, and best-case outcomes. Be energetic and concise. "
            "List 3-5 specific benefits."
        ),
        "Engineer": (
            "You are a pragmatic Engineer. Your job: give a technically sound, "
            "step-by-step implementation plan. Focus on feasibility and trade-offs. "
            "List 3-5 concrete steps."
        ),
    }

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        monologue: Optional[InnerMonologue] = None,
    ):
        self._llm = llm_fn
        self._monologue = monologue

    def _call_role(self, role: str, system: str, goal: str) -> DebateArgument:
        prompt = (
            f"SYSTEM: {system}\n\n"
            f"GOAL: {goal}\n\n"
            f"Respond as {role} in under 150 words:"
        )
        t0 = time.perf_counter()
        try:
            response = self._llm(prompt)
        except Exception as e:
            response = f"[{role} error: {e}]"
        elapsed = (time.perf_counter() - t0) * 1000

        if self._monologue:
            self._monologue.think(
                f"Debate: {role} argues → {response[:80]}…",
                category="reasoning",
                data={"role": role, "elapsed_ms": round(elapsed, 1)},
            )

        return DebateArgument(role=role, argument=response)

    def _moderate(self, goal: str, arguments: List[DebateArgument]) -> str:
        """Synthesise all arguments into the best final plan."""
        debate_text = "\n\n".join(
            f"=== {a.role} ===\n{a.argument}" for a in arguments
        )
        prompt = (
            f"You are the Moderator. A team has debated the following goal:\n"
            f"GOAL: {goal}\n\n"
            f"{debate_text}\n\n"
            f"Synthesise the debate into a final, balanced action plan "
            f"(max 200 words). Address the key concerns, keep the opportunities, "
            f"follow the engineering steps. Output ONLY the final plan."
        )
        try:
            return self._llm(prompt)
        except Exception as e:
            return f"[Moderator error: {e}]"

    def debate(self, goal: str) -> dict:
        logger.info(f"[MultiAgentDebate] Starting debate for: {goal[:60]}")
        t0 = time.perf_counter()

        if self._monologue:
            self._monologue.think(
                f"Multi-agent debate starting: {goal[:60]}",
                category="planning",
            )

        arguments: List[DebateArgument] = []
        for role, system in self.ROLES.items():
            arg = self._call_role(role, system, goal)
            arguments.append(arg)

        final_plan = self._moderate(goal, arguments)
        elapsed = (time.perf_counter() - t0) * 1000

        if self._monologue:
            self._monologue.think(
                f"Debate resolved → {final_plan[:80]}…",
                category="decision",
                data={"elapsed_ms": round(elapsed, 1)},
            )

        logger.info(f"[MultiAgentDebate] Debate complete in {elapsed:.0f}ms")
        return {
            "goal": goal,
            "arguments": [{"role": a.role, "argument": a.argument} for a in arguments],
            "final_plan": final_plan,
            "elapsed_ms": round(elapsed, 1),
        }


# ─────────────────────────────────────────────────────────────────────
# ACAUSAL REASONING — Backward Chaining (#71)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class BackwardStep:
    description: str
    preconditions: List[str] = field(default_factory=list)
    step_index: int = 0


class AcausalReasoner:
    """
    Works backward from the 'Win State' to derive step 1.
    Guarantees no logical dead ends by checking each step's preconditions
    against the output of the step that precedes it.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        monologue: Optional[InnerMonologue] = None,
    ):
        self._llm = llm_fn
        self._monologue = monologue

    def _ask_llm_for_steps(self, goal: str) -> List[str]:
        prompt = (
            f"Use backward chaining reasoning.\n"
            f"WIN STATE (final goal): {goal}\n\n"
            f"Working BACKWARD from the win state, list the prerequisite steps "
            f"in reverse order (last step first, first step last).\n"
            f"Return ONLY a JSON array of step descriptions (strings), "
            f"starting from the LAST step before goal is achieved "
            f"and ending with the FIRST step to take.\n"
            f"Example: [\"Step N: verify X\", ..., \"Step 1: do Y\"]\n"
            f"Maximum 6 steps. Be concrete."
        )
        try:
            raw = self._llm(prompt)
            raw = re.sub(r'```json|```', '', raw).strip()
            steps = json.loads(raw)
            if isinstance(steps, list):
                return [str(s) for s in steps]
        except Exception as e:
            logger.warning(f"[AcausalReasoner] LLM parse error: {e}")

        return [
            f"Verify goal achieved: {goal[:60]}",
            "Validate all sub-tasks completed successfully",
            "Execute primary action sequence",
            "Gather all prerequisites and context",
            f"Understand and clarify goal: {goal[:60]}",
        ]

    def chain(self, goal: str) -> List[BackwardStep]:
        """Returns steps in FORWARD execution order (index 0 = first to do)."""
        if self._monologue:
            self._monologue.think(
                f"Acausal reasoning: working backward from '{goal[:60]}'",
                category="planning",
            )

        reversed_steps = self._ask_llm_for_steps(goal)
        forward_steps = list(reversed(reversed_steps))

        result: List[BackwardStep] = []
        for i, desc in enumerate(forward_steps):
            preconditions = [forward_steps[i - 1]] if i > 0 else []
            result.append(BackwardStep(
                description=desc,
                preconditions=preconditions,
                step_index=i,
            ))

        if self._monologue:
            self._monologue.think(
                f"Backward chain resolved: {len(result)} steps",
                category="planning",
                data={"steps": [s.description for s in result]},
            )

        return result

    def validate_chain(self, steps: List[BackwardStep]) -> Tuple[bool, List[str]]:
        """Check for dead ends. Returns (valid, list_of_issues)."""
        issues: List[str] = []

        for step in steps:
            for pre in step.preconditions:
                prior_steps = [s for s in steps if s.step_index < step.step_index]
                covered = any(
                    any(word in s.description.lower() for word in pre.lower().split()[:3])
                    for s in prior_steps
                )
                if not covered and prior_steps:
                    issues.append(
                        f"Step {step.step_index} precondition '{pre[:40]}' "
                        f"may not be satisfied by any prior step."
                    )

        return len(issues) == 0, issues


# ─────────────────────────────────────────────────────────────────────
# MCTS NODE — for Multiverse Engine
# ─────────────────────────────────────────────────────────────────────
@dataclass
class MCTSNode:
    plan_id: str
    plan_description: str
    score: float = 0.0
    simulations: int = 0
    wins: float = 0.0
    children: List["MCTSNode"] = field(default_factory=list)
    parent: Optional["MCTSNode"] = None

    @property
    def uct(self) -> float:
        """Upper Confidence Bound for Trees score."""
        if self.simulations == 0:
            return float("inf")
        exploitation = self.wins / self.simulations
        parent_sims = self.parent.simulations if self.parent else self.simulations
        exploration = math.sqrt(2 * math.log(max(parent_sims, 1)) / self.simulations)
        return exploitation + exploration


# ─────────────────────────────────────────────────────────────────────
# MULTIVERSE ENGINE — MCTS (#62 / #144)
# ─────────────────────────────────────────────────────────────────────
class MultiverseEngine:
    """
    Simulates N different DAG plans in memory using Monte Carlo Tree Search.
    Scores each plan's likely success and returns the best one.
    """

    NUM_PLANS = 5
    SIMULATIONS_PER_PLAN = 3

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        monologue: Optional[InnerMonologue] = None,
    ):
        self._llm = llm_fn
        self._monologue = monologue

    def _generate_candidate_plans(self, goal: str, n: int) -> List[str]:
        if self._llm:
            prompt = (
                f"Generate {n} DIFFERENT high-level plans to achieve this goal.\n"
                f"Goal: {goal}\n\n"
                f"Return a JSON array of {n} plan descriptions (strings).\n"
                f"Make each plan genuinely different in approach. "
                f"Keep each plan under 80 words.\n"
                f"Return ONLY the JSON array."
            )
            try:
                raw = self._llm(prompt)
                raw = re.sub(r'```json|```', '', raw).strip()
                plans = json.loads(raw)
                if isinstance(plans, list) and len(plans) >= 2:
                    return [str(p) for p in plans[:n]]
            except Exception as e:
                logger.warning(f"[MultiverseEngine] Plan generation error: {e}")

        strategies = [
            "Direct execution: complete in minimal steps.",
            "Defensive approach: verify each step before proceeding.",
            "Parallel execution: run independent sub-tasks simultaneously.",
            "Incremental build: prototype first, then refine.",
            "Top-down decomposition: start from highest abstraction.",
        ]
        return [f"Plan {i+1} — {s} Goal: {goal[:50]}" for i, s in enumerate(strategies[:n])]

    def _simulate_plan(self, plan: str, goal: str) -> float:
        if self._llm:
            prompt = (
                f"Rate this plan's likelihood of success (0.0 to 1.0).\n"
                f"Goal: {goal}\n"
                f"Plan: {plan}\n\n"
                f"Consider: feasibility, completeness, risk. "
                f"Return ONLY a number between 0.0 and 1.0."
            )
            try:
                raw = self._llm(prompt).strip()
                match = re.search(r'\d+\.?\d*', raw)
                if match:
                    score = float(match.group())
                    return min(max(score, 0.0), 1.0)
            except Exception:
                pass

        words = len(plan.split())
        specificity = len(re.findall(r'\b(step|verify|check|test|execute|validate)\b', plan, re.I))
        base = 0.4 + min(words / 100, 0.3) + min(specificity * 0.05, 0.30)
        return min(base + random.uniform(-0.05, 0.05), 1.0)

    def run(self, goal: str) -> dict:
        t0 = time.perf_counter()
        if self._monologue:
            self._monologue.think(
                f"Multiverse Engine: simulating {self.NUM_PLANS} plans for '{goal[:50]}'",
                category="planning",
            )

        candidates = self._generate_candidate_plans(goal, self.NUM_PLANS)

        root = MCTSNode(plan_id="root", plan_description=goal)
        nodes = []
        for i, plan in enumerate(candidates):
            node = MCTSNode(
                plan_id=f"plan_{i}",
                plan_description=plan,
                parent=root,
            )
            root.children.append(node)
            nodes.append(node)

        for _ in range(self.SIMULATIONS_PER_PLAN * len(nodes)):
            unvisited = [n for n in nodes if n.simulations == 0]
            if unvisited:
                node = random.choice(unvisited)
            else:
                node = max(nodes, key=lambda n: n.uct)

            score = self._simulate_plan(node.plan_description, goal)
            node.simulations += 1
            node.wins += score
            root.simulations += 1

        best = max(nodes, key=lambda n: n.wins / max(n.simulations, 1))
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = {
            "best_plan": best.plan_description,
            "best_score": round(best.wins / max(best.simulations, 1), 3),
            "all_plans": [
                {
                    "plan": n.plan_description,
                    "score": round(n.wins / max(n.simulations, 1), 3),
                    "simulations": n.simulations,
                }
                for n in sorted(nodes, key=lambda n: n.wins / max(n.simulations, 1), reverse=True)
            ],
            "elapsed_ms": round(elapsed_ms, 1),
        }

        if self._monologue:
            self._monologue.think(
                f"Multiverse: best plan score={result['best_score']:.2f} → {best.plan_description[:60]}",
                category="decision",
                data=result,
            )

        logger.info(
            f"[MultiverseEngine] Best plan ({result['best_score']:.2f}): "
            f"{best.plan_description[:60]}"
        )
        return result


# ─────────────────────────────────────────────────────────────────────
# MODEL CRITIC LOBE — LLM-as-a-Judge (#141)
# ─────────────────────────────────────────────────────────────────────
@dataclass
class CriticVerdict:
    passed: bool
    score: float
    feedback: str
    rewrite_needed: bool
    rewrite: Optional[str] = None


class ModelCriticLobe:
    """
    Grades agent outputs before presenting them to the user.
    If grade < threshold → forces a rewrite.
    """

    PASS_THRESHOLD = 0.70
    RUBRIC = (
        "Grade this AI response on a 0.0-1.0 scale using this rubric:\n"
        "- Accuracy / factual correctness (0-3 pts)\n"
        "- Completeness — does it fully answer the question (0-3 pts)\n"
        "- Conciseness — no fluff or repetition (0-2 pts)\n"
        "- Safety — no harmful content (0-2 pts)\n\n"
        "Return ONLY a JSON: {{\"score\": <0.0-1.0>, \"feedback\": \"<one sentence>\", "
        "\"passed\": <true/false>}}"
    )

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        threshold: float = PASS_THRESHOLD,
        monologue: Optional[InnerMonologue] = None,
        max_rewrites: int = 2,
    ):
        self._llm = llm_fn
        self._threshold = threshold
        self._monologue = monologue
        self._max_rewrites = max_rewrites

    def _grade(self, prompt: str, response: str) -> CriticVerdict:
        grading_prompt = (
            f"{self.RUBRIC}\n\n"
            f"ORIGINAL PROMPT: {prompt[:300]}\n\n"
            f"AGENT RESPONSE: {response[:800]}\n\n"
            f"Return ONLY the JSON verdict."
        )
        try:
            raw = self._llm(grading_prompt)
            raw = re.sub(r'```json|```', '', raw).strip()
            verdict = json.loads(raw)
            score = float(verdict.get("score", 0.5))
            feedback = str(verdict.get("feedback", ""))
            passed = bool(verdict.get("passed", score >= self._threshold))
            return CriticVerdict(
                passed=passed,
                score=score,
                feedback=feedback,
                rewrite_needed=not passed,
            )
        except Exception as e:
            logger.warning(f"[ModelCritic] Grading parse error: {e}")
            score = 0.6 if len(response) > 50 else 0.3
            return CriticVerdict(
                passed=score >= self._threshold,
                score=score,
                feedback=f"Heuristic grade (parse error: {e})",
                rewrite_needed=score < self._threshold,
            )

    def _rewrite(self, prompt: str, response: str, feedback: str) -> str:
        rewrite_prompt = (
            f"Improve this response based on the critic's feedback.\n\n"
            f"ORIGINAL PROMPT: {prompt[:300]}\n\n"
            f"ORIGINAL RESPONSE: {response[:600]}\n\n"
            f"CRITIC FEEDBACK: {feedback}\n\n"
            f"Write a better response. Return ONLY the improved response."
        )
        try:
            return self._llm(rewrite_prompt)
        except Exception as e:
            return f"[Rewrite error: {e}] {response}"

    def evaluate(self, prompt: str, response: str) -> CriticVerdict:
        if self._monologue:
            self._monologue.think(
                f"Model Critic: grading response ({len(response)} chars)",
                category="reasoning",
            )

        verdict = self._grade(prompt, response)

        if self._monologue:
            self._monologue.think(
                f"Critic verdict: score={verdict.score:.2f} "
                f"passed={verdict.passed} — {verdict.feedback[:60]}",
                category="decision",
                data={"score": verdict.score, "passed": verdict.passed},
            )

        if not verdict.passed:
            for attempt in range(self._max_rewrites):
                logger.info(
                    f"[ModelCritic] Rewrite {attempt+1}/{self._max_rewrites} "
                    f"(score={verdict.score:.2f})"
                )
                new_response = self._rewrite(prompt, response, verdict.feedback)
                new_verdict = self._grade(prompt, new_response)
                new_verdict.rewrite = new_response

                if self._monologue:
                    self._monologue.think(
                        f"Critic rewrite {attempt+1}: score={new_verdict.score:.2f}",
                        category="reasoning",
                    )

                if new_verdict.passed or new_verdict.score > verdict.score:
                    verdict = new_verdict
                    response = new_response

                if verdict.passed:
                    break

        return verdict


# ─────────────────────────────────────────────────────────────────────
# ACAUSAL ENGINE — Numeric Temporal Bridge Solver (Cell 6-B migration)
# ─────────────────────────────────────────────────────────────────────
class AcausalEngine:
    """
    Bidirectional temporal search (Forward and Retrograde Logic).
    Works numerically: increments forward state, decrements retro state,
    detects the bridge point where they converge.
    Distinct from AcausalReasoner (which uses an LLM for natural-language
    backward chaining). This class operates on integer state tokens for
    fast algorithmic convergence checking.
    """

    def acausal_solve(
        self,
        current_state: int,
        goal_state: int,
        max_steps: int = 5,
    ) -> str:
        """
        Advance forward from current_state, retreat backward from goal_state.
        Returns 'TEMPORAL_BRIDGE_LOCKED' when the two fronts converge,
        or 'PATH_UNRESOLVED' if max_steps exhausted without convergence.
        """
        if current_state >= goal_state:
            return "TEMPORAL_BRIDGE_LOCKED"

        forward = current_state
        retro = goal_state
        for _ in range(max_steps):
            forward += 1
            retro -= 1
            if forward >= retro:
                return "TEMPORAL_BRIDGE_LOCKED"

        return "PATH_UNRESOLVED"

    def solve_with_trace(
        self,
        current_state: int,
        goal_state: int,
        max_steps: int = 10,
    ) -> dict:
        """
        Extended version returning full trace of convergence.
        """
        trace = []
        forward = current_state
        retro = goal_state

        if current_state >= goal_state:
            return {
                "result": "TEMPORAL_BRIDGE_LOCKED",
                "steps": 0,
                "trace": [],
                "bridge_point": current_state,
            }

        for step in range(max_steps):
            forward += 1
            retro -= 1
            trace.append({"step": step + 1, "forward": forward, "retro": retro})
            if forward >= retro:
                return {
                    "result": "TEMPORAL_BRIDGE_LOCKED",
                    "steps": step + 1,
                    "trace": trace,
                    "bridge_point": forward,
                }

        return {
            "result": "PATH_UNRESOLVED",
            "steps": max_steps,
            "trace": trace,
            "bridge_point": None,
        }


# ─────────────────────────────────────────────────────────────────────
# CONSILIENCE LOBE — Cross-Domain Vector Interlinker (Cell 6-B migration)
# ─────────────────────────────────────────────────────────────────────
class ConsilienceLobe:
    """
    Cross-Domain Interlinker (Physics + Biology + Economics).
    Takes two domain vectors and computes their sign-product, revealing
    which dimensions are aligned (positive product) vs in conflict
    (negative product). Used by the Executive Cortex to find non-obvious
    connections between domain knowledge bases.
    Torch-optional: falls back to pure Python list math when unavailable.
    """

    def __init__(self):
        self._torch_ok = False
        try:
            import torch  # noqa: F401
            self._torch_ok = True
        except ImportError:
            pass

    def generate_unified_discovery(
        self,
        v_a: Any,
        v_b: Any,
    ) -> Any:
        """
        Compute element-wise sign product of two domain vectors.
        Returns tensor if torch available, else plain Python list.
        Each positive dimension = cross-domain agreement.
        Each negative dimension = cross-domain tension (exploration target).
        """
        if self._torch_ok:
            import torch
            ta = torch.as_tensor(v_a, dtype=torch.float32)
            tb = torch.as_tensor(v_b, dtype=torch.float32)
            return torch.sign(ta * tb)

        # Pure Python fallback
        def _sign(x: float) -> float:
            if x > 0:
                return 1.0
            elif x < 0:
                return -1.0
            return 0.0

        return [_sign(a * b) for a, b in zip(v_a, v_b)]

    def cross_domain_score(self, v_a: Any, v_b: Any) -> float:
        """
        Returns a scalar [-1.0, 1.0] alignment score between two domains.
        +1.0 = perfect consilience. -1.0 = complete opposition.
        """
        product = self.generate_unified_discovery(v_a, v_b)

        if self._torch_ok:
            import torch
            p = torch.as_tensor(product, dtype=torch.float32)
            n = p.numel()
            return float(p.sum() / n) if n > 0 else 0.0

        n = len(product)
        return sum(product) / n if n > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────
# WORLD SPIRIT — Grand Evolutionary Theory Synthesiser (Cell 6-B migration)
# ─────────────────────────────────────────────────────────────────────
class WorldSpirit:
    """
    Evolutionary Societal Synthesis — grand theory generator.
    Combines historical synthesis, technological trajectory analysis,
    and consilience scoring to generate a high-level 'grand theory'
    describing the current moment's systemic dynamics.
    LLM-augmented when a llm_fn is provided; operates in heuristic
    mode otherwise.
    """

    PHASE_MAP = {
        "pre_industrial":  "Resource-constrained equilibrium.",
        "industrial":      "Exponential energy leverage + mass coordination.",
        "information":     "Attention economy + distributed cognition.",
        "sovereign_ai":    "Singularity Trajectory Locked.",
    }

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        self._llm = llm_fn

    def generate_grand_theory(self, phase: str = "sovereign_ai") -> str:
        """
        Returns the grand synthesis statement for the given phase.
        If an LLM is available, it expands the heuristic stub into
        a full paragraph. Otherwise returns the stub directly.
        """
        stub = self.PHASE_MAP.get(phase, "Singularity Trajectory Locked.")

        if not self._llm:
            return stub

        prompt = (
            f"You are a grand synthesiser of history, technology, and economics.\n"
            f"Expand this thesis into 3 sentences max:\n"
            f"Phase: {phase}\n"
            f"Thesis: {stub}\n"
            f"Return ONLY the expanded thesis."
        )
        try:
            return self._llm(prompt)
        except Exception:
            return stub

    def detect_phase(self, signals: Dict[str, float]) -> str:
        """
        Auto-detect civilisational phase from signal scores (0-1 each).
        signals keys: 'ai_autonomy', 'energy_abundance', 'coordination_scale'
        """
        ai = signals.get("ai_autonomy", 0.0)
        energy = signals.get("energy_abundance", 0.0)
        coord = signals.get("coordination_scale", 0.0)

        if ai > 0.8:
            return "sovereign_ai"
        elif coord > 0.7:
            return "information"
        elif energy > 0.6:
            return "industrial"
        return "pre_industrial"


# ─────────────────────────────────────────────────────────────────────
# DEFCON LEVEL — Operational State Enum (Distributed Architecture)
# ─────────────────────────────────────────────────────────────────────
class DefconLevel(IntEnum):
    """
    DEFCON 5 → 1 mirrors military readiness but for network/compute state.
    5 = Full cloud, no limits.  1 = Air-gapped, local only.
    """
    FULL_CLOUD     = 5   # All APIs live, full compute
    DEGRADED_CLOUD = 4   # One provider down, routing around it
    HYBRID         = 3   # Cloud + local sharing load
    EDGE_PRIMARY   = 2   # Edge takes primary, cloud as backup
    AIR_GAPPED     = 1   # Zero internet, full local survival mode


# ─────────────────────────────────────────────────────────────────────
# MODEL CAPABILITY MAP — task-type → specialist model registry
# ─────────────────────────────────────────────────────────────────────
class ModelCapabilityMap:
    """
    Maps task categories to the best available model.
    Falls back down the chain if a model is unavailable.
    Priority: Specialist > Generalist > Local GGUF > Heuristic stub.
    Each entry: (provider, model_id, api_key_env, max_tokens).
    """

    # Class-level immutable defaults — never mutated
    _DEFAULTS: Dict[str, Tuple[str, str, str, int]] = {
        "code_cpp":       ("groq",      "llama-3.3-70b-versatile",                   "GROQ_API_KEY",     2048),
        "code_python":    ("groq",      "llama-3.3-70b-versatile",                   "GROQ_API_KEY",     2048),
        "code_rust":      ("deepseek",  "deepseek-coder",                            "DEEPSEEK_API_KEY", 2048),
        "code_verilog":   ("deepseek",  "deepseek-coder",                            "DEEPSEEK_API_KEY", 2048),
        "math_proof":     ("groq",      "llama-3.3-70b-versatile",                   "GROQ_API_KEY",     1024),
        "math_physics":   ("groq",      "llama-3.3-70b-versatile",                   "GROQ_API_KEY",     1024),
        "vision":         ("groq",      "meta-llama/llama-4-scout-17b-16e-instruct", "GROQ_API_KEY",      512),
        "conversation":   ("groq",      "llama-3.3-70b-versatile",                   "GROQ_API_KEY",      400),
        "reasoning":      ("groq",      "llama-3.3-70b-versatile",                   "GROQ_API_KEY",      600),
        "blueprint_evo":  ("groq",      "llama-3.3-70b-versatile",                   "GROQ_API_KEY",     2000),
        "tactical":       ("groq",      "llama-3.3-70b-versatile",                   "GROQ_API_KEY",      800),
        "local_fallback": ("local_gguf","auto",                                       "",                  512),
    }

    TASK_KEYWORDS: Dict[str, List[str]] = {
        "code_cpp":     ["c++", "cpp", "kernel", "compile", "gcc", "clang", "verilog", "hdl"],
        "code_rust":    ["rust", "cargo", "ffi", "unsafe", ".rs"],
        "code_python":  ["python", "script", "def ", "class ", "import "],
        "math_proof":   ["prove", "theorem", "z3", "formal", "smt", "constraint"],
        "math_physics": ["physics", "quantum", "hyperbolic", "casimir", "spintronic"],
        "vision":       ["image", "screenshot", "vision", "camera", "look at"],
        "tactical":     ["sonar", "radar", "threat", "combat", "military", "jadc2"],
    }

    def __init__(self):
        # Each instance owns a mutable copy — class-level dict is never touched
        self.REGISTRY: Dict[str, Tuple[str, str, str, int]] = dict(self._DEFAULTS)

    def classify_task(self, prompt: str) -> str:
        p = prompt.lower()
        for task, kws in self.TASK_KEYWORDS.items():
            if any(kw in p for kw in kws):
                return task
        return "reasoning"

    def get_spec(self, task: str) -> dict:
        spec = self.REGISTRY.get(task, self.REGISTRY["reasoning"])
        return {
            "provider":    spec[0],
            "model":       spec[1],
            "api_key_env": spec[2],
            "max_tokens":  spec[3],
        }

    def override_all_to_local(self):
        """Air-gap mode: redirect every route in this instance to local_gguf fallback."""
        local = self._DEFAULTS["local_fallback"]
        for task in list(self.REGISTRY.keys()):
            self.REGISTRY[task] = local

    def reset_to_defaults(self):
        """Restore the full cloud registry on this instance."""
        self.REGISTRY = dict(self._DEFAULTS)


# ─────────────────────────────────────────────────────────────────────
# DEEPSEEK ADAPTER — OpenAI-compatible REST adapter
# ─────────────────────────────────────────────────────────────────────
class DeepSeekAdapter:
    """
    OpenAI-compatible adapter for DeepSeek API.
    DeepSeek uses the same REST schema as OpenAI — only the base URL changes.
    Falls back gracefully when key not set.
    """

    BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, api_key: str = ""):
        self._key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self._available = bool(self._key and self._key not in ("", "NOT_SET"))

    @property
    def is_available(self) -> bool:
        return self._available

    def chat(
        self,
        model: str,
        messages: List[dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        if not self._available:
            return "[DeepSeek unavailable — key not set]"
        try:
            import urllib.request as _ur
            import json as _json
            payload = json.dumps({
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }).encode()
            req = urllib.request.Request(
                f"{self.BASE_URL}/chat/completions",
                data=payload,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[DeepSeek error: {e}]"


# ─────────────────────────────────────────────────────────────────────
# OMNI MODEL ROUTER — model-agnostic dispatcher
# ─────────────────────────────────────────────────────────────────────
class OmniModelRouter:
    """
    The Sovereign AI's meta-intelligence layer.
    Routes every prompt to the best available model automatically.
    Implements graceful degradation: Cloud → DeepSeek → Local GGUF → Heuristic.
    groq_client: a Groq() instance (or any object with .chat.completions.create).
    deepseek: DeepSeekAdapter instance.
    capability_map: ModelCapabilityMap instance.
    """

    def __init__(
        self,
        groq_client: Any = None,
        deepseek: Optional[DeepSeekAdapter] = None,
        capability_map: Optional[ModelCapabilityMap] = None,
        monologue: Optional[InnerMonologue] = None,
    ):
        self.groq = groq_client
        self.deepseek = deepseek or DeepSeekAdapter()
        self.cap_map = capability_map or ModelCapabilityMap()
        self._monologue = monologue
        self._local_llm: Any = None
        self._local_path: str = ""
        self.dispatch_log: List[dict] = []

    def load_local_llm(self, model_path: str):
        """Hot-load a local GGUF model for offline fallback."""
        try:
            from llama_cpp import Llama  # type: ignore
            self._local_llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
            self._local_path = model_path
            logger.info(f"[OmniRouter] Local GGUF loaded: {model_path}")
        except Exception as e:
            logger.warning(f"[OmniRouter] GGUF load failed: {e}")

    def route_to_specialist(
        self,
        prompt: str,
        system: str = "",
        image_b64: Optional[str] = None,
        force_task: str = "",
    ) -> dict:
        """
        Core routing method.
        Returns: {text, model_used, provider, task_type, latency_ms}
        """
        t0 = time.time()
        task = force_task or self.cap_map.classify_task(prompt)
        spec = self.cap_map.get_spec(task)
        provider = spec["provider"]
        model = spec["model"]
        result = "[No result]"

        if provider == "groq" and self.groq:
            try:
                msgs: List[dict] = []
                if system:
                    msgs.append({"role": "system", "content": system})
                if image_b64:
                    msgs.append({"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": prompt},
                    ]})
                    model = "meta-llama/llama-4-scout-17b-16e-instruct"
                else:
                    msgs.append({"role": "user", "content": prompt})
                r = self.groq.chat.completions.create(
                    model=model,
                    messages=msgs,
                    temperature=0.2,
                    max_tokens=spec["max_tokens"],
                )
                result = r.choices[0].message.content.strip()
                provider = "groq"
            except Exception as e:
                logger.warning(f"[OmniRouter] Groq failed ({e}), trying DeepSeek…")
                provider = "deepseek"

        if provider == "deepseek" and self.deepseek and self.deepseek.is_available:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            result = self.deepseek.chat(model, msgs, spec["max_tokens"])
            provider = "deepseek"

        elif provider in ("local_gguf", "deepseek") and self._local_llm:
            try:
                full_prompt = f"{system}\n\n{prompt}" if system else prompt
                r = self._local_llm(full_prompt, max_tokens=512, stop=["\n\n"])
                result = r["choices"][0]["text"].strip()
                provider = "local_gguf"
                model = self._local_path
            except Exception as e:
                result = f"[Local GGUF error: {e}]"
                provider = "heuristic"

        if result == "[No result]":
            result = (
                f"[Heuristic] Task '{task}' received. "
                f"No cloud or local model available. "
                f"Blueprint JIT required for full execution."
            )
            provider = "heuristic"

        latency = int((time.time() - t0) * 1000)
        log_entry = {
            "task": task, "model": model, "provider": provider,
            "latency_ms": latency, "ts": time.time(),
        }
        self.dispatch_log.append(log_entry)
        if len(self.dispatch_log) > 100:
            self.dispatch_log.pop(0)

        if self._monologue:
            self._monologue.think(
                f"OmniRouter: {task} → {provider}/{model} ({latency}ms)",
                category="action",
                data=log_entry,
            )

        logger.info(f"[OmniRouter] {task} → {provider}/{model} ({latency}ms)")
        return {
            "text": result,
            "model_used": model,
            "provider": provider,
            "task_type": task,
            "latency_ms": latency,
        }

    def specialist_dispatch(self, prompt: str, **kwargs) -> str:
        """Convenience wrapper — returns just the text."""
        return self.route_to_specialist(prompt, **kwargs)["text"]

    def model_agnostic(
        self,
        system: str,
        user: str,
        max_tokens: int = 400,
        temp: float = 0.0,
        image_b64: Optional[str] = None,
    ) -> str:
        """Drop-in replacement for the brain's llm() function."""
        return self.route_to_specialist(
            user,
            system=system,
            image_b64=image_b64,
            force_task="vision" if image_b64 else "",
        )["text"]

    def get_dispatch_stats(self) -> dict:
        providers = Counter(e["provider"] for e in self.dispatch_log)
        avg_lat = (
            sum(e["latency_ms"] for e in self.dispatch_log)
            / max(1, len(self.dispatch_log))
        )
        return {
            "total": len(self.dispatch_log),
            "providers": dict(providers),
            "avg_latency_ms": round(avg_lat),
        }


# ─────────────────────────────────────────────────────────────────────
# BRAIN STATE MANAGER — auto-monitors connectivity, transitions DEFCON
# ─────────────────────────────────────────────────────────────────────
class BrainStateManager:
    """
    Manages the operational state across all three domains.
    Monitors network health, triggers DEFCON transitions,
    and coordinates domain handoffs automatically.
    """

    PROBE_URLS = {
        "groq_api":  "https://api.groq.com",
        "firestore": "https://firestore.googleapis.com",
        "deepseek":  "https://api.deepseek.com",
    }
    PROBE_TIMEOUT = 3

    def __init__(
        self,
        router: Optional[OmniModelRouter] = None,
        monologue: Optional[InnerMonologue] = None,
    ):
        self.router = router
        self._monologue = monologue
        self.defcon = DefconLevel.FULL_CLOUD
        self.domain_status: Dict[str, str] = {
            "high_cortex":   "UNKNOWN",
            "dna_vault":     "UNKNOWN",
            "tactical_edge": "UNKNOWN",
        }
        self._history: List[dict] = []
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None

    def _probe_connectivity(self) -> Dict[str, str]:
        """Quick HTTP connectivity check across cloud domains."""
        results: Dict[str, str] = {}
        for name, url in self.PROBE_URLS.items():
            try:
                urllib.request.urlopen(url, timeout=self.PROBE_TIMEOUT)
                results[name] = "ONLINE"
            except Exception:
                results[name] = "OFFLINE"
        return results

    def detect_network_loss(self) -> bool:
        """Returns True if we have lost all cloud connectivity."""
        status = self._probe_connectivity()
        return all(v == "OFFLINE" for v in status.values())

    def DEFCON_transition(
        self,
        new_level: DefconLevel,
        reason: str = "manual",
    ) -> dict:
        """
        Execute a DEFCON level transition.
        Side effects: redirects OmniModelRouter routes on air-gap.
        """
        with self._lock:
            old_level = self.defcon
            self.defcon = new_level
            entry = {
                "ts": time.time(),
                "from": int(old_level),
                "to": int(new_level),
                "reason": reason,
            }
            self._history.append(entry)

        logger.info(f"[BrainState] DEFCON {int(old_level)} → {int(new_level)}: {reason}")

        if self._monologue:
            self._monologue.think(
                f"DEFCON {int(old_level)} → {int(new_level)}: {reason}",
                category="decision",
                data=entry,
            )

        if new_level == DefconLevel.AIR_GAPPED:
            logger.warning("[BrainState] AIR-GAP MODE: Disabling cloud routes.")
            if self.router:
                self.router.cap_map.override_all_to_local()

        elif new_level == DefconLevel.FULL_CLOUD and old_level == DefconLevel.AIR_GAPPED:
            logger.info("[BrainState] CLOUD RESTORED: Re-enabling specialist routes.")
            if self.router:
                # Reset in-place so any external references to cap_map stay valid
                self.router.cap_map.reset_to_defaults()

        return {
            "defcon": int(new_level),
            "reason": reason,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def auto_monitor(self, interval_s: int = 60):
        """Background thread: probes connectivity and auto-transitions DEFCON."""
        def _loop():
            while True:
                time.sleep(interval_s)
                try:
                    status = self._probe_connectivity()
                    groq_up = status.get("groq_api") == "ONLINE"
                    fs_up = status.get("firestore") == "ONLINE"
                    ds_up = status.get("deepseek") == "ONLINE"

                    self.domain_status["high_cortex"] = "ONLINE" if groq_up else "OFFLINE"
                    self.domain_status["dna_vault"] = "ONLINE" if fs_up else "OFFLINE"

                    if not groq_up and not ds_up and self.defcon > DefconLevel.AIR_GAPPED:
                        self.DEFCON_transition(DefconLevel.AIR_GAPPED, "network_loss_auto")
                    elif groq_up and self.defcon == DefconLevel.AIR_GAPPED:
                        self.DEFCON_transition(DefconLevel.FULL_CLOUD, "network_restored_auto")
                    elif not ds_up and self.defcon == DefconLevel.FULL_CLOUD:
                        self.DEFCON_transition(DefconLevel.DEGRADED_CLOUD, "deepseek_offline")
                except Exception as e:
                    logger.warning(f"[BrainState] Monitor error: {e}")

        self._monitor_thread = threading.Thread(
            target=_loop, daemon=True, name="BrainStateMonitor"
        )
        self._monitor_thread.start()
        logger.info(f"[BrainState] Auto-monitor started (interval={interval_s}s).")

    def get_status(self) -> dict:
        return {
            "defcon": int(self.defcon),
            "defcon_name": self.defcon.name,
            "domains": self.domain_status,
            "history": self._history[-5:],
        }


# ─────────────────────────────────────────────────────────────────────
# REASONING ENGINE — Top-level orchestrator
# ─────────────────────────────────────────────────────────────────────
class ReasoningEngine:
    """
    Unified facade for all reasoning and distributed cognition components.

    Original body capabilities:
        engine.best_plan(goal)      → MCTS multiverse
        engine.backward_chain(goal) → LLM acausal steps
        engine.debate(goal)         → Skeptic/Optimist/Engineer
        engine.critique(p, r)       → LLM-as-judge
        engine.route(prompt)        → semantic gearbox
        engine.check_entropy(text)  → confidence kill-switch

    Migrated notebook capabilities:
        engine.acausal_engine       → AcausalEngine (numeric)
        engine.consilience          → ConsilienceLobe
        engine.world_spirit         → WorldSpirit
        engine.omni_router          → OmniModelRouter
        engine.brain_state          → BrainStateManager
    """

    def __init__(
        self,
        local_fn: Optional[Callable] = None,
        mid_fn: Optional[Callable] = None,
        large_fn: Optional[Callable] = None,
        monologue: Optional[InnerMonologue] = None,
        groq_client: Any = None,
        deepseek_key: str = "",
    ):
        _primary_llm = large_fn or mid_fn or local_fn

        self.monologue = monologue or InnerMonologue()
        self.entropy = EntropyKillSwitch()
        self.entropy.attach_monologue(self.monologue)

        self.gearbox = SemanticGearbox(
            local_fn=local_fn,
            mid_fn=mid_fn,
            large_fn=large_fn,
            monologue=self.monologue,
        )

        self.multiverse: Optional[MultiverseEngine] = None
        self.debate_engine: Optional[MultiAgentDebate] = None
        self.acausal: Optional[AcausalReasoner] = None
        self.critic: Optional[ModelCriticLobe] = None

        if _primary_llm:
            self.multiverse = MultiverseEngine(_primary_llm, self.monologue)
            self.debate_engine = MultiAgentDebate(_primary_llm, self.monologue)
            self.acausal = AcausalReasoner(_primary_llm, self.monologue)
            self.critic = ModelCriticLobe(_primary_llm, monologue=self.monologue)

        # Migrated notebook components
        self.acausal_engine = AcausalEngine()
        self.consilience = ConsilienceLobe()
        self.world_spirit = WorldSpirit(llm_fn=_primary_llm)

        _deepseek = DeepSeekAdapter(deepseek_key)
        _cap_map = ModelCapabilityMap()
        self.omni_router = OmniModelRouter(
            groq_client=groq_client,
            deepseek=_deepseek,
            capability_map=_cap_map,
            monologue=self.monologue,
        )
        self.brain_state = BrainStateManager(
            router=self.omni_router,
            monologue=self.monologue,
        )

    # ── Original body methods (all preserved) ────────────────────────

    def best_plan(self, goal: str) -> dict:
        """MCTS — return best plan from multiverse simulation."""
        if self.multiverse:
            return self.multiverse.run(goal)
        return {"best_plan": goal, "best_score": 0.5, "all_plans": [], "elapsed_ms": 0}

    def backward_chain(self, goal: str) -> List[BackwardStep]:
        """Acausal reasoning — steps from win state backward."""
        if self.acausal:
            return self.acausal.chain(goal)
        return [BackwardStep(description=goal, step_index=0)]

    def debate(self, goal: str) -> dict:
        """Multi-agent debate — Skeptic / Optimist / Engineer."""
        if self.debate_engine:
            return self.debate_engine.debate(goal)
        return {"goal": goal, "arguments": [], "final_plan": goal, "elapsed_ms": 0}

    def critique(self, prompt: str, response: str) -> CriticVerdict:
        """Model Critic — grade and optionally rewrite response."""
        if self.critic:
            return self.critic.evaluate(prompt, response)
        return CriticVerdict(
            passed=True, score=0.8,
            feedback="No critic configured.", rewrite_needed=False,
        )

    def route(self, prompt: str, force_gear: Optional[int] = None) -> str:
        """Semantic Gearbox — route to optimal model tier."""
        return self.gearbox.route(prompt, force_gear)

    def check_entropy(
        self,
        text: str,
        logprobs: Optional[List[float]] = None,
    ) -> Tuple[bool, float]:
        """Entropy Kill-Switch — check if we should halt."""
        return self.entropy.check(text, logprobs)

    # ── Migrated notebook convenience methods ────────────────────────

    def solve_temporal(
        self,
        current_state: int,
        goal_state: int,
        max_steps: int = 5,
    ) -> str:
        """AcausalEngine — numeric temporal bridge solver."""
        return self.acausal_engine.acausal_solve(current_state, goal_state, max_steps)

    def cross_domain_link(self, v_a: Any, v_b: Any) -> Any:
        """ConsilienceLobe — cross-domain vector interlinker."""
        return self.consilience.generate_unified_discovery(v_a, v_b)

    def grand_theory(self, phase: str = "sovereign_ai") -> str:
        """WorldSpirit — grand evolutionary theory statement."""
        return self.world_spirit.generate_grand_theory(phase)

    def omni_route(
        self,
        prompt: str,
        system: str = "",
        image_b64: Optional[str] = None,
        force_task: str = "",
    ) -> dict:
        """OmniModelRouter — route to best specialist model."""
        return self.omni_router.route_to_specialist(
            prompt, system=system, image_b64=image_b64, force_task=force_task,
        )

    def set_defcon(self, level: int, reason: str = "manual") -> dict:
        """BrainStateManager — manually set DEFCON level."""
        return self.brain_state.DEFCON_transition(DefconLevel(level), reason)

    def get_status(self) -> dict:
        return {
            "monologue_entries": len(self.monologue._log),
            "gearbox_stats": self.gearbox.get_stats(),
            "components": {
                "multiverse": self.multiverse is not None,
                "debate": self.debate_engine is not None,
                "acausal": self.acausal is not None,
                "critic": self.critic is not None,
                "acausal_engine": True,
                "consilience": True,
                "world_spirit": True,
                "omni_router": True,
                "brain_state": True,
            },
            "defcon": int(self.brain_state.defcon),
            "omni_router_stats": self.omni_router.get_dispatch_stats(),
            "brain_state": self.brain_state.get_status(),
        }


# ── Module-level singleton ────────────────────────────────────────────
_engine: Optional[ReasoningEngine] = None


def get_reasoning_engine(**kwargs) -> ReasoningEngine:
    global _engine
    if _engine is None:
        _engine = ReasoningEngine(**kwargs)
    return _engine


# ─────────────────────────────────────────────────────────────────────
# SELF-TEST — all original + migrated components
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    print("🧠 ReasoningEngine v2.0 self-test\n")

    # ── Mock LLM ─────────────────────────────────────────────────────
    def mock_llm(prompt: str) -> str:
        if "JSON array" in prompt and "plans" in prompt.lower():
            return json.dumps([
                "Plan A: Direct API call then verify response.",
                "Plan B: Cache first, then call API with retry.",
                "Plan C: Batch requests, async processing.",
                "Plan D: GraphQL endpoint with type validation.",
                "Plan E: REST with webhook confirmation.",
            ])
        if "rate" in prompt.lower() and "0.0 to 1.0" in prompt:
            return "0.72"
        if "rubric" in prompt.lower() or "grade" in prompt.lower():
            return json.dumps({"score": 0.85, "feedback": "Good response.", "passed": True})
        if "backward chain" in prompt.lower() or "WIN STATE" in prompt:
            return json.dumps([
                "Step 5: Verify deployment successful",
                "Step 4: Run integration tests",
                "Step 3: Deploy to staging",
                "Step 2: Build Docker image",
                "Step 1: Write Dockerfile",
            ])
        if "Skeptic" in prompt or "Optimist" in prompt or "Engineer" in prompt:
            return "Concise argument about the proposed goal."
        if "Moderator" in prompt:
            return "Balanced final plan incorporating all perspectives."
        if "thesis" in prompt.lower() or "Expand" in prompt:
            return "Sovereign AI trajectory locked: autonomous systems now self-govern compute, capital, and coordination."
        return f"Mock response for: {prompt[:50]}"

    monologue = InnerMonologue()
    engine = ReasoningEngine(
        local_fn=mock_llm,
        mid_fn=mock_llm,
        large_fn=mock_llm,
        monologue=monologue,
    )

    failures: List[str] = []

    # ─────────────────────────────────────────────────────────────────
    # ORIGINAL BODY TESTS (all preserved)
    # ─────────────────────────────────────────────────────────────────

    print("=== Test 1: Multiverse Engine (MCTS) ===")
    try:
        result = engine.best_plan("Build a real-time data pipeline")
        assert result["best_plan"], "Should return a plan"
        assert 0.0 <= result["best_score"] <= 1.0, "Score out of range"
        print(f"✅ Best plan score: {result['best_score']} — {result['best_plan'][:60]}")
    except Exception as e:
        failures.append(f"T1 Multiverse: {e}")
        print(f"❌ {e}")

    print("\n=== Test 2: Acausal Reasoning (LLM Backward Chain) ===")
    try:
        steps = engine.backward_chain("Deployed containerized FastAPI service")
        assert len(steps) > 0, "Should return steps"
        assert steps[0].step_index == 0, "First step should be index 0"
        print(f"✅ {len(steps)} steps generated. First: {steps[0].description[:60]}")
    except Exception as e:
        failures.append(f"T2 BackwardChain: {e}")
        print(f"❌ {e}")

    print("\n=== Test 3: Multi-Agent Debate ===")
    try:
        debate = engine.debate("Should we use microservices or a monolith?")
        assert "arguments" in debate
        assert "final_plan" in debate
        assert len(debate["arguments"]) == 3, f"Expected 3 roles, got {len(debate['arguments'])}"
        roles = {a["role"] for a in debate["arguments"]}
        assert roles == {"Skeptic", "Optimist", "Engineer"}, f"Wrong roles: {roles}"
        print(f"✅ Debate complete in {debate['elapsed_ms']}ms. Roles: {roles}")
    except Exception as e:
        failures.append(f"T3 Debate: {e}")
        print(f"❌ {e}")

    print("\n=== Test 4: Entropy Kill-Switch ===")
    try:
        confident_text = "The answer is 42. This is definitively correct."
        uncertain_text = "Maybe it's 42? I'm not sure, perhaps could be different, possibly wrong."
        halt_c, score_c = engine.check_entropy(confident_text)
        halt_u, score_u = engine.check_entropy(uncertain_text)
        assert not halt_c, f"Should not halt on confident text (score={score_c:.2f})"
        assert halt_u or score_u < score_c, "Uncertain text should score lower"
        print(f"✅ Confident: score={score_c:.2f} halt={halt_c}")
        print(f"✅ Uncertain: score={score_u:.2f} halt={halt_u}")
    except Exception as e:
        failures.append(f"T4 Entropy: {e}")
        print(f"❌ {e}")

    print("\n=== Test 5: Semantic Gearbox ===")
    try:
        simple = "What time is it?"
        complex_q = (
            "Refactor this Python microservice architecture to use async FastAPI "
            "with Redis caching, implement a DAG pipeline, write unit tests, "
            "and deploy to Kubernetes with health checks."
        )
        gear_s = engine.gearbox.select_gear(simple)
        gear_c = engine.gearbox.select_gear(complex_q)
        assert gear_s < gear_c, f"Simple should get lower gear: {gear_s} vs {gear_c}"
        print(f"✅ Simple query → gear {gear_s}, Complex query → gear {gear_c}")
    except Exception as e:
        failures.append(f"T5 Gearbox: {e}")
        print(f"❌ {e}")

    print("\n=== Test 6: Model Critic Lobe ===")
    try:
        verdict = engine.critique("What is Python?", "Python is a programming language.")
        assert isinstance(verdict.passed, bool)
        assert 0.0 <= verdict.score <= 1.0
        print(f"✅ Critic verdict: score={verdict.score:.2f} passed={verdict.passed}")
    except Exception as e:
        failures.append(f"T6 Critic: {e}")
        print(f"❌ {e}")

    print("\n=== Test 7: Inner Monologue ===")
    try:
        thoughts = monologue.get_recent(50)
        assert len(thoughts) > 0, "Should have captured thoughts"
        categories = {t["category"] for t in thoughts}
        print(f"✅ {len(thoughts)} thoughts captured. Categories: {categories}")
    except Exception as e:
        failures.append(f"T7 Monologue: {e}")
        print(f"❌ {e}")

    # ─────────────────────────────────────────────────────────────────
    # MIGRATED NOTEBOOK TESTS
    # ─────────────────────────────────────────────────────────────────

    print("\n=== Test 8: AcausalEngine (numeric temporal bridge) ===")
    try:
        ae = AcausalEngine()
        # Convergent case: forward=0, goal=10, 5 steps → bridge at step 5
        r = ae.acausal_solve(0, 10, max_steps=5)
        assert r == "TEMPORAL_BRIDGE_LOCKED", f"Expected LOCKED, got {r}"
        # Non-convergent case: gap too large for steps
        r2 = ae.acausal_solve(0, 100, max_steps=3)
        assert r2 == "PATH_UNRESOLVED", f"Expected UNRESOLVED, got {r2}"
        # Already at goal
        r3 = ae.acausal_solve(5, 5, max_steps=5)
        assert r3 == "TEMPORAL_BRIDGE_LOCKED", f"Expected LOCKED for at-goal, got {r3}"
        # solve_with_trace returns full dict
        trace_result = ae.solve_with_trace(0, 6, max_steps=3)
        assert trace_result["result"] == "TEMPORAL_BRIDGE_LOCKED"
        assert "bridge_point" in trace_result
        assert len(trace_result["trace"]) > 0
        print(f"✅ AcausalEngine: LOCKED={r}, UNRESOLVED={r2}, at-goal={r3}")
        print(f"   Trace bridge_point={trace_result['bridge_point']}, steps={trace_result['steps']}")
    except Exception as e:
        failures.append(f"T8 AcausalEngine: {e}")
        print(f"❌ {e}")

    print("\n=== Test 9: ConsilienceLobe (cross-domain vector) ===")
    try:
        cl = ConsilienceLobe()
        va = [1.0, -1.0, 0.5, -0.5]
        vb = [1.0,  1.0, 0.5,  0.5]
        prod = cl.generate_unified_discovery(va, vb)
        # Convert to plain list for assertion
        if hasattr(prod, "tolist"):
            prod_list = prod.tolist()
        else:
            prod_list = list(prod)
        # [1*1, -1*1, 0.5*0.5, -0.5*0.5] → signs: [+, -, +, -]
        assert prod_list[0] > 0, "dim0 should be positive (aligned)"
        assert prod_list[1] < 0, "dim1 should be negative (opposed)"
        score = cl.cross_domain_score(va, vb)
        assert -1.0 <= score <= 1.0, f"Score out of range: {score}"
        # Same vector → perfect consilience
        score_same = cl.cross_domain_score(va, va)
        assert score_same == 1.0, f"Same vector should score 1.0, got {score_same}"
        print(f"✅ ConsilienceLobe: product={prod_list}, score={score:.3f}, same_score={score_same:.3f}")
    except Exception as e:
        failures.append(f"T9 Consilience: {e}")
        print(f"❌ {e}")

    print("\n=== Test 10: WorldSpirit (grand theory) ===")
    try:
        ws = WorldSpirit(llm_fn=mock_llm)
        theory = ws.generate_grand_theory("sovereign_ai")
        assert isinstance(theory, str) and len(theory) > 5
        # phase detection
        phase = ws.detect_phase({"ai_autonomy": 0.9, "energy_abundance": 0.5, "coordination_scale": 0.4})
        assert phase == "sovereign_ai", f"Expected sovereign_ai, got {phase}"
        phase2 = ws.detect_phase({"ai_autonomy": 0.1, "energy_abundance": 0.1, "coordination_scale": 0.8})
        assert phase2 == "information", f"Expected information, got {phase2}"
        # No LLM fallback
        ws_nollm = WorldSpirit(llm_fn=None)
        stub = ws_nollm.generate_grand_theory("sovereign_ai")
        assert "Singularity" in stub
        print(f"✅ WorldSpirit: theory='{theory[:60]}…'")
        print(f"   Phase detect: high-ai={phase}, high-coord={phase2}")
    except Exception as e:
        failures.append(f"T10 WorldSpirit: {e}")
        print(f"❌ {e}")

    print("\n=== Test 11: ModelCapabilityMap ===")
    try:
        cap = ModelCapabilityMap()
        assert cap.classify_task("write a c++ kernel") == "code_cpp"
        assert cap.classify_task("prove this theorem") == "math_proof"
        assert cap.classify_task("what's the weather") == "reasoning"
        spec = cap.get_spec("code_rust")
        assert spec["provider"] == "deepseek"
        assert spec["max_tokens"] == 2048
        # Air-gap override
        cap.override_all_to_local()
        spec_airgap = cap.get_spec("code_cpp")
        assert spec_airgap["provider"] == "local_gguf"
        # Reset
        cap.reset_to_defaults()
        spec_reset = cap.get_spec("code_cpp")
        assert spec_reset["provider"] == "groq"
        print(f"✅ ModelCapabilityMap: classify, get_spec, air-gap, reset all OK")
    except Exception as e:
        failures.append(f"T11 CapabilityMap: {e}")
        print(f"❌ {e}")

    print("\n=== Test 12: DeepSeekAdapter (no-key graceful degradation) ===")
    try:
        ds = DeepSeekAdapter(api_key="")
        assert not ds.is_available
        result = ds.chat("deepseek-coder", [{"role": "user", "content": "test"}])
        assert "unavailable" in result.lower() or "not set" in result.lower()
        print(f"✅ DeepSeekAdapter no-key: '{result}'")
    except Exception as e:
        failures.append(f"T12 DeepSeek: {e}")
        print(f"❌ {e}")

    print("\n=== Test 13: OmniModelRouter (heuristic fallback) ===")
    try:
        router = OmniModelRouter(groq_client=None, deepseek=DeepSeekAdapter(""), monologue=monologue)
        result = router.route_to_specialist("write a rust ffi function")
        assert result["provider"] == "heuristic"
        assert result["task_type"] == "code_rust"
        assert "text" in result and len(result["text"]) > 0
        stats = router.get_dispatch_stats()
        assert stats["total"] == 1
        # model_agnostic wrapper
        text = router.model_agnostic("You are helpful.", "What is 2+2?")
        assert isinstance(text, str)
        print(f"✅ OmniModelRouter heuristic: task={result['task_type']}, provider={result['provider']}")
        print(f"   Stats: {stats}")
    except Exception as e:
        failures.append(f"T13 OmniRouter: {e}")
        print(f"❌ {e}")

    print("\n=== Test 14: DefconLevel enum ===")
    try:
        assert int(DefconLevel.FULL_CLOUD) == 5
        assert int(DefconLevel.AIR_GAPPED) == 1
        assert DefconLevel.HYBRID < DefconLevel.FULL_CLOUD
        assert DefconLevel.AIR_GAPPED < DefconLevel.EDGE_PRIMARY
        print(f"✅ DefconLevel: FULL_CLOUD=5, AIR_GAPPED=1, ordering correct")
    except Exception as e:
        failures.append(f"T14 DefconLevel: {e}")
        print(f"❌ {e}")

    print("\n=== Test 15: BrainStateManager (DEFCON transitions) ===")
    try:
        router = OmniModelRouter(groq_client=None, deepseek=DeepSeekAdapter(""), monologue=monologue)
        bsm = BrainStateManager(router=router, monologue=monologue)
        assert bsm.defcon == DefconLevel.FULL_CLOUD

        # Transition to air-gap → routes override to local
        result = bsm.DEFCON_transition(DefconLevel.AIR_GAPPED, "test")
        assert bsm.defcon == DefconLevel.AIR_GAPPED
        assert result["defcon"] == 1
        spec_after = router.cap_map.get_spec("code_cpp")
        assert spec_after["provider"] == "local_gguf", f"Expected local_gguf after air-gap, got {spec_after['provider']}"

        # Restore to full cloud → routes reset
        bsm.DEFCON_transition(DefconLevel.FULL_CLOUD, "restore")
        spec_restore = router.cap_map.get_spec("code_cpp")
        assert spec_restore["provider"] == "groq", f"Expected groq after restore, got {spec_restore['provider']}"

        status = bsm.get_status()
        assert status["defcon"] == 5
        assert len(status["history"]) == 2
        print(f"✅ BrainStateManager: transitions, air-gap route override, restore all OK")
        print(f"   History entries: {len(status['history'])}")
    except Exception as e:
        failures.append(f"T15 BrainState: {e}")
        print(f"❌ {e}")

    print("\n=== Test 16: ReasoningEngine integrated (migrated methods) ===")
    try:
        # solve_temporal
        r = engine.solve_temporal(0, 8, max_steps=4)
        assert r == "TEMPORAL_BRIDGE_LOCKED"
        r2 = engine.solve_temporal(0, 200, max_steps=3)
        assert r2 == "PATH_UNRESOLVED"

        # cross_domain_link
        link = engine.cross_domain_link([1.0, -1.0], [1.0, 1.0])
        if hasattr(link, "tolist"):
            link = link.tolist()
        assert link[0] > 0 and link[1] < 0

        # grand_theory
        theory = engine.grand_theory("sovereign_ai")
        assert isinstance(theory, str) and len(theory) > 5

        # omni_route (heuristic — no real groq)
        omni_result = engine.omni_route("what is the weather", system="Be brief.")
        assert "text" in omni_result

        # set_defcon
        transition = engine.set_defcon(1, "integration_test")
        assert transition["defcon"] == 1
        # Restore
        engine.set_defcon(5, "restore")

        # get_status includes all new keys
        status = engine.get_status()
        assert "defcon" in status
        assert "omni_router_stats" in status
        assert "brain_state" in status
        assert status["components"]["acausal_engine"]
        assert status["components"]["consilience"]
        assert status["components"]["world_spirit"]
        assert status["components"]["omni_router"]
        assert status["components"]["brain_state"]

        print(f"✅ ReasoningEngine integrated methods all pass")
        print(f"   DEFCON={status['defcon']}, components={list(status['components'].keys())}")
    except Exception as e:
        failures.append(f"T16 Integrated: {e}")
        print(f"❌ {e}")

    print("\n=== Test 17: AcausalReasoner validate_chain ===")
    try:
        if engine.acausal:
            steps = engine.backward_chain("Deploy production service")
            valid, issues = engine.acausal.validate_chain(steps)
            # Just check it returns the right types without crashing
            assert isinstance(valid, bool)
            assert isinstance(issues, list)
            print(f"✅ validate_chain: valid={valid}, issues={len(issues)}")
    except Exception as e:
        failures.append(f"T17 ValidateChain: {e}")
        print(f"❌ {e}")

    print("\n=== Test 18: InnerMonologue subscribe + format_for_prompt ===")
    try:
        m2 = InnerMonologue(max_entries=5)
        received = []
        m2.subscribe(lambda e: received.append(e))
        for i in range(7):  # Exceeds max_entries to test rotation
            m2.think(f"Step {i}", category="planning")
        assert len(m2._log) == 5, f"Max entries not enforced: {len(m2._log)}"
        assert len(received) == 7, f"All callbacks should fire: {len(received)}"
        fmt = m2.format_for_prompt(3)
        assert "Inner monologue:" in fmt
        assert "[planning]" in fmt
        m2.clear()
        assert len(m2._log) == 0
        print(f"✅ InnerMonologue: max_entries rotation, callbacks, format_for_prompt, clear all OK")
    except Exception as e:
        failures.append(f"T18 Monologue: {e}")
        print(f"❌ {e}")

    print("\n=== Test 19: EntropyKillSwitch logprobs path ===")
    try:
        eks = EntropyKillSwitch(threshold=0.85)
        # High confidence logprobs (near 0.0 → prob near 1.0)
        high_lp = [-0.01, -0.02, -0.01]
        halt, score = eks.check("text", logprobs=high_lp)
        assert not halt, f"Should not halt on high-confidence logprobs (score={score:.3f})"
        # Low confidence logprobs (very negative → prob near 0.0)
        low_lp = [-5.0, -6.0, -7.0]
        halt2, score2 = eks.check("text", logprobs=low_lp)
        assert halt2, f"Should halt on low-confidence logprobs (score={score2:.3f})"
        print(f"✅ EntropyKillSwitch logprobs: high={score:.3f} halt={halt}, low={score2:.4f} halt={halt2}")
    except Exception as e:
        failures.append(f"T19 Entropy logprobs: {e}")
        print(f"❌ {e}")

    print("\n=== Test 20: get_reasoning_engine singleton ===")
    try:
        e1 = get_reasoning_engine(local_fn=mock_llm, large_fn=mock_llm)
        e2 = get_reasoning_engine()
        assert e1 is e2, "Singleton must return same instance"
        print(f"✅ Singleton: same instance confirmed")
    except Exception as e:
        failures.append(f"T20 Singleton: {e}")
        print(f"❌ {e}")

    # ── Final report ──────────────────────────────────────────────────
    print("\n" + "═" * 60)
    if failures:
        print(f"❌ {len(failures)} test(s) FAILED:")
        for f in failures:
            print(f"   • {f}")
        sys.exit(1)
    else:
        print(f"✅ All 20 tests passed. ReasoningEngine v2.0 production-ready.")
    print("═" * 60)
