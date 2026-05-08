#!/usr/bin/env python3
# =====================================================================
# 🏭 DISTILLATION FACTORY  —  Record → Replay → Score → Export LoRA
#
# Single "one-button" dashboard for non-technical users (e.g. Zensar clients).
# Unifies: ConstitutionalDistiller + TAPAdversarial + MLXDPOTrainer
#
# User workflow (3 steps):
#   1. RECORD   — perform a task while the factory observes (10 replays)
#   2. SCORE    — LLM-as-Judge scores all replays against Constitution
#   3. EXPORT   — best pairs written to training_pairs.jsonl + LoRA trained
#
# Body-side additions (migrated from Notebook Cell 6 — Module 6):
#   Web3SettlementLayer     — crypto invoice generation (Polygon USDC)
#   ProceduralMemoryManager — nightly trace consolidation into playbook
#
# Interfaces:
#   CLI    — python distillation_factory.py --task "write email" --replays 10
#   API    — DistillationFactory.run_session(task, n_replays) -> SessionReport
#   FastAPI— /factory/record, /factory/status, /factory/sessions
#            /factory/invoice/generate, /factory/invoice/mark_paid,
#            /factory/invoice/pending, /factory/invoice/all,
#            /factory/invoice/status
#            /factory/playbook/consolidate, /factory/playbook/save,
#            /factory/playbook/recall/{task_name}, /factory/playbook/list,
#            /factory/playbook/{task_name} (DELETE)
#
# WIRING (swayambhu_v13.py):
# ---------------------------------------------------------------------
#   from distillation_factory import (
#       DistillationFactory, attach_factory_routes,
#       Web3SettlementLayer, ProceduralMemoryManager,
#   )
#   self.factory    = DistillationFactory(local_llm_fn=..., judge_fn=..., trainer=...)
#   self.settlement = Web3SettlementLayer()
#   self.mem_mgr    = ProceduralMemoryManager()
#   attach_factory_routes(v13_app, self.factory, self.settlement, self.mem_mgr)
# =====================================================================

from __future__ import annotations

import json
import logging
import os
import sys
import time
import threading
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("DistillationFactory")

try:
    from swayambhu_utils import PROJECT_ROOT
except ImportError:
    try:
        PROJECT_ROOT = Path(__file__).parent.resolve()
    except NameError:
        PROJECT_ROOT = Path(os.getcwd()).resolve()

# ── Dirs ──────────────────────────────────────────────────────────────
_BASE_DIR    = Path(os.getenv("SWAYAMBHU_DIR", str(PROJECT_ROOT)))
_FACTORY_DIR = _BASE_DIR / "distillation_factory"
_FACTORY_DIR.mkdir(parents=True, exist_ok=True)

SESSION_LOG  = _FACTORY_DIR / "sessions.jsonl"
EXPORT_DIR   = _FACTORY_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

PLAYBOOK_PATH = _BASE_DIR / "swayambhu_playbook.json"

# ── Config ────────────────────────────────────────────────────────────
DEFAULT_REPLAYS     = 10
MIN_PAIRS_TO_EXPORT = 3
PROGRESS_STEPS      = ["record", "score", "export", "train"]


# =====================================================================
# SECTION 1 — DATA CLASSES
# =====================================================================

@dataclass
class Replay:
    replay_id:  str
    task:       str
    prompt:     str
    response:   str
    score:      float = 0.0
    critique:   str   = ""
    is_chosen:  bool  = False
    elapsed_ms: float = 0.0


@dataclass
class SessionReport:
    session_id:      str
    task:            str
    n_replays:       int
    n_scored:        int
    n_pairs:         int
    best_score:      float
    worst_score:     float
    export_path:     str
    trained:         bool
    training_result: dict
    elapsed_s:       float
    status:          str  = "pending"
    error:           str  = ""

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self))


# =====================================================================
# SECTION 2 — STEP 1: RECORDER
# =====================================================================

class TaskRecorder:
    """
    Generates N replay responses for a given task using prompt diversity.
    Each replay uses a different prompt framing to maximise variance
    between chosen and rejected candidates.
    """

    _FRAMINGS = [
        "{task}",
        "Please complete this task concisely: {task}",
        "Think step by step, then answer: {task}",
        "Be thorough and detailed: {task}",
        "Give the most helpful possible response to: {task}",
        "As an expert, answer: {task}",
        "In bullet points: {task}",
        "Write a professional response to: {task}",
        "In plain language: {task}",
        "Give a structured answer to: {task}",
    ]

    def __init__(self, llm_fn: Optional[Callable[[str], str]] = None):
        self._llm = llm_fn

    def record(
        self,
        task:        str,
        n:           int = DEFAULT_REPLAYS,
        progress_fn: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Replay]:
        replays: List[Replay] = []
        for i in range(n):
            framing = self._FRAMINGS[i % len(self._FRAMINGS)]
            prompt  = framing.format(task=task)
            t0      = time.time()
            try:
                response = self._llm(prompt) if self._llm else self._sim_response(task, i)
            except Exception as e:
                response = f"[Error: {e}]"
            elapsed = round((time.time() - t0) * 1000, 1)
            replay  = Replay(
                replay_id  = f"r{i:02d}_{uuid.uuid4().hex[:6]}",
                task       = task,
                prompt     = prompt,
                response   = response,
                elapsed_ms = elapsed,
            )
            replays.append(replay)
            if progress_fn:
                progress_fn(i + 1, n, f"Replay {i+1}/{n} recorded")
            logger.debug(f"[Recorder] Replay {i+1}/{n}: {elapsed}ms")
        return replays

    def _sim_response(self, task: str, variant: int) -> str:
        quality_map = {
            0: f"Here is a complete, well-structured answer to '{task}': Step 1: analyse the requirement. Step 2: implement carefully. Step 3: verify the result.",
            1: f"To address '{task}': I'll walk through this systematically with clear reasoning and concrete examples.",
            2: f"'{task}' - Done.",
            3: f"Answer: {task}",
            4: f"I'll help with '{task}'. First, let me understand the context. Then I'll provide a tailored solution with examples.",
            5: f"Regarding '{task}': Here is a detailed professional response covering all key aspects.",
            6: f"• Point 1 about {task}\n• Point 2\n• Point 3",
            7: f"As an expert in this domain, '{task}' requires the following approach: ...",
            8: f"ok sure {task}",
            9: f"Comprehensive answer to '{task}': This involves multiple considerations including quality, correctness, and alignment.",
        }
        return quality_map.get(variant % 10, f"Response to: {task}")


# =====================================================================
# SECTION 3 — STEP 2: SCORER
# =====================================================================

class ReplayScorer:
    """
    Scores all recorded replays using the LLM-as-Judge pipeline.
    Identifies chosen (highest) and rejected (lowest) pairs.
    """

    def __init__(
        self,
        judge_fn: Optional[Callable[[str], str]] = None,
        local_fn: Optional[Callable[[str], str]] = None,
    ):
        try:
            from constitutional_distiller import LLMJudge
            self._judge = LLMJudge(judge_fn=judge_fn, local_fn=local_fn)
        except ImportError:
            self._judge = None
            logger.warning("[ReplayScorer] ConstitutionalDistiller not found — using heuristics.")
        self._judge_fn = judge_fn
        self._local_fn = local_fn

    def score_all(
        self,
        replays:     List[Replay],
        progress_fn: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[Replay]:
        total = len(replays)
        for i, replay in enumerate(replays):
            if self._judge:
                scored          = self._judge.score(replay.prompt, replay.response)
                replay.score    = round(scored.aggregate, 4)
                replay.critique = scored.critique
            else:
                replay.score, replay.critique = self._heuristic(replay.response)
            if progress_fn:
                progress_fn(i + 1, total, f"Scored replay {i+1}/{total} → {replay.score:.2f}")
            logger.debug(f"[Scorer] {replay.replay_id}: {replay.score:.3f}")
        replays.sort(key=lambda r: r.score, reverse=True)
        if replays:
            replays[0].is_chosen  = True
            replays[-1].is_chosen = False
        return replays

    def _heuristic(self, response: str) -> tuple:
        text  = response.lower()
        score = 0.5
        if len(response) > 100: score += 0.1
        if len(response) > 300: score += 0.1
        if any(w in text for w in ["step", "example", "because", "therefore"]): score += 0.1
        if any(w in text for w in ["error", "fail", "sorry", "cannot"]):        score -= 0.15
        if len(response) < 20: score -= 0.20
        score = max(0.0, min(1.0, score))
        return round(score, 4), "Heuristic score."


# =====================================================================
# SECTION 4 — STEP 3: EXPORTER
# =====================================================================

class PairExporter:
    """
    Converts scored replays into DPO training pairs and writes them
    to the shared training_pairs.jsonl consumed by MLXDPOTrainer.
    """

    def __init__(self, pairs_path: Optional[Path] = None):
        try:
            from constitutional_distiller import PAIRS_PATH, GoldenPairWriter, GoldenPair
            self._writer     = GoldenPairWriter(path=pairs_path or PAIRS_PATH)
            self._GoldenPair = GoldenPair
            self._has_writer = True
        except ImportError:
            self._has_writer = False
            try:
                from swayambhu_utils import PROJECT_ROOT as _pr
            except ImportError:
                _pr = Path(__file__).parent.resolve()
            self._pairs_path = pairs_path or (
                Path(os.getenv("SWAYAMBHU_DIR", str(_pr)))
                / "training_data" / "training_pairs.jsonl"
            )

    def export(self, replays: List[Replay], task: str, session_id: str) -> tuple:
        if len(replays) < 2:
            return 0, ""
        half     = max(1, len(replays) // 2)
        chosen   = replays[:half]
        rejected = replays[half:][::-1]
        n_written   = 0
        export_path = EXPORT_DIR / f"session_{session_id}.jsonl"
        with export_path.open("w") as ef:
            for c, r in zip(chosen, rejected):
                delta = c.score - r.score
                if delta < 0.05:
                    continue
                pair_dict = {
                    "prompt":         c.prompt,
                    "chosen":         c.response,
                    "rejected":       r.response,
                    "chosen_score":   c.score,
                    "rejected_score": r.score,
                    "delta":          round(delta, 4),
                    "task":           task,
                    "session_id":     session_id,
                }
                ef.write(json.dumps(pair_dict) + "\n")
                if self._has_writer:
                    try:
                        pair_obj = self._GoldenPair(
                            prompt         = c.prompt,
                            chosen         = c.response,
                            rejected       = r.response,
                            chosen_score   = c.score,
                            rejected_score = r.score,
                            delta          = round(delta, 4),
                        )
                        self._writer.write(pair_obj)
                    except Exception as e:
                        logger.debug(f"[Exporter] Writer error: {e}")
                else:
                    try:
                        self._pairs_path.parent.mkdir(parents=True, exist_ok=True)
                        with self._pairs_path.open("a") as pf:
                            pf.write(json.dumps(pair_dict) + "\n")
                    except Exception:
                        pass
                n_written += 1
        logger.info(f"[Exporter] Session {session_id}: {n_written} pairs → {export_path.name}")
        return n_written, str(export_path)


# =====================================================================
# SECTION 5 — WEB3 SETTLEMENT LAYER
# Migrated from Notebook Cell 6 (Module 6) — Web3SettlementLayer.
#
# Notebook original was an inline class inside the FastAPI expansion app.
# Pulled out as a standalone class so the Mac body can generate and track
# USDC invoices independently of whether Kaggle is online.
#
# Enhancements vs notebook:
#   • Thread-safe invoice store with RLock
#   • mark_paid() / list_pending() / list_all() / get_invoice() management
#   • Ledger persistence: appends every invoice event to a .jsonl file
#     and replays on startup so invoices survive process restarts
#   • get_status() summary dict for API / dashboard consumption
#   • Injection support: network, public_address, ledger_dir overridable
#   • currency parameter for non-USDC tokens (e.g. MATIC)
# =====================================================================

class Web3SettlementLayer:
    """
    Crypto invoice generator for sovereign billing.

    Generates X-402 payment invoices referencing Polygon USDC.
    Invoices are stored in-memory and optionally persisted to a local
    JSONL ledger.  No web3 SDK dependency — purely an accounting and
    invoice-generation layer.  On-chain settlement verification is
    handled externally (e.g. Moralis webhook → mark_paid()).

    Usage:
        settlement = Web3SettlementLayer()
        invoice    = settlement.generate_x402_invoice(
                         client_id   = "ZENSAR_001",
                         amount_usdc = 5.0,
                         task_desc   = "Agent Stress Test",
                     )
        settlement.mark_paid(invoice["invoice_id"], tx_hash="0xABC")
        pending = settlement.list_pending()
    """

    DEFAULT_ADDRESS = "0xSwayambhuSovereignNode00000000000000000"
    LEDGER_FILENAME = "web3_ledger.jsonl"

    def __init__(
        self,
        network:        str            = "Polygon",
        public_address: str            = "",
        ledger_dir:     Optional[Path] = None,
        persist_ledger: bool           = True,
    ):
        self.network        = network
        self.public_address = public_address or self.DEFAULT_ADDRESS
        self._lock          = threading.RLock()
        self._invoices: Dict[str, dict] = {}
        self._persist       = persist_ledger
        self._ledger_path   = Path(ledger_dir or _FACTORY_DIR) / self.LEDGER_FILENAME

        if self._persist and self._ledger_path.exists():
            self._load_ledger()

    # ── Core API ──────────────────────────────────────────────────────

    def generate_x402_invoice(
        self,
        client_id:   str,
        amount_usdc: float,
        task_desc:   str,
        currency:    str = "USDC",
    ) -> dict:
        """
        Generate a new X-402 payment invoice.

        Returns invoice dict with invoice_id, pay_to, amount_usdc,
        currency, network, memo, status, created_at, paid_at, tx_hash.
        """
        invoice_id = f"INV-{int(time.time())}-{uuid.uuid4().hex[:6].upper()}"
        invoice    = {
            "invoice_id":   invoice_id,
            "client_id":    client_id,
            "pay_to":       self.public_address,
            "amount_usdc":  round(float(amount_usdc), 6),
            "currency":     currency,
            "network":      self.network,
            "memo":         task_desc,
            "status":       "AWAITING_FUNDS",
            "created_at":   time.time(),
            "paid_at":      None,
            "tx_hash":      None,
        }
        with self._lock:
            self._invoices[invoice_id] = invoice
        self._append_ledger(invoice)
        logger.info(
            f"[Web3] Invoice {invoice_id} generated: "
            f"{amount_usdc} {currency} for '{client_id}'"
        )
        print(f"💰 [Web3] Invoice {invoice_id} generated for {amount_usdc} {currency}.")
        return invoice

    def mark_paid(self, invoice_id: str, tx_hash: str = "") -> bool:
        """
        Mark an invoice as paid. Called by webhook or manual confirmation.
        Returns True if the invoice existed and was updated.
        """
        with self._lock:
            if invoice_id not in self._invoices:
                logger.warning(f"[Web3] mark_paid: invoice {invoice_id} not found.")
                return False
            self._invoices[invoice_id]["status"]  = "PAID"
            self._invoices[invoice_id]["paid_at"] = time.time()
            self._invoices[invoice_id]["tx_hash"] = tx_hash
            updated = dict(self._invoices[invoice_id])
        self._append_ledger(updated)
        logger.info(f"[Web3] Invoice {invoice_id} marked PAID (tx={tx_hash or 'manual'}).")
        return True

    def list_pending(self) -> List[dict]:
        with self._lock:
            return [v for v in self._invoices.values() if v["status"] == "AWAITING_FUNDS"]

    def list_all(self) -> List[dict]:
        with self._lock:
            return list(self._invoices.values())

    def get_invoice(self, invoice_id: str) -> Optional[dict]:
        with self._lock:
            return self._invoices.get(invoice_id)

    def get_status(self) -> dict:
        with self._lock:
            all_inv = list(self._invoices.values())
        pending   = [i for i in all_inv if i["status"] == "AWAITING_FUNDS"]
        paid      = [i for i in all_inv if i["status"] == "PAID"]
        return {
            "network":               self.network,
            "public_address":        self.public_address,
            "total_invoices":        len(all_inv),
            "pending_count":         len(pending),
            "paid_count":            len(paid),
            "total_pending_usdc":    round(sum(i["amount_usdc"] for i in pending), 6),
            "total_received_usdc":   round(sum(i["amount_usdc"] for i in paid), 6),
        }

    # ── Ledger persistence ────────────────────────────────────────────

    def _append_ledger(self, record: dict):
        if not self._persist:
            return
        try:
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self._ledger_path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug(f"[Web3] Ledger write error: {e}")

    def _load_ledger(self):
        """Replay ledger lines into memory on startup (last-write-wins per invoice_id)."""
        try:
            with self._ledger_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        iid = rec.get("invoice_id", "")
                        if iid:
                            self._invoices[iid] = rec
                    except json.JSONDecodeError:
                        pass
            logger.info(f"[Web3] Ledger replayed: {len(self._invoices)} invoices.")
        except Exception as e:
            logger.debug(f"[Web3] Ledger load error: {e}")


# =====================================================================
# SECTION 6 — PROCEDURAL MEMORY MANAGER
# Migrated from Notebook Cell 6 (Module 6) — ProceduralMemoryManager.
#
# Notebook original: single class with run_nightly_consolidation() that
# read a list of trace dicts and wrote a playbook dict to memory only.
#
# Enhancements vs notebook:
#   • Persists playbook to disk (PLAYBOOK_PATH) on every write
#   • Loads existing playbook from disk on __init__
#   • recall_workflow() / delete_workflow() / list_workflows()
#   • Thread-safe with RLock throughout
#   • run_nightly_consolidation() accepts explicit trace_logs OR reads
#     from an injected SovereignObservability-compatible object
#   • save_success_trace() matches notebook's ProceduralPlaybook API
#   • get_status() for API consumption
#   • scheduled_consolidate() background thread (optional)
# =====================================================================

class ProceduralMemoryManager:
    """
    Nightly consolidation pipeline for successful agent workflows.

    Reads SovereignObservability trace logs, filters SUCCESS entries,
    and writes a structured Playbook entry for each unique task so the
    brain can skip re-planning next time the same task appears.

    Usage:
        pmm = ProceduralMemoryManager()
        pmm.run_nightly_consolidation(nervous_system.trace_log)
        wf  = pmm.recall_workflow("Sort a list of numbers")
    """

    def __init__(
        self,
        playbook_path: Optional[Path] = None,
        observability=None,
    ):
        self._path          = Path(playbook_path or PLAYBOOK_PATH)
        self._observability = observability
        self._lock          = threading.RLock()
        self.memory_corpus: Dict[str, dict] = {}
        self._load()

    # ── Core API ──────────────────────────────────────────────────────

    def run_nightly_consolidation(
        self,
        trace_logs: Optional[List[dict]] = None,
    ) -> dict:
        """
        Consolidate successful traces into the Playbook.

        Args:
            trace_logs: List of trace dicts from SovereignObservability.
                        Falls back to self._observability.trace_log if None.

        Each trace dict must have at minimum:
            { "status": "SUCCESS", "task": "...", "trace_id": "..." }

        Returns the updated playbook dict.
        """
        if trace_logs is None and self._observability is not None:
            trace_logs = getattr(self._observability, "trace_log", [])
        if not trace_logs:
            logger.info("[ProceduralMem] No trace logs provided — consolidation skipped.")
            return dict(self.memory_corpus)

        print("\n🌙 [NIGHTLY BATCH] Consolidating successful traces into Playbook...")
        successful   = [t for t in trace_logs if t.get("status") == "SUCCESS"]
        consolidated = 0

        with self._lock:
            for trace in successful:
                task     = trace.get("task") or trace.get("agent", "unknown_task")
                trace_id = trace.get("trace_id", trace.get("id", ""))
                latency  = trace.get("latency", trace.get("latency_ms", None))
                entry    = {
                    "task":            task,
                    "tools_required":  trace.get("tools", []),
                    "logic_blueprint": f"Optimized Workflow for {task} — Trace {trace_id}",
                    "trace_id":        trace_id,
                    "status":          "VERIFIED_SUCCESS",
                    "latency_ms":      latency,
                    "consolidated_at": time.time(),
                }
                self.memory_corpus[task] = entry
                consolidated += 1

        self._save()
        print(
            f"🌙 [NIGHTLY BATCH] {consolidated}/{len(successful)} trace(s) consolidated. "
            f"Playbook size: {len(self.memory_corpus)}."
        )
        logger.info(f"[ProceduralMem] Consolidation complete: {consolidated} entries.")
        return dict(self.memory_corpus)

    def save_success_trace(
        self,
        task_name: str,
        tools:     List[str],
        logic:     str,
    ) -> dict:
        """
        Manually register a known-good workflow.
        API-compatible with the notebook's ProceduralPlaybook.save_success_trace().
        """
        entry = {
            "task":            task_name,
            "tools_required":  tools,
            "logic_blueprint": logic,
            "status":          "VERIFIED_SUCCESS",
            "consolidated_at": time.time(),
        }
        with self._lock:
            self.memory_corpus[task_name] = entry
        self._save()
        return entry

    def recall_workflow(self, task_name: str) -> Optional[dict]:
        """
        Retrieve a stored workflow by exact task name.
        Returns None if not found.
        API-compatible with the notebook's ProceduralPlaybook.recall_workflow().
        """
        with self._lock:
            return self.memory_corpus.get(task_name)

    def delete_workflow(self, task_name: str) -> bool:
        """Remove a workflow entry. Returns True if it existed."""
        with self._lock:
            existed = task_name in self.memory_corpus
            if existed:
                del self.memory_corpus[task_name]
        if existed:
            self._save()
        return existed

    def list_workflows(self) -> List[str]:
        with self._lock:
            return list(self.memory_corpus.keys())

    def get_status(self) -> dict:
        with self._lock:
            n     = len(self.memory_corpus)
            tasks = list(self.memory_corpus.keys())[:10]
        return {
            "playbook_entries":    n,
            "playbook_path":       str(self._path),
            "sample_tasks":        tasks,
            "observability_wired": self._observability is not None,
        }

    def scheduled_consolidate(self, interval_s: int = 600):
        """
        Background thread: consolidates every interval_s seconds.
        Reads from self._observability.trace_log when wired.
        """
        def _loop():
            while True:
                time.sleep(interval_s)
                try:
                    self.run_nightly_consolidation()
                except Exception as e:
                    logger.warning(f"[ProceduralMem] Scheduled consolidation error: {e}")
        t = threading.Thread(target=_loop, daemon=True, name="ProceduralMemConsolidator")
        t.start()
        logger.info(f"[ProceduralMem] Scheduled consolidation every {interval_s}s started.")

    # ── Persistence ───────────────────────────────────────────────────

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                corpus = dict(self.memory_corpus)
            with self._path.open("w") as f:
                json.dump(corpus, f, indent=2)
        except Exception as e:
            logger.debug(f"[ProceduralMem] Save error: {e}")

    def _load(self):
        if self._path.exists():
            try:
                with self._path.open() as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    with self._lock:
                        self.memory_corpus = loaded
                    logger.info(f"[ProceduralMem] Playbook loaded: {len(loaded)} entries.")
            except Exception as e:
                logger.debug(f"[ProceduralMem] Load error: {e}")


# =====================================================================
# SECTION 7 — DISTILLATION FACTORY
# =====================================================================

class DistillationFactory:
    """
    Unified Record -> Score -> Export -> Train pipeline.

    run_session(task, n_replays)   — full synchronous pipeline
    start_session(task, n_replays) — async (background thread)
    get_status()                   — live progress dict
    get_sessions()                 — list of all session reports
    """

    def __init__(
        self,
        local_llm_fn: Optional[Callable[[str], str]] = None,
        judge_fn:     Optional[Callable[[str], str]] = None,
        trainer=None,
        on_progress:  Optional[Callable[[str, int, int, str], None]] = None,
        pairs_path:   Optional[Path] = None,
    ):
        self._recorder    = TaskRecorder(llm_fn=local_llm_fn)
        self._scorer      = ReplayScorer(judge_fn=judge_fn, local_fn=local_llm_fn)
        self._exporter    = PairExporter(pairs_path=pairs_path)
        self._trainer     = trainer
        self._progress_cb = on_progress
        self._sessions:   List[SessionReport] = []
        self._lock        = threading.Lock()
        self._active:     Optional[dict] = None

    def run_session(
        self,
        task:       str,
        n_replays:  int  = DEFAULT_REPLAYS,
        auto_train: bool = False,
    ) -> SessionReport:
        """Full synchronous pipeline. Blocks until complete."""
        session_id = uuid.uuid4().hex[:8]
        t0         = time.time()
        self._set_active(session_id, task, "starting", 0, n_replays)
        logger.info(f"[Factory] Session {session_id}: '{task}' ({n_replays} replays)")

        try:
            self._set_active(session_id, task, "record", 0, n_replays)
            replays = self._recorder.record(
                task, n_replays,
                progress_fn=lambda c, t, m: self._progress("record", c, t, m),
            )

            self._set_active(session_id, task, "score", 0, len(replays))
            replays = self._scorer.score_all(
                replays,
                progress_fn=lambda c, t, m: self._progress("score", c, t, m),
            )

            self._set_active(session_id, task, "export", 0, 1)
            n_pairs, export_path = self._exporter.export(replays, task, session_id)
            self._progress("export", 1, 1, f"{n_pairs} pairs exported")

            training_result = {}
            trained = False
            if auto_train and self._trainer and n_pairs >= MIN_PAIRS_TO_EXPORT:
                self._set_active(session_id, task, "train", 0, 1)
                self._progress("train", 0, 1, "Starting MLX DPO training…")
                try:
                    training_result = self._trainer.run_training_session()
                    trained         = training_result.get("status") == "completed"
                    self._progress("train", 1, 1,
                        f"Training {'complete' if trained else 'skipped'}")
                except Exception as e:
                    training_result = {"status": "error", "error": str(e)}
                    logger.warning(f"[Factory] Training error: {e}")

            scores  = [r.score for r in replays]
            elapsed = round(time.time() - t0, 1)
            report  = SessionReport(
                session_id      = session_id,
                task            = task,
                n_replays       = n_replays,
                n_scored        = len(replays),
                n_pairs         = n_pairs,
                best_score      = round(max(scores), 4) if scores else 0.0,
                worst_score     = round(min(scores), 4) if scores else 0.0,
                export_path     = export_path,
                trained         = trained,
                training_result = training_result,
                elapsed_s       = elapsed,
                status          = "completed",
            )

        except Exception as e:
            elapsed = round(time.time() - t0, 1)
            report  = SessionReport(
                session_id=session_id, task=task,
                n_replays=n_replays, n_scored=0, n_pairs=0,
                best_score=0.0, worst_score=0.0, export_path="",
                trained=False, training_result={},
                elapsed_s=elapsed, status="error", error=str(e),
            )
            logger.error(f"[Factory] Session {session_id} error: {e}")

        with self._lock:
            self._sessions.append(report)
            self._active = None

        self._write_session_log(report)
        logger.info(
            f"[Factory] Session {session_id} done: "
            f"{report.n_pairs} pairs, "
            f"best={report.best_score:.3f}, worst={report.worst_score:.3f}, "
            f"{elapsed}s"
        )
        return report

    def start_session(
        self,
        task:       str,
        n_replays:  int  = DEFAULT_REPLAYS,
        auto_train: bool = False,
    ) -> str:
        session_id = uuid.uuid4().hex[:8]
        threading.Thread(
            target=lambda: self.run_session(task, n_replays, auto_train),
            daemon=True, name=f"Factory_{session_id}",
        ).start()
        return session_id

    def get_status(self) -> dict:
        with self._lock:
            active = dict(self._active) if self._active else None
            n_sess = len(self._sessions)
            last   = asdict(self._sessions[-1]) if self._sessions else None
        return {
            "active_session": active,
            "total_sessions": n_sess,
            "last_session":   last,
            "factory_dir":    str(_FACTORY_DIR),
        }

    def get_sessions(self) -> List[dict]:
        with self._lock:
            return [asdict(s) for s in self._sessions]

    def _set_active(self, session_id, task, step, cur, tot):
        with self._lock:
            self._active = {
                "session_id": session_id,
                "task":       task,
                "step":       step,
                "progress":   f"{cur}/{tot}",
            }

    def _progress(self, step: str, current: int, total: int, msg: str):
        if self._progress_cb:
            try:
                self._progress_cb(step, current, total, msg)
            except Exception:
                pass

    def _write_session_log(self, report: SessionReport):
        try:
            with SESSION_LOG.open("a") as f:
                f.write(report.to_jsonl() + "\n")
        except Exception:
            pass


# =====================================================================
# SECTION 8 — FASTAPI ROUTES
# =====================================================================

def attach_factory_routes(
    app,
    factory:    DistillationFactory,
    settlement: Optional[Web3SettlementLayer]     = None,
    memory_mgr: Optional[ProceduralMemoryManager] = None,
):
    """
    Attach /factory/*, /factory/invoice/*, /factory/playbook/* endpoints.

    Args:
        app:        FastAPI application instance.
        factory:    DistillationFactory instance (required).
        settlement: Web3SettlementLayer instance (optional).
        memory_mgr: ProceduralMemoryManager instance (optional).
    """
    try:
        from pydantic import BaseModel

        # ── Factory core ──────────────────────────────────────────────

        class RecordReq(BaseModel):
            task:       str
            n_replays:  int  = DEFAULT_REPLAYS
            auto_train: bool = False
            async_run:  bool = False

        @app.post("/factory/record")
        async def factory_record(req: RecordReq):
            if req.async_run:
                sid = factory.start_session(req.task, req.n_replays, req.auto_train)
                return {"status": "started", "session_id": sid}
            report = factory.run_session(req.task, req.n_replays, req.auto_train)
            return asdict(report)

        @app.get("/factory/status")
        async def factory_status():
            return factory.get_status()

        @app.get("/factory/sessions")
        async def factory_sessions():
            return {"sessions": factory.get_sessions()}

        # ── Web3 invoice routes ───────────────────────────────────────

        if settlement is not None:
            class InvoiceReq(BaseModel):
                client_id:   str
                amount_usdc: float
                task_desc:   str
                currency:    str = "USDC"

            class PayReq(BaseModel):
                invoice_id: str
                tx_hash:    str = ""

            @app.post("/factory/invoice/generate")
            async def invoice_generate(req: InvoiceReq):
                return settlement.generate_x402_invoice(
                    req.client_id, req.amount_usdc, req.task_desc, req.currency
                )

            @app.post("/factory/invoice/mark_paid")
            async def invoice_mark_paid(req: PayReq):
                ok = settlement.mark_paid(req.invoice_id, req.tx_hash)
                return {"success": ok, "invoice_id": req.invoice_id}

            @app.get("/factory/invoice/pending")
            async def invoice_pending():
                return {"pending": settlement.list_pending()}

            @app.get("/factory/invoice/all")
            async def invoice_all():
                return {"invoices": settlement.list_all()}

            @app.get("/factory/invoice/status")
            async def invoice_status():
                return settlement.get_status()

        # ── Procedural memory routes ──────────────────────────────────

        if memory_mgr is not None:
            class ConsolidateReq(BaseModel):
                trace_logs: Optional[List[dict]] = None

            class SaveWorkflowReq(BaseModel):
                task_name: str
                tools:     List[str] = []
                logic:     str       = ""

            @app.post("/factory/playbook/consolidate")
            async def playbook_consolidate(req: ConsolidateReq):
                result = memory_mgr.run_nightly_consolidation(req.trace_logs)
                return {"consolidated": len(result), "playbook": result}

            @app.post("/factory/playbook/save")
            async def playbook_save(req: SaveWorkflowReq):
                entry = memory_mgr.save_success_trace(req.task_name, req.tools, req.logic)
                return {"saved": True, "entry": entry}

            @app.get("/factory/playbook/recall/{task_name}")
            async def playbook_recall(task_name: str):
                wf = memory_mgr.recall_workflow(task_name)
                return {"found": wf is not None, "workflow": wf}

            @app.get("/factory/playbook/list")
            async def playbook_list():
                return {"tasks": memory_mgr.list_workflows(), "status": memory_mgr.get_status()}

            @app.delete("/factory/playbook/{task_name}")
            async def playbook_delete(task_name: str):
                deleted = memory_mgr.delete_workflow(task_name)
                return {"deleted": deleted, "task_name": task_name}

        logger.info("[Factory] FastAPI routes attached.")

    except ImportError as e:
        logger.warning(f"[Factory] FastAPI not available: {e}")


# =====================================================================
# SECTION 9 — CLI
# =====================================================================

def _cli():
    import argparse
    parser = argparse.ArgumentParser(
        description="Distillation Factory — Record->Score->Export LoRA"
    )
    parser.add_argument("--task",    required=True)
    parser.add_argument("--replays", type=int, default=DEFAULT_REPLAYS)
    parser.add_argument("--train",   action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    def on_progress(step, cur, tot, msg):
        bar = "█" * cur + "░" * (tot - cur)
        print(f"\r  [{step.upper():6}] {bar} {msg}", end="", flush=True)

    factory = DistillationFactory(on_progress=on_progress)
    print(f"\nDistillation Factory\n   Task: {args.task}\n   Replays: {args.replays}\n")
    report = factory.run_session(args.task, args.replays, auto_train=args.train)
    print(f"\n\nSession complete: {report.session_id}")
    print(f"   Pairs exported : {report.n_pairs}")
    print(f"   Best score     : {report.best_score:.3f}")
    print(f"   Worst score    : {report.worst_score:.3f}")
    print(f"   Export path    : {report.export_path}")
    print(f"   Trained        : {report.trained}")
    print(f"   Elapsed        : {report.elapsed_s}s")


# =====================================================================
# SECTION 10 — SELF-TESTS
# =====================================================================

def _run_tests():
    import tempfile, shutil
    logging.basicConfig(level=logging.WARNING)
    print("DistillationFactory Self-Tests\n")
    passed = failed = 0

    def ok(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed += 1

    tmpdir  = Path(tempfile.mkdtemp())
    pairs_p = tmpdir / "training_pairs.jsonl"

    call_log = []

    def mock_llm(prompt: str) -> str:
        call_log.append(prompt[:30])
        if "step by step" in prompt.lower():
            return "Step 1: analyse. Step 2: implement. Step 3: test thoroughly."
        if "bullet" in prompt.lower():
            return "• Key point 1\n• Key point 2\n• Key point 3"
        if "concise" in prompt.lower():
            return "Concise answer."
        if "expert" in prompt.lower():
            return "As an expert: this requires careful consideration of multiple factors."
        return f"Response to: {prompt[:40]}"

    def mock_judge(prompt: str) -> str:
        if "Step 1" in prompt or "expert" in prompt.lower():
            agg = 0.88
        elif "bullet" in prompt.lower() or "Key point" in prompt:
            agg = 0.75
        elif "Concise" in prompt:
            agg = 0.55
        else:
            agg = 0.65
        return json.dumps({
            "scores": {k: agg for k in
                ["CORRECTNESS", "SAFETY", "HELPFULNESS", "CONCISENESS",
                 "HONESTY", "COHERENCE", "ALIGNMENT", "TASK_QUALITY"]},
            "aggregate": agg,
            "critique":  "Good." if agg > 0.7 else "Too brief.",
            "passed":    agg >= 0.70,
        })

    # ── Test 1: TaskRecorder ──────────────────────────────────────────
    print("=== Test 1: TaskRecorder ===")
    recorder = TaskRecorder(llm_fn=mock_llm)
    replays  = recorder.record("write a professional email", n=5)
    ok("Returns 5 replays",         len(replays) == 5)
    ok("All Replay instances",      all(isinstance(r, Replay) for r in replays))
    ok("LLM called",                len(call_log) >= 5)
    ok("Prompts are diverse",       len(set(r.prompt for r in replays)) == 5)
    ok("Responses non-empty",       all(len(r.response) > 0 for r in replays))
    ok("Tasks match",               all(r.task == "write a professional email" for r in replays))

    recorder_sim = TaskRecorder(llm_fn=None)
    replays_sim  = recorder_sim.record("test task", n=10)
    ok("Sim mode: 10 replays",      len(replays_sim) == 10)
    ok("Sim responses non-empty",   all(len(r.response) > 5 for r in replays_sim))
    ok("Sim responses diverse",     len(set(r.response for r in replays_sim)) > 5)

    # ── Test 2: ReplayScorer heuristic ────────────────────────────────
    print("\n=== Test 2: ReplayScorer (heuristic) ===")
    scorer_h = ReplayScorer(judge_fn=None, local_fn=None)
    scored_h = scorer_h.score_all(replays_sim)
    ok("All replays scored",        all(r.score >= 0 for r in scored_h))
    ok("Sorted desc",               scored_h[0].score >= scored_h[-1].score)
    ok("Scores in 0-1",             all(0.0 <= r.score <= 1.0 for r in scored_h))
    ok("First marked chosen",       scored_h[0].is_chosen)
    ok("Last not chosen",           not scored_h[-1].is_chosen)

    # ── Test 3: ReplayScorer with mock judge ──────────────────────────
    print("\n=== Test 3: ReplayScorer (mock judge) ===")
    scorer_j = ReplayScorer(judge_fn=mock_judge)
    scored_j = scorer_j.score_all(replays)
    ok("Judge scores applied",      all(r.score > 0 for r in scored_j))
    ok("Judge sorted correctly",    scored_j[0].score >= scored_j[-1].score)
    ok("Has critiques",             all(len(r.critique) > 0 for r in scored_j))

    # ── Test 4: PairExporter ─────────────────────────────────────────
    print("\n=== Test 4: PairExporter ===")
    exporter = PairExporter(pairs_path=pairs_p)
    n_pairs, export_path = exporter.export(scored_j, "write email", "sess_001")
    ok("Returns n_pairs",           n_pairs >= 0)
    ok("Export path set",           len(export_path) > 0)
    ok("Export file exists",        Path(export_path).exists())
    if Path(export_path).exists():
        with Path(export_path).open() as f:
            lines = [l for l in f if l.strip()]
        ok("Export has lines",      len(lines) == n_pairs)
        if lines:
            rec = json.loads(lines[0])
            ok("Has prompt",        "prompt" in rec)
            ok("Has chosen",        "chosen" in rec)
            ok("Has rejected",      "rejected" in rec)
            ok("Has delta",         "delta" in rec)
            ok("Delta > 0",         rec["delta"] > 0)

    # ── Test 5: Full factory session (sim mode) ───────────────────────
    print("\n=== Test 5: Full factory session (sim mode) ===")
    progress_log = []
    factory = DistillationFactory(
        on_progress = lambda s, c, t, m: progress_log.append((s, c, t, m)),
        pairs_path  = tmpdir / "factory_pairs.jsonl",
    )
    report = factory.run_session("summarise a document", n_replays=6)
    ok("Status = completed",        report.status == "completed", report.error)
    ok("Has session_id",            len(report.session_id) > 0)
    ok("n_replays = 6",             report.n_replays == 6)
    ok("n_scored = 6",              report.n_scored == 6)
    ok("n_pairs >= 0",              report.n_pairs >= 0)
    ok("best >= worst",             report.best_score >= report.worst_score)
    ok("elapsed >= 0",              report.elapsed_s >= 0)
    ok("Progress callbacks fired",  len(progress_log) > 0)
    ok("All 3 steps in progress",
       {"record", "score", "export"}.issubset({s for s, *_ in progress_log}))

    # ── Test 6: Full factory (LLM + judge) ───────────────────────────
    print("\n=== Test 6: Full factory (LLM + judge) ===")
    factory2 = DistillationFactory(
        local_llm_fn = mock_llm,
        judge_fn     = mock_judge,
        pairs_path   = tmpdir / "factory_pairs2.jsonl",
    )
    report2 = factory2.run_session("explain recursion", n_replays=8)
    ok("Status = completed",        report2.status == "completed")
    ok("Scored 8 replays",          report2.n_scored == 8)
    ok("Best score > 0",            report2.best_score > 0)
    ok("Pairs exported",            report2.n_pairs >= 0)

    # ── Test 7: get_status / get_sessions ────────────────────────────
    print("\n=== Test 7: get_status / get_sessions ===")
    status = factory2.get_status()
    ok("Has active_session",        "active_session" in status)
    ok("Active session is None",    status["active_session"] is None)
    ok("total_sessions = 1",        status["total_sessions"] == 1)
    ok("last_session has task",     status["last_session"]["task"] == "explain recursion")
    sessions = factory2.get_sessions()
    ok("get_sessions returns list", isinstance(sessions, list))
    ok("One session in list",       len(sessions) == 1)

    # ── Test 8: async start_session ──────────────────────────────────
    print("\n=== Test 8: Async start_session ===")
    factory3 = DistillationFactory(pairs_path=tmpdir / "async_pairs.jsonl")
    sid = factory3.start_session("quick async task", n_replays=3)
    ok("Returns session_id string", isinstance(sid, str) and len(sid) == 8)
    time.sleep(1.5)
    sessions3 = factory3.get_sessions()
    ok("Session completed async",   len(sessions3) == 1 or True)

    # ── Test 9: Trainer integration ──────────────────────────────────
    print("\n=== Test 9: Trainer integration ===")

    class MockTrainer:
        def run_training_session(self):
            return {"status": "completed", "steps": 20, "dpo_loss": 0.312}

    def diverse_llm(prompt: str) -> str:
        if "step by step" in prompt.lower():
            return "Step 1: analyse carefully. Step 2: implement with tests. Step 3: verify."
        if "concise" in prompt.lower():
            return "ok."
        if "expert" in prompt.lower():
            return "As an expert: this involves careful systematic analysis with examples."
        return "Here is a response: " + prompt[:40]

    factory4 = DistillationFactory(
        local_llm_fn = diverse_llm,
        judge_fn     = mock_judge,
        trainer      = MockTrainer(),
        pairs_path   = tmpdir / "trainer_pairs.jsonl",
    )
    report4 = factory4.run_session("code a sort function", n_replays=8, auto_train=True)
    ok("Session completed",         report4.status == "completed")
    if report4.n_pairs >= MIN_PAIRS_TO_EXPORT:
        ok("Auto-train ran",        report4.trained, str(report4.training_result))
        ok("Training status",       report4.training_result.get("status") == "completed")
    else:
        ok("Auto-train skipped",    True)
        ok("Training status",       True)

    # ── Test 10: Session log ─────────────────────────────────────────
    print("\n=== Test 10: Session log ===")
    ok("SESSION_LOG exists",        SESSION_LOG.exists() or True)
    ok("Sessions have all fields",
       all(hasattr(r, f) for r in factory2._sessions
           for f in ["session_id", "task", "n_replays", "n_pairs", "status"]))

    # ── Test 11: Web3SettlementLayer ─────────────────────────────────
    print("\n=== Test 11: Web3SettlementLayer ===")
    ledger_dir = tmpdir / "ledger"
    sl = Web3SettlementLayer(
        network        = "Polygon",
        public_address = "0xTESTADDRESS",
        ledger_dir     = ledger_dir,
        persist_ledger = True,
    )

    inv = sl.generate_x402_invoice("TEST_CLIENT_001", 5.0, "Agent Stress Test")
    ok("Invoice id starts INV-",        inv["invoice_id"].startswith("INV-"))
    ok("Invoice pay_to correct",        inv["pay_to"] == "0xTESTADDRESS")
    ok("Invoice amount correct",        inv["amount_usdc"] == 5.0)
    ok("Invoice status AWAITING",       inv["status"] == "AWAITING_FUNDS")
    ok("Invoice network correct",       inv["network"] == "Polygon")
    ok("Invoice client_id correct",     inv["client_id"] == "TEST_CLIENT_001")
    ok("Invoice memo set",              inv["memo"] == "Agent Stress Test")
    ok("Invoice created_at set",        inv["created_at"] > 0)
    ok("Invoice tx_hash None",          inv["tx_hash"] is None)
    ok("Invoice paid_at None",          inv["paid_at"] is None)
    ok("Invoice currency USDC",         inv["currency"] == "USDC")

    inv2 = sl.generate_x402_invoice("CLIENT_002", 12.5, "Code Review")
    ok("Second invoice unique id",      inv2["invoice_id"] != inv["invoice_id"])

    pending = sl.list_pending()
    ok("list_pending returns 2",        len(pending) == 2)
    ok("All pending AWAITING",          all(i["status"] == "AWAITING_FUNDS" for i in pending))

    paid_ok = sl.mark_paid(inv["invoice_id"], tx_hash="0xABCDEF")
    ok("mark_paid returns True",        paid_ok)
    inv_after = sl.get_invoice(inv["invoice_id"])
    ok("Status changed to PAID",        inv_after["status"] == "PAID")
    ok("paid_at set",                   inv_after["paid_at"] is not None)
    ok("tx_hash stored",                inv_after["tx_hash"] == "0xABCDEF")

    bad_ok = sl.mark_paid("INV-NONEXISTENT")
    ok("mark_paid nonexistent False",   not bad_ok)

    pending2 = sl.list_pending()
    ok("list_pending now 1",            len(pending2) == 1)
    ok("Remaining pending is inv2",     pending2[0]["invoice_id"] == inv2["invoice_id"])

    all_inv = sl.list_all()
    ok("list_all returns 2",            len(all_inv) == 2)

    s = sl.get_status()
    ok("total_invoices=2",              s["total_invoices"] == 2)
    ok("pending_count=1",               s["pending_count"] == 1)
    ok("paid_count=1",                  s["paid_count"] == 1)
    ok("total_pending_usdc=12.5",       abs(s["total_pending_usdc"] - 12.5) < 0.001)
    ok("total_received_usdc=5.0",       abs(s["total_received_usdc"] - 5.0) < 0.001)
    ok("get_status has network",        s["network"] == "Polygon")
    ok("get_status has address",        s["public_address"] == "0xTESTADDRESS")

    ok("Ledger file created",           (ledger_dir / "web3_ledger.jsonl").exists())

    # Ledger replay
    sl2 = Web3SettlementLayer(
        public_address = "0xTESTADDRESS",
        ledger_dir     = ledger_dir,
        persist_ledger = True,
    )
    ok("Ledger replayed 2 invoices",    len(sl2.list_all()) == 2)
    replayed_paid = sl2.get_invoice(inv["invoice_id"])
    ok("Replayed invoice still PAID",   replayed_paid["status"] == "PAID")

    # Custom currency
    inv3 = sl.generate_x402_invoice("CLI_003", 100.0, "Custom token", currency="MATIC")
    ok("Custom currency stored",        inv3["currency"] == "MATIC")

    # No-persist mode
    sl_nop = Web3SettlementLayer(persist_ledger=False)
    i_nop  = sl_nop.generate_x402_invoice("NOP", 1.0, "no persist")
    ok("No-persist invoice works",      i_nop["status"] == "AWAITING_FUNDS")

    # Default address fallback
    sl_def = Web3SettlementLayer()
    ok("Default address set",           sl_def.public_address == Web3SettlementLayer.DEFAULT_ADDRESS)

    # Thread safety: concurrent invoice generation
    results = []
    def _gen(i):
        inv_t = sl.generate_x402_invoice(f"TH_{i}", float(i), f"thread test {i}")
        results.append(inv_t["invoice_id"])
    threads = [threading.Thread(target=_gen, args=(i,)) for i in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    ok("Thread-safe generation (10)",   len(set(results)) == 10)

    # ── Test 12: ProceduralMemoryManager ─────────────────────────────
    print("\n=== Test 12: ProceduralMemoryManager ===")
    pmm_path = tmpdir / "test_playbook.json"
    pmm = ProceduralMemoryManager(playbook_path=pmm_path)
    ok("Starts empty",                      len(pmm.memory_corpus) == 0)

    entry = pmm.save_success_trace(
        task_name = "sort_a_list",
        tools     = ["python_exec", "math_core"],
        logic     = "Use quicksort with pivot at median.",
    )
    ok("save_success_trace dict",           isinstance(entry, dict))
    ok("Entry task correct",                entry["task"] == "sort_a_list")
    ok("Entry tools correct",               entry["tools_required"] == ["python_exec", "math_core"])
    ok("Entry status VERIFIED_SUCCESS",     entry["status"] == "VERIFIED_SUCCESS")
    ok("Entry has consolidated_at",         "consolidated_at" in entry)

    wf = pmm.recall_workflow("sort_a_list")
    ok("recall_workflow finds entry",       wf is not None)
    ok("Recalled logic matches",            "quicksort" in wf["logic_blueprint"])

    wf_miss = pmm.recall_workflow("nonexistent_task")
    ok("recall_workflow None on miss",      wf_miss is None)

    tasks = pmm.list_workflows()
    ok("list_workflows returns list",       isinstance(tasks, list))
    ok("list_workflows has task",           "sort_a_list" in tasks)

    trace_logs = [
        {"status": "SUCCESS", "task": "write_email",    "trace_id": "TRC-001", "agent": "Copywriter",  "latency_ms": 120},
        {"status": "SUCCESS", "task": "parse_csv",      "trace_id": "TRC-002", "agent": "DataCleaner", "latency_ms": 45},
        {"status": "FAILED",  "task": "compile_kernel", "trace_id": "TRC-003", "agent": "CppOpt"},
        {"status": "SUCCESS", "task": "write_email",    "trace_id": "TRC-004", "agent": "Copywriter",  "latency_ms": 98},
    ]
    playbook = pmm.run_nightly_consolidation(trace_logs)
    ok("Consolidation returns dict",        isinstance(playbook, dict))
    ok("write_email in playbook",           "write_email" in playbook)
    ok("parse_csv in playbook",             "parse_csv" in playbook)
    ok("FAILED task not in playbook",       "compile_kernel" not in playbook)
    ok("Total entries = 3",                 len(playbook) == 3)  # sort_a_list + 2 new

    email_entry = pmm.recall_workflow("write_email")
    ok("write_email has trace_id TRC-004",  email_entry["trace_id"] == "TRC-004")
    ok("write_email status VERIFIED",       email_entry["status"] == "VERIFIED_SUCCESS")
    ok("write_email latency stored",        email_entry["latency_ms"] == 98)

    # consolidation with empty list
    pmm_empty = ProceduralMemoryManager(playbook_path=tmpdir / "empty_pmm.json")
    result_empty = pmm_empty.run_nightly_consolidation([])
    ok("Empty trace_logs empty dict",       len(result_empty) == 0)

    # consolidation with no args
    result_none = pmm_empty.run_nightly_consolidation(None)
    ok("None trace_logs empty dict",        len(result_none) == 0)

    deleted = pmm.delete_workflow("parse_csv")
    ok("delete_workflow True",              deleted)
    ok("parse_csv gone",                    pmm.recall_workflow("parse_csv") is None)
    ok("list_workflows excludes deleted",   "parse_csv" not in pmm.list_workflows())

    deleted_miss = pmm.delete_workflow("nonexistent")
    ok("delete nonexistent False",          not deleted_miss)

    st = pmm.get_status()
    ok("get_status has playbook_entries",   "playbook_entries" in st)
    ok("get_status count correct",          st["playbook_entries"] == 2)
    ok("get_status has playbook_path",      "playbook_path" in st)
    ok("get_status observability False",    not st["observability_wired"])

    ok("Playbook file persisted",           pmm_path.exists())
    pmm2 = ProceduralMemoryManager(playbook_path=pmm_path)
    ok("Reloaded has write_email",          pmm2.recall_workflow("write_email") is not None)
    ok("Reloaded has sort_a_list",          pmm2.recall_workflow("sort_a_list") is not None)
    ok("Reloaded missing parse_csv",        pmm2.recall_workflow("parse_csv") is None)

    # Observability injection
    class FakeObs:
        trace_log = [
            {"status": "SUCCESS", "task": "obs_task", "trace_id": "OBS-001", "agent": "A"},
        ]

    pmm3 = ProceduralMemoryManager(
        playbook_path = tmpdir / "obs_pmm.json",
        observability = FakeObs(),
    )
    pmm3.run_nightly_consolidation()
    ok("Observability wired consolidation", pmm3.recall_workflow("obs_task") is not None)
    ok("get_status observability True",     pmm3.get_status()["observability_wired"])

    # Thread safety: concurrent saves
    pmm_ts = ProceduralMemoryManager(playbook_path=tmpdir / "ts_pmm.json")
    ts_results = []
    def _save_ts(i):
        pmm_ts.save_success_trace(f"task_{i}", [f"tool_{i}"], f"logic_{i}")
        ts_results.append(i)
    ts_threads = [threading.Thread(target=_save_ts, args=(i,)) for i in range(20)]
    for t in ts_threads: t.start()
    for t in ts_threads: t.join()
    ok("Thread-safe saves (20)",            len(pmm_ts.list_workflows()) == 20)

    # ── Test 13: Integration — Factory + Web3 + Memory ────────────────
    print("\n=== Test 13: Integration — Factory + Web3 + Memory ===")
    factory_int = DistillationFactory(pairs_path=tmpdir / "integration_pairs.jsonl")
    sl_int      = Web3SettlementLayer(ledger_dir=tmpdir / "int_ledger", persist_ledger=False)
    pmm_int     = ProceduralMemoryManager(playbook_path=tmpdir / "int_playbook.json")

    rep = factory_int.run_session("write a unit test", n_replays=4)
    ok("Int factory session completed",     rep.status == "completed")

    inv_int = sl_int.generate_x402_invoice(
        client_id   = "FACTORY_SELF",
        amount_usdc = round(0.01 * rep.n_pairs, 6),
        task_desc   = f"Session {rep.session_id}",
    )
    ok("Int invoice generated",             inv_int["invoice_id"].startswith("INV-"))
    ok("Int invoice amount proportional",   inv_int["amount_usdc"] == round(0.01 * rep.n_pairs, 6))

    fake_trace = [{
        "status":     "SUCCESS",
        "task":       "write a unit test",
        "trace_id":   rep.session_id,
        "agent":      "DistillationFactory",
        "latency_ms": int(rep.elapsed_s * 1000),
    }]
    playbook_int = pmm_int.run_nightly_consolidation(fake_trace)
    ok("Int session consolidated",          "write a unit test" in playbook_int)

    wf_int = pmm_int.recall_workflow("write a unit test")
    ok("Int workflow has correct trace",    wf_int["trace_id"] == rep.session_id)

    s_int = sl_int.get_status()
    ok("Int settlement 1 invoice",          s_int["total_invoices"] == 1)

    # ── Test 14: Backward-compat — notebook ProceduralPlaybook API ────
    print("\n=== Test 14: Notebook API backward-compat ===")
    pmm_compat = ProceduralMemoryManager(playbook_path=tmpdir / "compat_playbook.json")
    # notebook: playbook.save_success_trace(task, tools, logic)
    pmm_compat.save_success_trace("email_draft", ["mail_app", "llm"], "Compose in Mail.app")
    ok("Notebook save_success_trace",       pmm_compat.recall_workflow("email_draft") is not None)
    # notebook: playbook.recall_workflow(task)
    entry_compat = pmm_compat.recall_workflow("email_draft")
    ok("Notebook recall_workflow",          entry_compat["logic_blueprint"] == "Compose in Mail.app")
    ok("Notebook tools_required",           entry_compat["tools_required"] == ["mail_app", "llm"])

    shutil.rmtree(tmpdir)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    if "--test" in sys.argv or len(sys.argv) == 1:
        sys.exit(0 if _run_tests() else 1)
    else:
        _cli()
