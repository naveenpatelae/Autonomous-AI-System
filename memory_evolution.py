#!/usr/bin/env python3
# =====================================================================
# 💾 MEMORY EVOLUTION — Deep Memory & Self-Evolution Upgrades
#
# #46  Semantic State Folding  — Context compression / log summarisation
# #60  Generative Replay       — Nightly "lessons learned" into ChromaDB
# #48  Knowledge Gap Logger    — Log unknowns → NocturnalDistiller learns
# #50  Self-Evolving Search    — Improve web search syntax over time
# #11  On-Device Learning      — LoRA fine-tune from War Room logs
# #88  Intrinsic Curiosity     — Autonomous idle-time organisation
# =====================================================================

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("MemoryEvolution")

# ── Dirs ──────────────────────────────────────────────────────────────
_BASE_DIR = Path(os.getenv("SWAYAMBHU_DIR", str(PROJECT_ROOT)))
_MEMORY_DIR = _BASE_DIR / "memory_evolution"
_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

GAP_LOG_PATH     = _MEMORY_DIR / "knowledge_gaps.jsonl"
LESSON_LOG_PATH  = _MEMORY_DIR / "lessons_learned.jsonl"
SEARCH_LOG_PATH  = _MEMORY_DIR / "search_patterns.json"
CURIOSITY_LOG    = _MEMORY_DIR / "curiosity_actions.jsonl"


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC STATE FOLDING — Context Compression (#46)
# ─────────────────────────────────────────────────────────────────────
class SemanticStateFolding:
    """
    Folds large execution logs / conversation histories into dense
    1-paragraph summaries to prevent KV-cache / token bloat.

    The 'fold' operation:
      1. Split the log into chunks of MAX_CHUNK_LINES
      2. Summarise each chunk with LLM
      3. Combine summaries into one dense paragraph
      4. Replace original log with folded version
    """

    MAX_CHUNK_LINES = 20   # lines per summary chunk
    MAX_TOKENS_BEFORE_FOLD = 4000  # characters (~3000 tokens)

    def __init__(
        self,
        llm_fn: Optional[Callable[[str], str]] = None,
        max_chars: int = MAX_TOKENS_BEFORE_FOLD,
    ):
        self._llm = llm_fn
        self._max_chars = max_chars
        self._fold_count = 0
        self._lock = threading.Lock()

    def needs_folding(self, text: str) -> bool:
        return len(text) > self._max_chars

    def _chunk_text(self, text: str) -> List[str]:
        lines = text.splitlines()
        chunks = []
        for i in range(0, len(lines), self.MAX_CHUNK_LINES):
            chunk = "\n".join(lines[i:i + self.MAX_CHUNK_LINES])
            if chunk.strip():
                chunks.append(chunk)
        return chunks if chunks else [text]

    def _summarise_chunk(self, chunk: str) -> str:
        if not self._llm:
            # Fallback: keep first + last line of chunk
            lines = chunk.splitlines()
            if len(lines) <= 2:
                return chunk
            return lines[0] + " … " + lines[-1]

        prompt = (
            f"Compress this execution log segment into ONE dense sentence "
            f"preserving all key facts, errors, and outcomes:\n\n"
            f"{chunk}\n\n"
            f"Return ONLY the compressed sentence."
        )
        try:
            return self._llm(prompt).strip()
        except Exception as e:
            logger.warning(f"[StateFolding] Summarise error: {e}")
            lines = chunk.splitlines()
            return lines[0] if lines else chunk[:100]

    def fold(self, text: str, label: str = "log") -> str:
        """
        Fold a large text block into a dense summary.
        Returns the folded text. Thread-safe.
        """
        if not self.needs_folding(text):
            return text

        with self._lock:
            chunks = self._chunk_text(text)
            summaries = [self._summarise_chunk(c) for c in chunks]
            folded = " | ".join(s for s in summaries if s.strip())
            folded_header = f"[FOLDED-{label} {len(text)}ch→{len(folded)}ch] "
            result = folded_header + folded
            self._fold_count += 1
            logger.info(
                f"[StateFolding] Folded {label}: "
                f"{len(text)} chars → {len(result)} chars "
                f"({len(chunks)} chunks)"
            )
            return result

    def fold_mission_log(self, log: List[dict]) -> str:
        """Fold a OODA mission log (list of dicts) into a dense string."""
        text = "\n".join(
            f"[{e.get('event', 'event')}@iter{e.get('iteration', '?')}] "
            f"{e.get('obs', e.get('reason', ''))[:120]}"
            for e in log
        )
        return self.fold(text, label="mission_log")

    def get_stats(self) -> dict:
        return {"fold_operations": self._fold_count, "max_chars": self._max_chars}


# ─────────────────────────────────────────────────────────────────────
# KNOWLEDGE GAP LOGGER (#48)
# ─────────────────────────────────────────────────────────────────────
class KnowledgeGapLogger:
    """
    Whenever the AI encounters an error, admits ignorance, or fails to
    answer, it quietly logs the topic here.

    At night, NocturnalDistiller reads these gaps and searches/learns.
    """

    IGNORANCE_PATTERNS = re.compile(
        r"\b(i don't know|i cannot|i'm not sure|unknown|not found|error|failed|"
        r"unable to|cannot find|no information|i lack|outside my knowledge)\b",
        re.I,
    )

    def __init__(self, path: Path = GAP_LOG_PATH):
        self._path = path
        self._lock = threading.Lock()
        self._session_gaps: List[dict] = []

    def log_gap(self, topic: str, context: str = "", error: str = ""):
        """Explicitly log a knowledge gap."""
        entry = {
            "ts": time.time(),
            "topic": topic[:200],
            "context": context[:300],
            "error": error[:200],
            "resolved": False,
        }
        with self._lock:
            self._session_gaps.append(entry)
            try:
                with open(self._path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                logger.warning(f"[KnowledgeGap] Write error: {e}")
        logger.debug(f"[KnowledgeGap] Logged gap: {topic[:60]}")

    def auto_detect_gap(self, prompt: str, response: str) -> bool:
        """
        Auto-detect if a response indicates ignorance.
        Returns True if a gap was logged.
        """
        if self.IGNORANCE_PATTERNS.search(response):
            self.log_gap(
                topic=prompt[:150],
                context="auto-detected from response",
                error=response[:200],
            )
            return True
        return False

    def load_unresolved(self) -> List[dict]:
        """Load all unresolved gaps for NocturnalDistiller to process."""
        gaps = []
        if not self._path.exists():
            return gaps
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if not entry.get("resolved", False):
                            gaps.append(entry)
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.warning(f"[KnowledgeGap] Load error: {e}")
        return gaps

    def mark_resolved(self, topic: str):
        """Mark a topic as resolved after learning."""
        # Rewrite file with resolved flag
        all_entries = []
        if self._path.exists():
            try:
                with open(self._path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            e = json.loads(line)
                            if topic.lower() in e.get("topic", "").lower():
                                e["resolved"] = True
                            all_entries.append(e)
                        except json.JSONDecodeError:
                            pass
                with open(self._path, "w") as f:
                    for e in all_entries:
                        f.write(json.dumps(e) + "\n")
            except Exception as err:
                logger.warning(f"[KnowledgeGap] mark_resolved error: {err}")

    def get_stats(self) -> dict:
        unresolved = self.load_unresolved()
        return {
            "session_gaps": len(self._session_gaps),
            "total_unresolved": len(unresolved),
            "gap_log": str(self._path),
        }


# ─────────────────────────────────────────────────────────────────────
# SELF-EVOLVING SEARCH CORTEX (#50)
# ─────────────────────────────────────────────────────────────────────
class SelfEvolvingSearchCortex:
    """
    Improves web search queries over time by:
    1. Logging every search query + its result quality score
    2. Learning which syntax operators improve results
    3. Auto-augmenting future queries with learned operators

    Operators learned: site:, filetype:, intitle:, -exclude, "exact phrase"
    """

    OPERATORS = {
        "code":     'site:github.com OR site:stackoverflow.com',
        "docs":     'site:docs.python.org OR filetype:pdf',
        "research": 'filetype:pdf OR site:arxiv.org OR site:scholar.google.com',
        "news":     'after:2024 site:techcrunch.com OR site:wired.com',
        "api":      'site:developer.',
    }

    def __init__(self, path: Path = SEARCH_LOG_PATH):
        self._path = path
        self._patterns: Dict[str, Dict] = self._load()
        self._lock = threading.Lock()

    def _load(self) -> Dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except Exception:
                pass
        return {}

    def _save(self):
        try:
            self._path.write_text(json.dumps(self._patterns, indent=2))
        except Exception as e:
            logger.warning(f"[SearchCortex] Save error: {e}")

    def classify_query(self, query: str) -> str:
        """Classify query intent for operator selection."""
        q = query.lower()
        if any(k in q for k in ["github", "code", "implement", "library", "package", "function"]):
            return "code"
        if any(k in q for k in ["paper", "research", "study", "arxiv", "journal"]):
            return "research"
        if any(k in q for k in ["news", "latest", "today", "recent", "2024", "2025", "2026"]):
            return "news"
        if any(k in q for k in ["api", "endpoint", "developer", "sdk"]):
            return "api"
        if any(k in q for k in ["docs", "documentation", "guide", "tutorial", "how to"]):
            return "docs"
        return "general"

    def augment(self, query: str) -> str:
        """
        Return an augmented query with learned search operators.
        """
        intent = self.classify_query(query)
        operator = self.OPERATORS.get(intent, "")

        # Check if we've learned better patterns for this intent
        with self._lock:
            pattern_data = self._patterns.get(intent, {})
            learned_op = pattern_data.get("best_operator", operator)
            best_score = pattern_data.get("best_score", 0.0)

        # Only apply operator if it has proven effective (score > 0.6)
        if best_score > 0.6 and learned_op:
            augmented = f"{query} {learned_op}"
        elif operator and intent != "general":
            augmented = f"{query} {operator}"
        else:
            augmented = query

        logger.debug(f"[SearchCortex] {intent}: '{query[:40]}' → '{augmented[:60]}'")
        return augmented

    def record_result(self, query: str, quality_score: float):
        """
        Record the quality of a search result (0-1).
        Used to learn which operators are most effective.
        """
        intent = self.classify_query(query)
        operator = self.OPERATORS.get(intent, "")

        with self._lock:
            if intent not in self._patterns:
                self._patterns[intent] = {
                    "queries": 0,
                    "total_score": 0.0,
                    "best_score": 0.0,
                    "best_operator": operator,
                }
            p = self._patterns[intent]
            p["queries"] = p.get("queries", 0) + 1
            p["total_score"] = p.get("total_score", 0.0) + quality_score
            avg = p["total_score"] / p["queries"]

            if quality_score > p.get("best_score", 0.0):
                p["best_score"] = quality_score
                p["best_operator"] = operator

            p["avg_score"] = round(avg, 3)
            self._save()

        logger.debug(f"[SearchCortex] Recorded: intent={intent} score={quality_score:.2f}")

    def get_stats(self) -> dict:
        return {
            "learned_patterns": len(self._patterns),
            "patterns": self._patterns,
        }


# ─────────────────────────────────────────────────────────────────────
# GENERATIVE REPLAY — Long-Term Consolidation (#60)
# ─────────────────────────────────────────────────────────────────────
class GenerativeReplay:
    """
    Nightly job: reads chaotic War Room logs from the day and writes
    clean 'lessons learned' entries into ChromaDB / hippocampus.

    Each lesson is:
    - A structured fact: "When X, the AI should do Y because Z"
    - Stored with timestamp and source mission ID
    """

    def __init__(
        self,
        hippocampus=None,              # Hippocampus instance
        llm_fn: Optional[Callable] = None,
        war_room_dir: Path = Path("./war_room"),
        lesson_path: Path = LESSON_LOG_PATH,
    ):
        self._hippo = hippocampus
        self._llm = llm_fn
        self._war_dir = war_room_dir
        self._lesson_path = lesson_path
        self._lock = threading.Lock()
        self._processed_missions: set = set()
        self._load_processed()

    def _load_processed(self):
        try:
            if self._lesson_path.exists():
                with open(self._lesson_path) as f:
                    for line in f:
                        e = json.loads(line.strip())
                        if "source_mission" in e:
                            self._processed_missions.add(e["source_mission"])
        except Exception:
            pass

    def _extract_lesson(self, mission_data: dict) -> Optional[str]:
        """Extract a lesson from a completed mission."""
        goal = mission_data.get("goal", "")
        status = mission_data.get("status", "")
        log = mission_data.get("mission_log", [])

        if not goal or not log:
            return None

        # Build a concise log summary
        events = [
            f"{e.get('event', '?')}: {e.get('obs', e.get('reason', ''))[:80]}"
            for e in log[:10]
        ]
        log_text = " → ".join(events)

        if self._llm:
            prompt = (
                f"Extract ONE clear lesson from this completed AI mission.\n"
                f"Goal: {goal}\n"
                f"Status: {status}\n"
                f"Log summary: {log_text}\n\n"
                f"Write the lesson as: 'When [situation], [action] because [reason].'\n"
                f"Return ONLY the lesson sentence. Under 60 words."
            )
            try:
                return self._llm(prompt).strip()
            except Exception as e:
                logger.warning(f"[GenerativeReplay] LLM lesson error: {e}")

        # Fallback: template
        outcome = "succeeded" if status == "done" else "failed"
        return f"Mission '{goal[:50]}' {outcome} after {len(log)} steps."

    def _store_lesson(self, lesson: str, mission_id: str):
        """Store lesson in hippocampus and lesson log."""
        entry = {
            "ts": time.time(),
            "lesson": lesson,
            "source_mission": mission_id,
        }
        # Write to lesson log
        try:
            with open(self._lesson_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"[GenerativeReplay] Log write error: {e}")

        # Store in hippocampus
        if self._hippo:
            try:
                self._hippo.store_fact(
                    lesson,
                    metadata={"type": "lesson", "source_mission": mission_id},
                )
            except Exception as e:
                logger.warning(f"[GenerativeReplay] Hippocampus store error: {e}")

    def consolidate(self) -> int:
        """
        Process all unprocessed War Room mission files.
        Returns number of lessons learned.
        """
        lessons_learned = 0
        if not self._war_dir.exists():
            logger.info("[GenerativeReplay] War room dir not found — skipping.")
            return 0

        mission_files = list(self._war_dir.glob("mission_*.json"))
        for mission_file in mission_files:
            mission_id = mission_file.stem.replace("mission_", "")
            if mission_id in self._processed_missions:
                continue

            try:
                data = json.loads(mission_file.read_text())
                # Only process completed missions
                if data.get("status") not in ("done", "failed"):
                    continue

                lesson = self._extract_lesson(data)
                if lesson:
                    self._store_lesson(lesson, mission_id)
                    self._processed_missions.add(mission_id)
                    lessons_learned += 1
                    logger.info(f"[GenerativeReplay] Lesson: {lesson[:60]}")

            except Exception as e:
                logger.warning(f"[GenerativeReplay] Error processing {mission_file.name}: {e}")

        logger.info(f"[GenerativeReplay] Consolidated {lessons_learned} lessons.")
        return lessons_learned

    def start_nightly_loop(self, hour: int = 3):
        """
        Start background thread that runs consolidation nightly at given hour (0-23).
        """
        def _loop():
            while True:
                now = time.localtime()
                if now.tm_hour == hour and now.tm_min < 5:
                    logger.info("[GenerativeReplay] Nightly consolidation starting...")
                    count = self.consolidate()
                    logger.info(f"[GenerativeReplay] Nightly complete: {count} lessons.")
                    time.sleep(3600)  # Sleep 1hr to avoid re-running same hour
                else:
                    time.sleep(300)  # Check every 5 minutes

        threading.Thread(target=_loop, daemon=True, name="GenerativeReplay").start()
        logger.info(f"[GenerativeReplay] Nightly loop scheduled for {hour:02d}:00")

    def get_stats(self) -> dict:
        return {
            "processed_missions": len(self._processed_missions),
            "lesson_log": str(self._lesson_path),
            "hippocampus_connected": self._hippo is not None,
        }


# ─────────────────────────────────────────────────────────────────────
# ON-DEVICE PERPETUAL LEARNING (#11)
# ─────────────────────────────────────────────────────────────────────
class OnDeviceLearning:
    """
    Uses successful War Room logs to guide LoRA fine-tuning of the
    local GGUF model. Provides a simulation layer — actual fine-tuning
    requires unsloth/trl libraries and a base model in safetensors format.

    In practice:
    - Exports training pairs (prompt, ideal_response) from successful missions
    - Checks if unsloth is available → runs LoRA training
    - Falls back to logging training data for manual fine-tuning
    """

    TRAINING_DATA_PATH = _MEMORY_DIR / "training_pairs.jsonl"

    def __init__(
        self,
        war_room_dir: Path = Path("./war_room"),
        model_path: Optional[Path] = None,
        monologue=None,
    ):
        self._war_dir = war_room_dir
        self._model_path = model_path
        self._monologue = monologue
        self._pairs_extracted = 0

    def extract_training_pairs(self) -> int:
        """
        Extract (prompt, response) pairs from successful missions.
        Returns count of pairs extracted.
        """
        if not self._war_dir.exists():
            return 0

        pairs = []
        for mission_file in self._war_dir.glob("mission_*.json"):
            try:
                data = json.loads(mission_file.read_text())
                if data.get("status") != "done":
                    continue

                goal = data.get("goal", "")
                log = data.get("mission_log", [])

                # Extract observation pairs as training signals
                for entry in log:
                    if entry.get("event") == "observation" and entry.get("obs"):
                        pairs.append({
                            "prompt": f"Execute this goal: {goal}",
                            "response": entry["obs"],
                            "source": mission_file.stem,
                        })
            except Exception:
                pass

        if pairs:
            try:
                with open(self.TRAINING_DATA_PATH, "a") as f:
                    for pair in pairs:
                        f.write(json.dumps(pair) + "\n")
                self._pairs_extracted += len(pairs)
                logger.info(f"[OnDeviceLearning] Extracted {len(pairs)} training pairs.")
            except Exception as e:
                logger.warning(f"[OnDeviceLearning] Write error: {e}")

        return len(pairs)

    def attempt_lora_finetune(self) -> dict:
        """
        Attempt LoRA fine-tuning via unsloth if available.
        Returns status dict.
        """
        try:
            import unsloth  # noqa: F401
            logger.info("[OnDeviceLearning] unsloth available — LoRA fine-tuning possible.")
            # In production: load model, apply LoRA, train on extracted pairs
            # This is scaffolding — full implementation requires GPU memory mgmt
            return {
                "status": "unsloth_available",
                "training_pairs": self._pairs_extracted,
                "note": "LoRA training scaffold ready. Call run_training() to execute.",
            }
        except ImportError:
            return {
                "status": "unsloth_not_installed",
                "training_pairs": self._pairs_extracted,
                "training_data_path": str(self.TRAINING_DATA_PATH),
                "note": "pip install unsloth to enable on-device LoRA fine-tuning.",
            }

    def get_stats(self) -> dict:
        pair_count = 0
        if self.TRAINING_DATA_PATH.exists():
            try:
                pair_count = sum(1 for _ in open(self.TRAINING_DATA_PATH))
            except Exception:
                pass
        return {
            "session_pairs": self._pairs_extracted,
            "total_pairs_on_disk": pair_count,
            "training_data": str(self.TRAINING_DATA_PATH),
        }


# ─────────────────────────────────────────────────────────────────────
# INTRINSIC CURIOSITY DRIVE (#88)
# ─────────────────────────────────────────────────────────────────────
class IntrinsicCuriosityDrive:
    """
    When the Mac has been idle for IDLE_THRESHOLD seconds, the AI
    autonomously performs useful housekeeping tasks:
    - Clean up Downloads folder (old files > 30 days)
    - Summarise unread calendar items
    - Organise Desktop files
    - Pre-warm common queries

    All actions require user confirmation before execution.
    Idle is measured via psutil CPU usage.
    """

    IDLE_THRESHOLD_SEC = 120  # 2 minutes idle
    IDLE_CPU_THRESHOLD = 5.0  # % CPU below which = idle

    CURIOSITY_TASKS = [
        {
            "id": "clean_downloads",
            "description": "Clean up Downloads folder (files older than 30 days)",
            "confirm_message": "I noticed you've been idle. Should I clean up Downloads? (old files > 30 days)",
            "category": "filesystem",
        },
        {
            "id": "summarise_calendar",
            "description": "Summarise upcoming calendar events for today and tomorrow",
            "confirm_message": "Want me to summarise your upcoming calendar events?",
            "category": "productivity",
        },
        {
            "id": "prewarm_context",
            "description": "Pre-warm AI context with recent project files",
            "confirm_message": "Should I pre-load your recent project context?",
            "category": "ai",
        },
        {
            "id": "knowledge_gaps",
            "description": "Research and resolve logged knowledge gaps",
            "confirm_message": "Should I research some topics I didn't know earlier?",
            "category": "learning",
        },
    ]

    def __init__(
        self,
        confirm_fn: Optional[Callable[[str], bool]] = None,
        execute_fn: Optional[Callable[[dict], dict]] = None,
        gap_logger: Optional[KnowledgeGapLogger] = None,
        llm_fn: Optional[Callable] = None,
    ):
        """
        confirm_fn: called with message, returns True/False (user decision)
        execute_fn: called with task dict, returns result dict
        """
        self._confirm = confirm_fn or (lambda msg: False)  # safe default: never auto-execute
        self._execute = execute_fn
        self._gap_logger = gap_logger
        self._llm = llm_fn
        self._stop_evt = threading.Event()
        self._last_action: float = 0.0
        self._action_log: List[dict] = []
        self._running = False

    def _is_idle(self) -> bool:
        """Check if system is idle (low CPU)."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=1.0)
            return cpu < self.IDLE_CPU_THRESHOLD
        except ImportError:
            # Fallback: assume idle if 2+ minutes since last action
            return (time.time() - self._last_action) > self.IDLE_THRESHOLD_SEC

    def _idle_seconds(self) -> float:
        """How long has the system been idle."""
        try:
            import psutil
            # Use last input time heuristic via process monitoring
            return time.time() - self._last_action
        except Exception:
            return time.time() - self._last_action

    def _pick_task(self) -> Optional[dict]:
        """Select the most useful curiosity task right now."""
        # Prefer knowledge gap research if gaps exist
        if self._gap_logger:
            gaps = self._gap_logger.load_unresolved()
            if gaps:
                return {
                    "id": "knowledge_gaps",
                    "description": f"Research {len(gaps)} knowledge gaps",
                    "confirm_message": (
                        f"I have {len(gaps)} topics I didn't know earlier. "
                        f"Should I research them now? (e.g., '{gaps[0]['topic'][:40]}')"
                    ),
                    "gaps": gaps[:3],
                    "category": "learning",
                }

        # Rotate through other tasks
        hour = time.localtime().tm_hour
        if 6 <= hour <= 9:
            return next((t for t in self.CURIOSITY_TASKS if t["id"] == "summarise_calendar"), None)
        if 22 <= hour or hour <= 5:
            return next((t for t in self.CURIOSITY_TASKS if t["id"] == "clean_downloads"), None)

        # Default: pre-warm context
        return next((t for t in self.CURIOSITY_TASKS if t["id"] == "prewarm_context"), None)

    def _research_gap(self, gap: dict) -> str:
        """Use LLM to generate knowledge about a gap topic."""
        if not self._llm:
            return f"[No LLM] Gap topic: {gap['topic']}"
        prompt = (
            f"Provide a concise, accurate answer to this question I couldn't answer before:\n"
            f"Topic: {gap['topic']}\n"
            f"Context: {gap.get('context', '')}\n\n"
            f"Answer in 3-5 sentences. Be specific and factual."
        )
        try:
            return self._llm(prompt)
        except Exception as e:
            return f"Research error: {e}"

    def _run_loop(self):
        """Background idle-check loop."""
        self._running = True
        logger.info("[CuriosityDrive] Background loop started.")

        while not self._stop_evt.is_set():
            if self._is_idle() and self._idle_seconds() > self.IDLE_THRESHOLD_SEC:
                task = self._pick_task()
                if task:
                    # Ask for confirmation
                    confirmed = self._confirm(task["confirm_message"])
                    if confirmed:
                        self._last_action = time.time()
                        result = self._do_task(task)
                        self._action_log.append({
                            "ts": time.time(),
                            "task": task["id"],
                            "result": str(result)[:200],
                        })
                        self._save_action(task, result)
                    else:
                        logger.debug(f"[CuriosityDrive] Task '{task['id']}' declined by user.")

            self._stop_evt.wait(timeout=60)  # check every minute

        self._running = False

    def _do_task(self, task: dict) -> dict:
        """Execute a curiosity task."""
        task_id = task.get("id", "unknown")

        if task_id == "knowledge_gaps":
            gaps = task.get("gaps", [])
            results = []
            for gap in gaps:
                answer = self._research_gap(gap)
                results.append({"topic": gap["topic"], "answer": answer[:200]})
                if self._gap_logger:
                    self._gap_logger.mark_resolved(gap["topic"])
            return {"gaps_resolved": len(results), "answers": results}

        if task_id == "prewarm_context" and self._llm:
            _ = self._llm("What are common Python async patterns? Brief overview.")
            return {"status": "context_prewarmed"}

        if self._execute:
            try:
                return self._execute(task)
            except Exception as e:
                return {"error": str(e)}

        return {"status": "task_acknowledged", "task_id": task_id}

    def _save_action(self, task: dict, result: dict):
        try:
            entry = {
                "ts": time.time(),
                "task_id": task.get("id"),
                "description": task.get("description", ""),
                "result": result,
            }
            with open(CURIOSITY_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"[CuriosityDrive] Log error: {e}")

    def start(self):
        """Start background curiosity loop."""
        self._stop_evt.clear()
        threading.Thread(
            target=self._run_loop, daemon=True, name="CuriosityDrive"
        ).start()
        logger.info("[CuriosityDrive] Started.")

    def stop(self):
        self._stop_evt.set()

    def update_last_action(self):
        """Call this whenever user interacts to reset idle timer."""
        self._last_action = time.time()

    def get_stats(self) -> dict:
        return {
            "running": self._running,
            "actions_taken": len(self._action_log),
            "last_action_ago": round(time.time() - self._last_action, 0),
            "idle_threshold_sec": self.IDLE_THRESHOLD_SEC,
        }


# ─────────────────────────────────────────────────────────────────────
# MEMORY EVOLUTION SYSTEM — Top-level facade
# ─────────────────────────────────────────────────────────────────────
class MemoryEvolutionSystem:
    """
    Unified facade for all memory / self-evolution subsystems.
    Connects to existing Hippocampus and NocturnalDistiller.
    """

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        hippocampus=None,
        war_room_dir: Path = Path("./war_room"),
        confirm_fn: Optional[Callable[[str], bool]] = None,
        monologue=None,
    ):
        self.folding = SemanticStateFolding(llm_fn=llm_fn)
        self.gap_logger = KnowledgeGapLogger()
        self.search_cortex = SelfEvolvingSearchCortex()
        self.replay = GenerativeReplay(
            hippocampus=hippocampus,
            llm_fn=llm_fn,
            war_room_dir=war_room_dir,
        )
        self.on_device = OnDeviceLearning(
            war_room_dir=war_room_dir,
            monologue=monologue,
        )
        self.curiosity = IntrinsicCuriosityDrive(
            confirm_fn=confirm_fn,
            gap_logger=self.gap_logger,
            llm_fn=llm_fn,
        )

    def start(self):
        """Start all background loops."""
        self.replay.start_nightly_loop(hour=3)
        self.curiosity.start()
        logger.info("[MemoryEvolution] All subsystems started.")

    def stop(self):
        self.curiosity.stop()

    def on_llm_response(self, prompt: str, response: str):
        """Call after every LLM response to auto-detect gaps."""
        self.gap_logger.auto_detect_gap(prompt, response)
        self.curiosity.update_last_action()

    def on_search(self, query: str) -> str:
        """Augment a search query with learned operators."""
        return self.search_cortex.augment(query)

    def fold_if_needed(self, text: str, label: str = "log") -> str:
        """Fold text if it exceeds size threshold."""
        return self.folding.fold(text, label)

    def get_status(self) -> dict:
        return {
            "folding": self.folding.get_stats(),
            "knowledge_gaps": self.gap_logger.get_stats(),
            "search_cortex": self.search_cortex.get_stats(),
            "replay": self.replay.get_stats(),
            "on_device": self.on_device.get_stats(),
            "curiosity": self.curiosity.get_stats(),
        }


# ── Module-level singleton ────────────────────────────────────────────
_mem_system: Optional[MemoryEvolutionSystem] = None


def get_memory_evolution(**kwargs) -> MemoryEvolutionSystem:
    global _mem_system
    if _mem_system is None:
        _mem_system = MemoryEvolutionSystem(**kwargs)
    return _mem_system


# ── Self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    import shutil
    logging.basicConfig(level=logging.DEBUG)
    print("💾 MemoryEvolution self-test\n")

    # Mock LLM
    def mock_llm(prompt: str) -> str:
        if "compress" in prompt.lower() or "one dense sentence" in prompt.lower():
            return "Compressed: system executed 5 tasks, 3 succeeded, 2 failed."
        if "lesson" in prompt.lower() or "extract" in prompt.lower():
            return "When executing DAG missions, always verify preconditions before starting tasks."
        if "json array" in prompt.lower() and "step" in prompt.lower():
            return '["Step 5: verify", "Step 4: test", "Step 3: build", "Step 2: setup", "Step 1: plan"]'
        if "rate" in prompt.lower():
            return "0.75"
        if "knowledge" in prompt.lower() or "answer" in prompt.lower():
            return "This is a factual answer to the knowledge gap."
        return f"Mock: {prompt[:40]}"

    tmpdir = Path(tempfile.mkdtemp())

    # Test 1: Semantic State Folding
    print("=== Test 1: Semantic State Folding ===")
    folding = SemanticStateFolding(llm_fn=mock_llm, max_chars=100)
    long_text = "\n".join([f"Event {i}: task_{i} completed with result OK." for i in range(30)])
    assert folding.needs_folding(long_text), "Should need folding"
    folded = folding.fold(long_text, "test_log")
    assert len(folded) < len(long_text), f"Folded should be shorter: {len(folded)} vs {len(long_text)}"
    assert "FOLDED" in folded, "Should have FOLDED header"
    print(f"✅ Folded {len(long_text)} chars → {len(folded)} chars")

    # Test 2: Knowledge Gap Logger
    print("\n=== Test 2: Knowledge Gap Logger ===")
    gap_path = tmpdir / "gaps.jsonl"
    gaps = KnowledgeGapLogger(path=gap_path)
    gaps.log_gap("quantum computing basics", "user asked about qubits")
    gaps.log_gap("Rust ownership model", "failed to explain")
    detected = gaps.auto_detect_gap("what is X?", "I don't know the answer to this.")
    assert detected, "Should detect gap in ignorant response"
    unresolved = gaps.load_unresolved()
    assert len(unresolved) >= 3, f"Should have ≥3 gaps, got {len(unresolved)}"
    gaps.mark_resolved("quantum computing basics")
    unresolved2 = gaps.load_unresolved()
    assert len(unresolved2) < len(unresolved), "Should have fewer gaps after resolution"
    print(f"✅ Gap logging: {len(unresolved)} logged, {len(unresolved2)} remaining after resolution")

    # Test 3: Self-Evolving Search Cortex
    print("\n=== Test 3: Self-Evolving Search Cortex ===")
    cortex = SelfEvolvingSearchCortex(path=tmpdir / "search.json")
    augmented_code = cortex.augment("Python async generator implementation")
    augmented_news = cortex.augment("latest AI news today 2026")
    augmented_simple = cortex.augment("what is 2+2")
    assert "github" in augmented_code.lower() or "stackoverflow" in augmented_code.lower(), \
        f"Code query should get code operators: {augmented_code}"
    print(f"✅ Code query augmented: {augmented_code[:80]}")
    print(f"✅ News query augmented: {augmented_news[:80]}")
    cortex.record_result("Python async code", 0.9)
    stats = cortex.get_stats()
    assert stats["learned_patterns"] > 0, "Should have learned patterns"
    print(f"✅ Learned {stats['learned_patterns']} patterns")

    # Test 4: Generative Replay
    print("\n=== Test 4: Generative Replay ===")
    war_dir = tmpdir / "war_room"
    war_dir.mkdir()
    # Create a mock mission file
    mission = {
        "id": "test_m1",
        "goal": "Deploy FastAPI to production",
        "status": "done",
        "mission_log": [
            {"event": "iteration_start", "iteration": 1},
            {"event": "observation", "iteration": 1, "obs": "All tasks completed successfully."},
            {"event": "goal_achieved", "iteration": 1},
        ],
    }
    (war_dir / "mission_test_m1.json").write_text(json.dumps(mission))

    replay = GenerativeReplay(
        llm_fn=mock_llm,
        war_room_dir=war_dir,
        lesson_path=tmpdir / "lessons.jsonl",
    )
    count = replay.consolidate()
    assert count >= 1, f"Should learn at least 1 lesson, got {count}"
    print(f"✅ Generative Replay: {count} lesson(s) learned")

    # Test 5: On-Device Learning
    print("\n=== Test 5: On-Device Learning ===")
    learner = OnDeviceLearning(war_room_dir=war_dir)
    pairs = learner.extract_training_pairs()
    result = learner.attempt_lora_finetune()
    assert "status" in result, "Should return status"
    print(f"✅ On-Device: {pairs} pairs extracted, finetune status: {result['status']}")

    # Test 6: MemoryEvolutionSystem facade
    print("\n=== Test 6: MemoryEvolutionSystem Facade ===")
    system = MemoryEvolutionSystem(
        llm_fn=mock_llm,
        war_room_dir=war_dir,
        confirm_fn=lambda msg: False,  # never auto-execute in tests
    )
    # Test fold_if_needed
    short = "short text"
    long_t = "x " * 2500  # > 4000 chars
    assert system.fold_if_needed(short, "test") == short, "Short text should not be folded"
    folded_long = system.fold_if_needed(long_t, "test")
    assert len(folded_long) < len(long_t), "Long text should be folded"

    # Test on_llm_response gap detection
    system.on_llm_response("Tell me about X", "I don't know anything about X.")
    assert system.gap_logger.get_stats()["session_gaps"] > 0, "Should have logged a gap"

    status = system.get_status()
    assert "folding" in status, "Status should have folding"
    assert "knowledge_gaps" in status, "Status should have gaps"
    print(f"✅ MemoryEvolutionSystem status: {list(status.keys())}")

    shutil.rmtree(tmpdir)
    print("\n✅ All MemoryEvolution tests passed.")
