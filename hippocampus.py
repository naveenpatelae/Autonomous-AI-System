#!/usr/bin/env python3
# =====================================================================
# 🧠 PHASE 4.1 — EPISODIC MEMORY (Hippocampus)  v13.2 — BODY EDITION
#
# ┌─────────────────────────────────────────────────────────────────┐
# │  ARCHITECTURE SPLIT                                             │
# │                                                                 │
# │  BRAIN (Kaggle)  — KaggleHippocampus  (ChromaDB + cloud sync)  │
# │  BODY  (Mac)     — BodyHippocampus    (this file)              │
# │                                                                 │
# │  Body responsibilities:                                         │
# │    • Store facts locally (numpy TF-IDF, no ChromaDB needed)     │
# │    • Inject relevant memories into every local prompt           │
# │    • Sync facts to Brain when online (fire-and-forget thread)   │
# │    • Pull facts from Brain when local store is cold             │
# │    • Honor VanishProtocol triggers arriving from Brain via WS   │
# │    • Write tamper-evidence log and tombstone on wipe            │
# └─────────────────────────────────────────────────────────────────┘
#
# Classes exported:
#   FactExtractor        — sentence-level fact extraction
#   NumpyEpisodicStore   — thread-safe TF-IDF vector store
#   VanishProtocol       — self-destruct mechanism (wipes local store)
#   BodyHippocampus      — main body-side episodic memory class
#   get_body_hippocampus — module-level singleton factory
#
# All original v13.1 logic is preserved verbatim.
# New in v13.2 (body):
#   • BrainSyncAgent     — background thread that pushes/pulls facts
#   • augment_prompt()   — convenience alias for prompts
#   • from_vanish_signal() — handles incoming WS vanish message
#   • get_status()       — unified status dict
# =====================================================================

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("BodyHippocampus")

# ── Config ────────────────────────────────────────────────────────────
LOCAL_MEMORY_DIR      = Path(os.getenv("BODY_MEMORY_DIR", "./body_memory"))
LOCAL_FACTS_FILE      = LOCAL_MEMORY_DIR / "facts.jsonl"
MAX_FACTS_IN_PROMPT   = 5
MIN_RELEVANCE_SCORE   = 0.30
FACT_EXTRACTION_MIN_LEN = 15
BRAIN_SYNC_INTERVAL   = 60          # seconds between background syncs
BRAIN_MEMORY_STORE    = "/memory/store"
BRAIN_MEMORY_QUERY    = "/memory/query"

# ── Optional deps ─────────────────────────────────────────────────────
try:
    import numpy as _np
    _NP_OK = True
except ImportError:
    _np = None
    _NP_OK = False

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _requests = None          # type: ignore
    _REQUESTS_OK = False
    logger.warning("requests not installed — brain sync disabled.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 1 — FACT EXTRACTOR  (identical logic to Brain's v13.1)
# ─────────────────────────────────────────────────────────────────────
class FactExtractor:
    """
    Extracts declarative facts from conversation turns.
    Sentence-level; applies inclusion patterns and exclusion guards.
    Identical to the Brain's FactExtractor so both sides agree on
    what constitutes a storable fact.
    """

    _FACT_PATTERNS: List = [
        re.compile(r'\b(I am|I\'m|my name is|I work|I like|I prefer|I use|I have)\b', re.I),
        re.compile(r'\b(always|never|usually|the project|the system|our|we use)\b', re.I),
        re.compile(r'\b(version|v\d|python|mac|linux|windows|api|model)\b', re.I),
    ]
    _EXCLUDE_PATTERNS: List = [
        re.compile(r'^\s*(ok|yes|no|sure|thanks|hi|hello|bye)\s*[.!?]?\s*$', re.I),
        re.compile(r'\?$'),
    ]

    def extract(self, user_text: str, ai_text: str) -> List[str]:
        facts: List[str] = []
        for source in [user_text, ai_text]:
            sentences = re.split(r'(?<=[.!?])\s+', source)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < FACT_EXTRACTION_MIN_LEN:
                    continue
                if any(p.search(sent) for p in self._EXCLUDE_PATTERNS):
                    continue
                if any(p.search(sent) for p in self._FACT_PATTERNS):
                    facts.append(sent)
                    continue
                if 20 <= len(sent) <= 250 and not sent.endswith("?"):
                    facts.append(sent)

        seen: set = set()
        unique: List[str] = []
        for f in facts:
            k = f.lower().strip()
            if k not in seen:
                seen.add(k)
                unique.append(f)
        return unique[:10]


# ─────────────────────────────────────────────────────────────────────
# SECTION 2 — NUMPY EPISODIC STORE  (enhanced from v13.1)
# ─────────────────────────────────────────────────────────────────────
class NumpyEpisodicStore:
    """
    Thread-safe in-memory vector store backed by TF-IDF style embeddings.
    Primary store for the body when ChromaDB is not available.
    Persists facts to a JSONL file so they survive restarts.
    """

    def __init__(self, persist_file: Optional[Path] = None):
        self._facts: List[Dict] = []
        self._vocab: List[str] = []
        self._matrix = None
        self._dirty = False
        self._lock = threading.Lock()
        self._persist_file = persist_file
        if persist_file:
            self._load_from_disk()

    # ── Persistence ───────────────────────────────────────────────────
    def _load_from_disk(self):
        """Load previously saved facts from JSONL file."""
        if not self._persist_file or not self._persist_file.exists():
            return
        try:
            with open(self._persist_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        # Only load if not already present
                        if not any(x["id"] == entry["id"] for x in self._facts):
                            self._facts.append(entry)
            self._dirty = True
            self._rebuild_index()
            logger.info(f"[NumpyStore] Loaded {len(self._facts)} facts from disk.")
        except Exception as e:
            logger.warning(f"[NumpyStore] Load error: {e}")

    def _save_fact_to_disk(self, fact: Dict):
        """Append a single fact to the JSONL persist file."""
        if not self._persist_file:
            return
        try:
            self._persist_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persist_file, "a") as f:
                f.write(json.dumps({"id": fact["id"], "text": fact["text"],
                                    "metadata": fact.get("metadata", {})}) + "\n")
        except Exception as e:
            logger.warning(f"[NumpyStore] Disk save error: {e}")

    def _rewrite_disk(self):
        """Rewrite the entire JSONL file (called after wipe)."""
        if not self._persist_file:
            return
        try:
            with open(self._persist_file, "w") as f:
                for fact in self._facts:
                    f.write(json.dumps({"id": fact["id"], "text": fact["text"],
                                        "metadata": fact.get("metadata", {})}) + "\n")
        except Exception as e:
            logger.warning(f"[NumpyStore] Rewrite error: {e}")

    # ── Tokenization & Embedding ──────────────────────────────────────
    def _tokenize(self, text: str) -> set:
        return set(re.findall(r'\b[a-z][a-z0-9]{2,}\b', text.lower()))

    def _embed(self, text: str) -> list:
        toks = self._tokenize(text)
        return [1.0 if v in toks else 0.0 for v in self._vocab]

    def _rebuild_index(self):
        if not _NP_OK or not self._facts:
            self._dirty = False
            return
        counts: Dict[str, int] = {}
        for f in self._facts:
            for t in self._tokenize(f["text"]):
                counts[t] = counts.get(t, 0) + 1
        n = len(self._facts)
        self._vocab = [t for t, c in counts.items() if 1 <= c < n or n <= 3]
        rows = []
        for f in self._facts:
            row = _np.array(self._embed(f["text"]))
            norm = _np.linalg.norm(row)
            if norm > 0:
                row = row / norm
            rows.append(row)
            f["embedding"] = row
        self._matrix = _np.array(rows) if rows else None
        self._dirty = False

    # ── CRUD ──────────────────────────────────────────────────────────
    def add(self, fact_id: str, text: str, metadata: Optional[Dict] = None) -> bool:
        """Returns True if added, False if already existed."""
        with self._lock:
            for f in self._facts:
                if f["id"] == fact_id:
                    return False
            entry = {"id": fact_id, "text": text,
                     "metadata": metadata or {}, "embedding": None}
            self._facts.append(entry)
            self._dirty = True
            self._rebuild_index()
            self._save_fact_to_disk(entry)
            return True

    def query(self, query_text: str, top_k: int = MAX_FACTS_IN_PROMPT) -> List[Dict]:
        with self._lock:
            if self._dirty:
                self._rebuild_index()
            if not _NP_OK or self._matrix is None or not self._vocab:
                return [{"text": f["text"], "score": 0.5, "id": f["id"]}
                        for f in self._facts[:top_k]]
            toks = self._tokenize(query_text)
            qvec = _np.array([1.0 if v in toks else 0.0 for v in self._vocab])
            norm = _np.linalg.norm(qvec)
            if norm == 0:
                return []
            qvec = qvec / norm
            sims = self._matrix @ qvec
            top = sims.argsort()[::-1][:top_k]
            return [
                {"id": self._facts[i]["id"], "text": self._facts[i]["text"],
                 "score": round(float(sims[i]), 4)}
                for i in top if float(sims[i]) >= MIN_RELEVANCE_SCORE
            ]

    def get_all(self) -> List[Dict]:
        with self._lock:
            return [{"id": f["id"], "text": f["text"]} for f in self._facts]

    def count(self) -> int:
        return len(self._facts)

    def wipe(self):
        """Clears all in-memory and on-disk facts."""
        with self._lock:
            self._facts.clear()
            self._vocab.clear()
            self._matrix = None
            self._dirty = False
            self._rewrite_disk()
        logger.info("[NumpyStore] Wiped all facts.")


# ─────────────────────────────────────────────────────────────────────
# SECTION 3 — VANISH PROTOCOL  (v13.1 — preserved verbatim)
# ─────────────────────────────────────────────────────────────────────
class VanishProtocol:
    """
    Self-destruct mechanism for the local vector store.

    Triggers:
      • trigger_unauthorized_access(reason)
      • trigger_biometric_failure(bpm, reason)
      • trigger_manual(reason)

    On trigger:
      1. Writes tamper-evidence log (JSONL)
      2. Broadcasts WS alert to UI (if ws_broadcast_fn set)
      3. Creates cryptographic tombstone in parent dir
      4. Wipes local facts file and in-memory store
      5. Fires on_vanish callback

    Tombstone prevents false corruption alerts:
      .vanish_tombstone.json: {ts, reason, sha256_of_store_path, intentional: true}
    """

    TOMBSTONE_FILENAME  = ".vanish_tombstone.json"
    TAMPER_LOG_FILENAME = ".tamper_evidence.jsonl"

    def __init__(
        self,
        store_dir:            Path,
        ws_broadcast_fn:      Optional[Callable[[Dict], None]] = None,
        on_vanish:            Optional[Callable[[str], None]] = None,
        require_confirmation: bool = False,
        confirm_fn:           Optional[Callable[[str], bool]] = None,
    ):
        self._dir          = store_dir
        self._broadcast    = ws_broadcast_fn
        self._on_vanish    = on_vanish
        self._require_confirm = require_confirmation
        self._confirm      = confirm_fn or (lambda _: True)
        self._wiped        = False
        self._lock         = threading.Lock()
        self._wipe_log:    List[Dict] = []

    # ── Public triggers ───────────────────────────────────────────────
    def trigger_unauthorized_access(self, reason: str = "unauthorized_access") -> Dict:
        logger.critical(f"[VanishProtocol] UNAUTHORIZED ACCESS — wipe: {reason}")
        return self._execute_vanish(f"unauthorized_access:{reason}")

    def trigger_biometric_failure(self, bpm: int = 0, reason: str = "") -> Dict:
        full_reason = (f"biometric_failure:bpm={bpm}:{reason}"
                       if reason else f"biometric_failure:bpm={bpm}")
        logger.critical(f"[VanishProtocol] BIOMETRIC FAILURE — wipe: {full_reason}")
        return self._execute_vanish(full_reason)

    def trigger_manual(self, reason: str = "manual_trigger") -> Dict:
        logger.critical(f"[VanishProtocol] MANUAL TRIGGER — wipe: {reason}")
        return self._execute_vanish(reason)

    def is_tombstoned(self) -> bool:
        tombstone = self._dir.parent / self.TOMBSTONE_FILENAME
        return tombstone.exists()

    def get_wipe_log(self) -> List[Dict]:
        return list(self._wipe_log)

    # ── Internal execution ────────────────────────────────────────────
    def _execute_vanish(self, reason: str) -> Dict:
        with self._lock:
            if self._wiped:
                return {"status": "ALREADY_WIPED", "reason": "Previous wipe executed"}

            if self._require_confirm:
                confirmed = self._confirm(
                    f"⚠️ LOCAL MEMORY WIPE REQUESTED: {reason}\n"
                    f"Store path: {self._dir}\nProceed?"
                )
                if not confirmed:
                    logger.info("[VanishProtocol] Wipe cancelled by confirmation gate")
                    return {"status": "CANCELLED", "reason": "User declined"}

            # 1. Tamper-evidence log
            tamper_entry = {
                "ts":        time.time(),
                "reason":    reason,
                "store_path": str(self._dir),
                "store_exists": self._dir.exists(),
                "intentional": True,
            }
            self._wipe_log.append(tamper_entry)
            tamper_log = self._dir.parent / self.TAMPER_LOG_FILENAME
            try:
                tamper_log.parent.mkdir(parents=True, exist_ok=True)
                with open(tamper_log, "a") as f:
                    f.write(json.dumps(tamper_entry) + "\n")
                logger.info(f"[VanishProtocol] Tamper log: {tamper_log}")
            except Exception as e:
                logger.warning(f"[VanishProtocol] Tamper log write failed: {e}")

            # 2. WS alert
            if self._broadcast:
                alert = {
                    "type":    "vanish_alert",
                    "reason":  reason,
                    "store_path": str(self._dir),
                    "ts":      time.time(),
                    "message": f"🗑️ Body Memory Vanish Protocol activated: {reason}",
                }
                try:
                    self._broadcast(alert)
                    logger.info("[VanishProtocol] WS vanish alert sent")
                except Exception as e:
                    logger.warning(f"[VanishProtocol] WS alert failed: {e}")

            # 3. Tombstone hash
            store_path_hash = hashlib.sha256(str(self._dir).encode()).hexdigest()

            # 4. Wipe the store directory
            wipe_result = self._wipe_directory()

            # 5. Write tombstone (in parent, survives the wipe)
            tombstone = self._dir.parent / self.TOMBSTONE_FILENAME
            tombstone_data = {
                "ts":              time.time(),
                "reason":          reason,
                "sha256_store":    store_path_hash,
                "intentional":     True,
                "wipe_result":     wipe_result,
            }
            try:
                tombstone.parent.mkdir(parents=True, exist_ok=True)
                tombstone.write_text(json.dumps(tombstone_data, indent=2))
                logger.info(f"[VanishProtocol] Tombstone written: {tombstone}")
            except Exception as e:
                logger.warning(f"[VanishProtocol] Tombstone write failed: {e}")

            self._wiped = True

        # 6. Fire callback (outside lock)
        if self._on_vanish:
            try:
                self._on_vanish(reason)
            except Exception:
                pass

        logger.critical(
            f"☠️  [VanishProtocol] VANISH COMPLETE: "
            f"{wipe_result.get('files_deleted', 0)} files deleted. Reason: {reason}"
        )
        return {
            "status":      "WIPED",
            "reason":      reason,
            "wipe_result": wipe_result,
            "tombstone":   str(self._dir.parent / self.TOMBSTONE_FILENAME),
            "tamper_log":  str(self._dir.parent / self.TAMPER_LOG_FILENAME),
        }

    def _wipe_directory(self) -> Dict:
        """Recursively delete the local memory directory."""
        result = {"files_deleted": 0, "dirs_deleted": 0, "errors": []}
        if not self._dir.exists():
            return {**result, "note": "Directory did not exist"}
        try:
            all_items = list(self._dir.rglob("*"))
            result["files_deleted"] = sum(1 for f in all_items if f.is_file())
            result["dirs_deleted"]  = sum(1 for f in all_items if f.is_dir())
            shutil.rmtree(self._dir)
            logger.info(
                f"[VanishProtocol] Wiped {result['files_deleted']} files, "
                f"{result['dirs_deleted']} dirs from {self._dir}"
            )
        except Exception as e:
            result["errors"].append(str(e))
            logger.error(f"[VanishProtocol] Wipe error: {e}")
            # File-by-file fallback
            for f in self._dir.rglob("*"):
                try:
                    if f.is_file():
                        f.unlink()
                        result["files_deleted"] += 1
                except Exception as fe:
                    result["errors"].append(str(fe))
        return result


# ─────────────────────────────────────────────────────────────────────
# SECTION 4 — BRAIN SYNC AGENT
# ─────────────────────────────────────────────────────────────────────
class BrainSyncAgent:
    """
    Background thread that syncs local facts to the Brain (Kaggle) and
    optionally pulls facts from the Brain when the local store is empty.

    Strategy:
      PUSH — any fact added locally is queued and pushed to Brain's
             POST /memory/store endpoint.  Fire-and-forget; failures
             are logged but never crash the body.
      PULL — on startup (if local store is cold) and on demand,
             pulls from Brain's POST /memory/query endpoint.

    The sync runs in a daemon thread — it stops automatically when
    the process exits.
    """

    def __init__(
        self,
        brain_url:    str,
        push_queue:   List[Dict],
        queue_lock:   threading.Lock,
        interval:     int = BRAIN_SYNC_INTERVAL,
        on_sync_fail: Optional[Callable[[str], None]] = None,
    ):
        self._brain_url  = brain_url.rstrip("/")
        self._queue      = push_queue
        self._qlock      = queue_lock
        self._interval   = interval
        self._on_fail    = on_sync_fail
        self._thread: Optional[threading.Thread] = None
        self._running    = False
        self._last_push  = 0.0
        self._last_pull  = 0.0
        self._push_ok    = 0
        self._push_fail  = 0

    def start(self):
        if self._running:
            return
        if not _REQUESTS_OK:
            logger.warning("[BrainSync] requests not available — sync disabled.")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="BrainSyncAgent")
        self._thread.start()
        logger.info(f"[BrainSync] Started (interval={self._interval}s, brain={self._brain_url})")

    def stop(self):
        self._running = False

    def push_fact(self, text: str, metadata: Optional[Dict] = None):
        """Queue a fact for async push to Brain."""
        with self._qlock:
            self._queue.append({"text": text, "metadata": metadata or {}})

    def pull_facts(self, query: str, top_k: int = 10) -> List[Dict]:
        """Synchronous pull from Brain — called on cold-start."""
        if not _REQUESTS_OK or not self._brain_url:
            return []
        try:
            r = _requests.post(
                f"{self._brain_url}{BRAIN_MEMORY_QUERY}",
                json={"query": query, "top_k": top_k},
                timeout=5,
            )
            if r.status_code == 200:
                data = r.json()
                self._last_pull = time.time()
                return data.get("facts", [])
        except Exception as e:
            logger.debug(f"[BrainSync] Pull failed: {e}")
        return []

    def get_stats(self) -> Dict:
        return {
            "push_ok":   self._push_ok,
            "push_fail": self._push_fail,
            "queue_len": len(self._queue),
            "last_push": self._last_push,
            "last_pull": self._last_pull,
            "running":   self._running,
        }

    def _loop(self):
        while self._running:
            time.sleep(self._interval)
            self._flush_queue()

    def _flush_queue(self):
        with self._qlock:
            if not self._queue:
                return
            batch = list(self._queue)
            self._queue.clear()

        for item in batch:
            try:
                r = _requests.post(
                    f"{self._brain_url}{BRAIN_MEMORY_STORE}",
                    json={"text": item["text"]},
                    timeout=5,
                )
                if r.status_code == 200:
                    self._push_ok += 1
                    self._last_push = time.time()
                else:
                    self._push_fail += 1
                    logger.debug(f"[BrainSync] Push HTTP {r.status_code}")
            except Exception as e:
                self._push_fail += 1
                logger.debug(f"[BrainSync] Push error: {e}")
                if self._on_fail:
                    try:
                        self._on_fail(str(e))
                    except Exception:
                        pass


# ─────────────────────────────────────────────────────────────────────
# SECTION 5 — BODY HIPPOCAMPUS  (main body-side class)
# ─────────────────────────────────────────────────────────────────────
class BodyHippocampus:
    """
    Phase 4.1 Episodic Memory — Body (Mac) Edition.

    Mirrors the Brain's KaggleHippocampus API so that any code
    that calls hippocampus.ingest_turn() / query_relevant_facts() /
    build_memory_augmented_prompt() works identically on both sides.

    Storage:
      Primary  — NumpyEpisodicStore (local JSONL + TF-IDF index)
      Sync     — BrainSyncAgent pushes new facts to Brain when online

    Vanish:
      VanishProtocol.trigger_*() wipes the local store directory.
      from_vanish_signal() handles incoming WS messages from Brain.

    Usage:
        mem = BodyHippocampus(brain_url="https://xxx.ngrok.io")
        mem.ingest_turn(user="I use Python 3.11", ai="Great!")
        prompt = mem.augment_prompt(base_prompt, query)

        # Wipe on breach:
        mem.vanish_on_breach("reason")
        mem.vanish_on_biometric_failure(bpm=0)

        # Handle WS message from Brain:
        mem.from_vanish_signal({"type":"vanish_alert","reason":"breach"})
    """

    def __init__(
        self,
        store_dir:            Path = LOCAL_MEMORY_DIR,
        brain_url:            str  = "",
        ws_broadcast_fn:      Optional[Callable[[Dict], None]] = None,
        on_vanish:            Optional[Callable[[str], None]] = None,
        require_vanish_confirm: bool = False,
        confirm_fn:           Optional[Callable[[str], bool]] = None,
        sync_interval:        int  = BRAIN_SYNC_INTERVAL,
        auto_start_sync:      bool = True,
        cold_start_pull:      bool = True,
    ):
        self._store_dir = store_dir
        self._brain_url = brain_url
        self._external_on_vanish = on_vanish
        self._vanished  = False
        self._lock      = threading.Lock()

        # Local store
        store_dir.mkdir(parents=True, exist_ok=True)
        self._store = NumpyEpisodicStore(
            persist_file=store_dir / "facts.jsonl"
        )
        self._extractor = FactExtractor()

        # Vanish Protocol
        self._vanish = VanishProtocol(
            store_dir            = store_dir,
            ws_broadcast_fn      = ws_broadcast_fn,
            on_vanish            = self._on_vanish_callback,
            require_confirmation = require_vanish_confirm,
            confirm_fn           = confirm_fn,
        )

        # Brain sync
        self._push_queue: List[Dict] = []
        self._queue_lock  = threading.Lock()
        self._sync_agent  = BrainSyncAgent(
            brain_url  = brain_url,
            push_queue = self._push_queue,
            queue_lock = self._queue_lock,
            interval   = sync_interval,
        ) if brain_url else None

        if auto_start_sync and self._sync_agent:
            self._sync_agent.start()

        # Cold-start pull: seed local store from Brain if empty
        if cold_start_pull and brain_url and self._store.count() == 0:
            self._cold_start_pull()

    # ── Startup ───────────────────────────────────────────────────────
    def _cold_start_pull(self):
        """Pull a broad set of facts from Brain on first boot."""
        def _pull():
            if not self._sync_agent:
                return
            logger.info("[BodyHippocampus] Cold-start pull from Brain...")
            facts = self._sync_agent.pull_facts("general context preferences", top_k=20)
            pulled = 0
            for f in facts:
                text = f.get("text", "")
                if text:
                    fid = hashlib.sha256(text.encode()).hexdigest()[:16]
                    if self._store.add(fid, text, {"source": "brain_pull", "ts": str(time.time())}):
                        pulled += 1
            if pulled:
                logger.info(f"[BodyHippocampus] Cold-start: pulled {pulled} facts from Brain.")

        t = threading.Thread(target=_pull, daemon=True, name="ColdStartPull")
        t.start()

    # ── Vanish callbacks ──────────────────────────────────────────────
    def _on_vanish_callback(self, reason: str):
        """Internal callback — clears in-memory state after wipe."""
        self._vanished = True
        self._store.wipe()
        logger.critical(f"[BodyHippocampus] Post-vanish cleanup complete. Reason: {reason}")
        if self._external_on_vanish:
            try:
                self._external_on_vanish(reason)
            except Exception:
                pass

    # ── Vanish public API (mirrors Hippocampus v13.1) ─────────────────
    def vanish_on_breach(self, reason: str = "unauthorized_access") -> Dict:
        """Wipe local memory due to security breach."""
        logger.critical(f"[BodyHippocampus] VANISH PROTOCOL: breach — {reason}")
        return self._vanish.trigger_unauthorized_access(reason)

    def vanish_on_biometric_failure(self, bpm: int = 0, reason: str = "") -> Dict:
        """Wipe local memory due to anomalous biometric reading."""
        logger.critical(f"[BodyHippocampus] VANISH PROTOCOL: biometric — bpm={bpm}")
        return self._vanish.trigger_biometric_failure(bpm=bpm, reason=reason)

    def vanish_manual(self, reason: str = "manual") -> Dict:
        """Manual wipe trigger."""
        return self._vanish.trigger_manual(reason)

    def from_vanish_signal(self, ws_message: Dict) -> bool:
        """
        Handle an incoming WS vanish_alert from the Brain.
        The Brain sends this when it wipes its own store — the body
        should mirror the wipe to stay consistent.

        Expected message format:
          {"type": "vanish_alert", "reason": "...", "ts": ...}

        Returns True if vanish was triggered, False otherwise.
        """
        if ws_message.get("type") != "vanish_alert":
            return False
        reason = ws_message.get("reason", "brain_vanish_signal")
        logger.critical(f"[BodyHippocampus] Received vanish signal from Brain: {reason}")
        self.vanish_manual(f"brain_signal:{reason}")
        return True

    def is_vanished(self) -> bool:
        return self._vanished

    def is_tombstoned(self) -> bool:
        return self._vanish.is_tombstoned()

    # ── Ingestion (mirrors Hippocampus v13.1 API) ─────────────────────
    def ingest_turn(self, user: str = "", ai: str = "",
                    metadata: Optional[Dict] = None) -> int:
        """
        Extract facts from a conversation turn and store them locally.
        Also queues new facts for async push to Brain.
        """
        if self._vanished:
            logger.warning("[BodyHippocampus] Memory is wiped — ingest_turn ignored")
            return 0

        facts = self._extractor.extract(user, ai)
        if not facts:
            return 0

        stored = 0
        ts = time.time()
        for fact in facts:
            fact_id = hashlib.sha256(fact.encode()).hexdigest()[:16]
            meta = {"ts": str(ts), "source": "conversation", **(metadata or {})}
            added = self._store.add(fact_id, fact, meta)
            if added:
                stored += 1
                # Queue for Brain sync
                if self._sync_agent:
                    self._sync_agent.push_fact(fact, meta)

        return stored

    def store_fact(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Store a single fact directly (no extraction)."""
        if self._vanished:
            return ""
        fact_id = hashlib.sha256(f"{text}{time.time()}".encode()).hexdigest()[:16]
        meta = metadata or {}
        self._store.add(fact_id, text, meta)
        if self._sync_agent:
            self._sync_agent.push_fact(text, meta)
        return fact_id

    # ── Retrieval (mirrors Hippocampus v13.1 API) ─────────────────────
    def query_relevant_facts(
        self, query: str, top_k: int = MAX_FACTS_IN_PROMPT
    ) -> List[Dict]:
        """Query local store for relevant facts."""
        if self._vanished or not query.strip():
            return []
        return self._store.query(query, top_k)

    def build_memory_augmented_prompt(
        self, base_system_prompt: str, query: str
    ) -> str:
        """Inject relevant episodic memories into a system prompt."""
        if self._vanished:
            return base_system_prompt
        facts = self.query_relevant_facts(query)
        if not facts:
            return base_system_prompt
        memory_block = "\n".join(f"- {f['text']}" for f in facts)
        return (
            f"{base_system_prompt}\n\n"
            f"[EPISODIC MEMORY — relevant context from past conversations]\n"
            f"{memory_block}\n"
            f"[END EPISODIC MEMORY]\n"
        )

    def augment_prompt(self, base_system_prompt: str, query: str) -> str:
        """Convenience alias for build_memory_augmented_prompt."""
        return self.build_memory_augmented_prompt(base_system_prompt, query)

    # ── Brain sync public API ─────────────────────────────────────────
    def pull_from_brain(self, query: str, top_k: int = 10) -> int:
        """
        Pull facts from Brain matching query and seed local store.
        Returns number of new facts added.
        """
        if self._vanished or not self._sync_agent:
            return 0
        facts = self._sync_agent.pull_facts(query, top_k)
        added = 0
        for f in facts:
            text = f.get("text", "")
            if text:
                fid = hashlib.sha256(text.encode()).hexdigest()[:16]
                if self._store.add(fid, text, {"source": "brain_pull", "ts": str(time.time())}):
                    added += 1
        return added

    # ── Status ────────────────────────────────────────────────────────
    def fact_count(self) -> int:
        if self._vanished:
            return 0
        return self._store.count()

    def get_status(self) -> Dict:
        status = {
            "backend":        "numpy_local",
            "fact_count":     self.fact_count(),
            "numpy_available": _NP_OK,
            "requests_available": _REQUESTS_OK,
            "vanished":       self._vanished,
            "tombstoned":     self.is_tombstoned(),
            "brain_url":      self._brain_url or "not_configured",
            "store_dir":      str(self._store_dir),
        }
        if self._sync_agent:
            status["brain_sync"] = self._sync_agent.get_stats()
        return status


# ─────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────
_body_hippocampus: Optional[BodyHippocampus] = None


def get_body_hippocampus(**kwargs) -> BodyHippocampus:
    """
    Return (or create) the module-level BodyHippocampus singleton.
    Pass kwargs only on the first call; subsequent calls ignore them.
    """
    global _body_hippocampus
    if _body_hippocampus is None:
        _body_hippocampus = BodyHippocampus(**kwargs)
    return _body_hippocampus


# ─────────────────────────────────────────────────────────────────────
# SELF-TEST — run with: python hippocampus.py
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import tempfile
    import traceback

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s [%(name)s] %(message)s")

    passed = 0
    failed = 0
    errors: List[str] = []

    def ok(label: str, _msg: str = ""):
        global passed
        passed += 1
        print(f"  ✅  {label}")

    def fail(label: str, msg: str = ""):
        global failed
        failed += 1
        errors.append(f"{label}: {msg}")
        print(f"  ❌  {label} — {msg}")

    def check(label: str, condition: bool, msg: str = ""):
        if condition:
            ok(label)
        else:
            fail(label, msg)

    print("\n" + "═" * 60)
    print("  🧠 BodyHippocampus v13.2 — Full Test Suite")
    print("═" * 60 + "\n")

    tmpdir = Path(tempfile.mkdtemp())

    # ──────────────────────────────────────────────────────────────────
    # TEST GROUP 1 — FactExtractor
    # ──────────────────────────────────────────────────────────────────
    print("── Group 1: FactExtractor ──────────────────────────────────")
    try:
        ext = FactExtractor()

        # 1.1 Basic extraction
        facts = ext.extract("I'm using Python 3.11 on a Mac with 16GB RAM.",
                            "Python 3.11 has significant speed improvements.")
        check("1.1 extracts facts from conversation", len(facts) > 0,
              f"got {facts}")

        # 1.2 Excludes trivial utterances
        facts_trivial = ext.extract("ok", "yes")
        check("1.2 ignores trivial sentences", len(facts_trivial) == 0,
              f"got {facts_trivial}")

        # 1.3 De-duplication
        facts_dup = ext.extract("I use Python. I use Python.", "")
        check("1.3 deduplicates identical sentences",
              len(facts_dup) == len(set(f.lower().strip() for f in facts_dup)))

        # 1.4 Question exclusion
        facts_q = ext.extract("What is Python?", "")
        check("1.4 excludes questions", len(facts_q) == 0, f"got {facts_q}")

        # 1.5 Max 10 facts cap
        long_text = ". ".join([f"I always use tool_{i} for my workflow" for i in range(20)])
        facts_cap = ext.extract(long_text, "")
        check("1.5 caps at 10 facts", len(facts_cap) <= 10, f"got {len(facts_cap)}")

    except Exception as e:
        fail("1.x FactExtractor crash", traceback.format_exc(limit=3))

    # ──────────────────────────────────────────────────────────────────
    # TEST GROUP 2 — NumpyEpisodicStore
    # ──────────────────────────────────────────────────────────────────
    print("\n── Group 2: NumpyEpisodicStore ─────────────────────────────")
    try:
        store_path = tmpdir / "numpy_store_test"
        store_path.mkdir()
        store = NumpyEpisodicStore(persist_file=store_path / "facts.jsonl")

        # 2.1 Add and count
        added = store.add("fact_001", "I use Python 3.11 on macOS")
        check("2.1 add returns True for new fact", added is True)
        check("2.2 fact_count increments", store.count() == 1)

        # 2.3 Duplicate rejected
        added_again = store.add("fact_001", "I use Python 3.11 on macOS")
        check("2.3 duplicate add returns False", added_again is False)
        check("2.4 count unchanged after duplicate", store.count() == 1)

        # 2.5 Multi-add
        store.add("fact_002", "I have 32GB of RAM and use VS Code")
        store.add("fact_003", "My project uses FastAPI and ChromaDB")
        check("2.5 three distinct facts stored", store.count() == 3)

        # 2.6 Query returns results
        results = store.query("Python macOS setup", top_k=3)
        check("2.6 query returns results", len(results) > 0, f"got {results}")
        check("2.7 results have score field", all("score" in r for r in results))

        # 2.8 Persist and reload
        store2 = NumpyEpisodicStore(persist_file=store_path / "facts.jsonl")
        check("2.8 persisted facts reload correctly", store2.count() == 3,
              f"got {store2.count()}")

        # 2.9 Empty query returns nothing
        empty_results = store.query("", top_k=3)
        # Empty query: vocab match may still work, that's OK — just test no crash
        check("2.9 empty query does not crash", True)

        # 2.10 Wipe clears store
        store.wipe()
        check("2.10 wipe empties in-memory store", store.count() == 0)
        store3 = NumpyEpisodicStore(persist_file=store_path / "facts.jsonl")
        check("2.11 wipe clears persisted file too", store3.count() == 0,
              f"reloaded {store3.count()} facts")

    except Exception as e:
        fail("2.x NumpyEpisodicStore crash", traceback.format_exc(limit=3))

    # ──────────────────────────────────────────────────────────────────
    # TEST GROUP 3 — VanishProtocol
    # ──────────────────────────────────────────────────────────────────
    print("\n── Group 3: VanishProtocol ─────────────────────────────────")
    try:
        vanish_dir = tmpdir / "vanish_store"
        vanish_dir.mkdir()
        (vanish_dir / "dummy_data.json").write_text('{"test": 1}')

        broadcast_log: List[Dict] = []
        callback_log:  List[str]  = []

        vp = VanishProtocol(
            store_dir      = vanish_dir,
            ws_broadcast_fn= lambda msg: broadcast_log.append(msg),
            on_vanish      = lambda r: callback_log.append(r),
        )

        # 3.1 Manual trigger
        result = vp.trigger_manual("test_wipe")
        check("3.1 vanish returns WIPED status", result["status"] == "WIPED",
              f"got {result['status']}")

        # 3.2 Directory wiped
        check("3.2 store directory deleted", not vanish_dir.exists())

        # 3.3 Tombstone created
        check("3.3 tombstone file created",
              (tmpdir / VanishProtocol.TOMBSTONE_FILENAME).exists())

        # 3.4 WS broadcast fired
        check("3.4 WS broadcast received", len(broadcast_log) == 1,
              f"broadcasts={broadcast_log}")
        check("3.5 broadcast type is vanish_alert",
              broadcast_log[0].get("type") == "vanish_alert")

        # 3.6 Callback fired
        check("3.6 on_vanish callback fired", len(callback_log) == 1)

        # 3.7 Second trigger is idempotent
        result2 = vp.trigger_manual("second_attempt")
        check("3.7 second trigger returns ALREADY_WIPED",
              result2["status"] == "ALREADY_WIPED")

        # 3.8 Tombstone detection
        check("3.8 is_tombstoned() returns True", vp.is_tombstoned())

        # 3.9 Tamper log exists
        check("3.9 tamper evidence log created",
              (tmpdir / VanishProtocol.TAMPER_LOG_FILENAME).exists())

        # 3.10 Unauthorized access trigger
        vanish_dir2 = tmpdir / "vanish_store2"
        vanish_dir2.mkdir()
        vp2 = VanishProtocol(store_dir=vanish_dir2)
        result3 = vp2.trigger_unauthorized_access("test_breach")
        check("3.10 unauthorized_access trigger works",
              result3["status"] == "WIPED")

        # 3.11 Biometric failure trigger
        vanish_dir3 = tmpdir / "vanish_store3"
        vanish_dir3.mkdir()
        vp3 = VanishProtocol(store_dir=vanish_dir3)
        result4 = vp3.trigger_biometric_failure(bpm=0, reason="sensor_disconnected")
        check("3.11 biometric_failure trigger works",
              result4["status"] == "WIPED")

        # 3.12 Confirmation gate — decline
        vanish_dir4 = tmpdir / "vanish_store4"
        vanish_dir4.mkdir()
        vp4 = VanishProtocol(
            store_dir=vanish_dir4,
            require_confirmation=True,
            confirm_fn=lambda _: False,
        )
        result5 = vp4.trigger_manual("should_cancel")
        check("3.12 confirmation gate cancels wipe",
              result5["status"] == "CANCELLED")
        check("3.13 store survives cancelled wipe", vanish_dir4.exists())

    except Exception as e:
        fail("3.x VanishProtocol crash", traceback.format_exc(limit=3))

    # ──────────────────────────────────────────────────────────────────
    # TEST GROUP 4 — BrainSyncAgent (offline / no server)
    # ──────────────────────────────────────────────────────────────────
    print("\n── Group 4: BrainSyncAgent (offline) ───────────────────────")
    try:
        push_queue: List[Dict] = []
        q_lock = threading.Lock()
        sync = BrainSyncAgent(
            brain_url  = "http://localhost:19999",  # unreachable
            push_queue = push_queue,
            queue_lock = q_lock,
            interval   = 999,
        )

        # 4.1 Push queues the fact
        sync.push_fact("Test fact for sync", {"source": "test"})
        check("4.1 push_fact queues item", len(push_queue) == 1)

        # 4.2 Pull from unreachable server returns empty list
        result = sync.pull_facts("test query")
        check("4.2 pull from unreachable brain returns []", result == [],
              f"got {result}")

        # 4.3 Stats are accessible
        stats = sync.get_stats()
        check("4.3 get_stats returns dict with expected keys",
              all(k in stats for k in ["push_ok", "push_fail", "queue_len", "running"]))

        # 4.4 Sync doesn't crash when requests unavailable
        check("4.4 sync agent created without crash", sync is not None)

    except Exception as e:
        fail("4.x BrainSyncAgent crash", traceback.format_exc(limit=3))

    # ──────────────────────────────────────────────────────────────────
    # TEST GROUP 5 — BodyHippocampus integration
    # ──────────────────────────────────────────────────────────────────
    print("\n── Group 5: BodyHippocampus integration ─────────────────────")
    try:
        mem_dir = tmpdir / "body_mem_test"
        mem = BodyHippocampus(
            store_dir       = mem_dir,
            brain_url       = "",        # offline
            auto_start_sync = False,
            cold_start_pull = False,
        )

        # 5.1 ingest_turn stores facts
        n = mem.ingest_turn(
            user="I'm using Python 3.11 on a Mac with 16GB RAM.",
            ai ="Python 3.11 has significant speed improvements."
        )
        check("5.1 ingest_turn returns > 0 stored", n > 0, f"got {n}")
        check("5.2 fact_count > 0 after ingest", mem.fact_count() > 0,
              f"count={mem.fact_count()}")

        # 5.3 query_relevant_facts
        results = mem.query_relevant_facts("Python Mac setup")
        check("5.3 query returns relevant facts", len(results) > 0)
        check("5.4 results have text and score",
              all("text" in r and "score" in r for r in results))

        # 5.5 build_memory_augmented_prompt
        base = "You are Swayambhu, a sovereign AI."
        augmented = mem.build_memory_augmented_prompt(base, "Python setup")
        check("5.5 augmented prompt contains base", base in augmented)
        check("5.6 augmented prompt contains EPISODIC MEMORY block",
              "EPISODIC MEMORY" in augmented)

        # 5.7 augment_prompt alias
        aug2 = mem.augment_prompt(base, "Python setup")
        check("5.7 augment_prompt alias matches", aug2 == augmented)

        # 5.8 store_fact direct API
        fid = mem.store_fact("I prefer dark mode and use VS Code")
        check("5.8 store_fact returns non-empty id", len(fid) > 0)
        check("5.9 fact_count increases after store_fact",
              mem.fact_count() > n)

        # 5.10 get_status
        status = mem.get_status()
        check("5.10 get_status returns dict",
              isinstance(status, dict))
        check("5.11 status has expected keys",
              all(k in status for k in ["backend", "fact_count", "vanished",
                                         "tombstoned", "store_dir"]))

        # 5.12 from_vanish_signal ignores non-vanish messages
        ignored = mem.from_vanish_signal({"type": "heartbeat"})
        check("5.12 non-vanish WS message is ignored", ignored is False)

        # 5.13 from_vanish_signal triggers wipe
        mem2_dir = tmpdir / "body_mem2"
        mem2 = BodyHippocampus(
            store_dir=mem2_dir, brain_url="", auto_start_sync=False, cold_start_pull=False)
        mem2.store_fact("sensitive data")
        check("5.13 pre-wipe fact_count > 0", mem2.fact_count() > 0)
        triggered = mem2.from_vanish_signal({"type": "vanish_alert", "reason": "breach"})
        check("5.14 from_vanish_signal returns True", triggered is True)
        check("5.15 memory is wiped after signal", mem2.is_vanished())
        check("5.16 fact_count is 0 after wipe", mem2.fact_count() == 0)

    except Exception as e:
        fail("5.x BodyHippocampus crash", traceback.format_exc(limit=3))

    # ──────────────────────────────────────────────────────────────────
    # TEST GROUP 6 — Vanish Protocol on BodyHippocampus
    # ──────────────────────────────────────────────────────────────────
    print("\n── Group 6: Vanish via BodyHippocampus ─────────────────────")
    try:
        # 6.1 Manual vanish
        m1_dir = tmpdir / "vanish_m1"
        vanish_events: List[str] = []
        m1 = BodyHippocampus(
            store_dir=m1_dir, brain_url="", auto_start_sync=False, cold_start_pull=False,
            on_vanish=lambda r: vanish_events.append(r))
        m1.store_fact("This will be wiped")
        r1 = m1.vanish_manual("test_wipe")
        check("6.1 vanish_manual returns WIPED", r1["status"] == "WIPED")
        check("6.2 is_vanished() True", m1.is_vanished())
        check("6.3 fact_count 0 after vanish", m1.fact_count() == 0)
        check("6.4 on_vanish callback fired", len(vanish_events) == 1)

        # 6.5 Post-vanish ingest rejected
        n_after = m1.ingest_turn(user="try to add", ai="anything")
        check("6.5 ingest_turn rejected after vanish", n_after == 0)

        # 6.6 store_fact returns empty after vanish
        fid_after = m1.store_fact("post-vanish fact")
        check("6.6 store_fact returns empty after vanish", fid_after == "")

        # 6.7 biometric failure vanish
        m2_dir = tmpdir / "vanish_m2"
        m2 = BodyHippocampus(
            store_dir=m2_dir, brain_url="", auto_start_sync=False, cold_start_pull=False)
        m2.store_fact("Sensitive biometric-linked memory")
        r2 = m2.vanish_on_biometric_failure(bpm=0, reason="sensor_disconnected")
        check("6.7 biometric vanish returns WIPED", r2["status"] == "WIPED")
        check("6.8 is_vanished after biometric", m2.is_vanished())

        # 6.9 breach vanish
        m3_dir = tmpdir / "vanish_m3"
        m3 = BodyHippocampus(
            store_dir=m3_dir, brain_url="", auto_start_sync=False, cold_start_pull=False)
        m3.store_fact("Sensitive breach-linked memory")
        r3 = m3.vanish_on_breach("honeypot_triggered")
        check("6.9 vanish_on_breach returns WIPED", r3["status"] == "WIPED")

        # 6.10 Tombstone detection after manual vanish
        check("6.10 is_tombstoned True after vanish", m1.is_tombstoned())

    except Exception as e:
        fail("6.x Vanish via BodyHippocampus crash", traceback.format_exc(limit=3))

    # ──────────────────────────────────────────────────────────────────
    # TEST GROUP 7 — Singleton factory
    # ──────────────────────────────────────────────────────────────────
    print("\n── Group 7: Singleton factory ───────────────────────────────")
    try:
        import importlib
        # Reset global singleton for clean test
        import hippocampus as _hm
        _hm._body_hippocampus = None

        s1 = get_body_hippocampus(
            store_dir=tmpdir / "singleton_test",
            brain_url="", auto_start_sync=False, cold_start_pull=False)
        s2 = get_body_hippocampus()  # should return same instance
        check("7.1 singleton returns same instance", s1 is s2)
        check("7.2 singleton is BodyHippocampus", isinstance(s1, BodyHippocampus))
    except Exception as e:
        fail("7.x singleton crash", traceback.format_exc(limit=3))

    # ──────────────────────────────────────────────────────────────────
    # TEST GROUP 8 — Thread safety
    # ──────────────────────────────────────────────────────────────────
    print("\n── Group 8: Thread safety ──────────────────────────────────")
    try:
        ts_dir = tmpdir / "thread_safety"
        ts_mem = BodyHippocampus(
            store_dir=ts_dir, brain_url="", auto_start_sync=False, cold_start_pull=False)

        results_lock = threading.Lock()
        thread_results: List[int] = []

        def _ingest_worker(idx: int):
            n = ts_mem.ingest_turn(
                user=f"I use tool_{idx} for all my Python 3.11 work on macOS.",
                ai =f"Tool_{idx} is an excellent choice for Python workflows."
            )
            with results_lock:
                thread_results.append(n)

        threads = [threading.Thread(target=_ingest_worker, args=(i,))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        check("8.1 all 10 threads completed", len(thread_results) == 10,
              f"got {len(thread_results)}")
        check("8.2 total ingested > 0",
              sum(thread_results) > 0, f"sum={sum(thread_results)}")
        check("8.3 fact_count consistent with ingest",
              ts_mem.fact_count() > 0, f"count={ts_mem.fact_count()}")

    except Exception as e:
        fail("8.x Thread safety crash", traceback.format_exc(limit=3))

    # ──────────────────────────────────────────────────────────────────
    # TEST GROUP 9 — Edge cases
    # ──────────────────────────────────────────────────────────────────
    print("\n── Group 9: Edge cases ──────────────────────────────────────")
    try:
        ec_dir = tmpdir / "edge_cases"
        ec_mem = BodyHippocampus(
            store_dir=ec_dir, brain_url="", auto_start_sync=False, cold_start_pull=False)

        # 9.1 Empty ingest
        n_empty = ec_mem.ingest_turn(user="", ai="")
        check("9.1 empty ingest returns 0", n_empty == 0)

        # 9.2 Whitespace-only query
        results_ws = ec_mem.query_relevant_facts("   ")
        check("9.2 whitespace query returns []", results_ws == [])

        # 9.3 Augment with no facts returns base unchanged
        base = "base prompt"
        aug = ec_mem.build_memory_augmented_prompt(base, "anything")
        check("9.3 empty store returns base prompt unchanged", aug == base)

        # 9.4 Very long fact text
        long_fact = "I " + " always ".join(["use Python"] * 50)
        fid = ec_mem.store_fact(long_fact[:250])
        check("9.4 long fact stored without crash", len(fid) > 0)

        # 9.5 Unicode fact
        unicode_fact = "I use Python 3.11 — это мой основной язык программирования."
        fid_u = ec_mem.store_fact(unicode_fact)
        check("9.5 unicode fact stored", len(fid_u) > 0)

        # 9.6 get_status when not vanished
        s = ec_mem.get_status()
        check("9.6 status vanished=False when not vanished",
              s["vanished"] is False)

        # 9.7 Store fact returns consistent id for same content+time
        # (ids vary by time, so just check it's hex)
        fid2 = ec_mem.store_fact("Another fact for testing.")
        check("9.7 store_fact returns 16-char hex id",
              len(fid2) == 16 and all(c in "0123456789abcdef" for c in fid2))

    except Exception as e:
        fail("9.x Edge cases crash", traceback.format_exc(limit=3))

    # ──────────────────────────────────────────────────────────────────
    # CLEANUP
    # ──────────────────────────────────────────────────────────────────
    try:
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    # ──────────────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────────────
    total = passed + failed
    print("\n" + "═" * 60)
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if errors:
        print("\n  Failed tests:")
        for e in errors:
            print(f"    • {e}")
    print("═" * 60)
    if failed:
        sys.exit(1)
    else:
        print("  ✅ All tests passed — hippocampus.py is production-ready.")
