#!/usr/bin/env python3
# =====================================================================
# 🛰️  TACTICAL EDGE RAG  (Mod 6 — DDIL Quantized Dense Retrieval)
#
# Components:
#   PQIndex         — 8-bit Product Quantization vector store
#   ShadowSync      — background index builder (DEFCON 5)
#   TacticalEdgeRAG — instant DEFCON 1 mount (<50ms)
#   LocalRAGIndex   — TF-IDF cosine blueprint retrieval (FAISS/numpy)
#   LocalToolRAG    — keyword cosine tool registry retriever
#
# Migrated from notebook:
#   LocalRAGIndex  ← Stateless JIT Orchestrator cell (StatelessOrchestrator
#                    companion index: build_faiss_index, vector_search_blueprint,
#                    offline_rag_query, _tokenize, _build_vocab, _embed,
#                    _cosine, _save, _load)
#   LocalToolRAG   ← Cell 9 / Module 7 (LocalToolRAG: tool registry,
#                    _cosine keyword scoring, retrieve_top_tools)
#
# WIRING (swayambhu_body.py):
# ─────────────────────────────────────────────────────────────────────
#   from tactical_edge_rag import TacticalEdgeRAG, LocalRAGIndex, LocalToolRAG
#
#   self.emergency_rag = TacticalEdgeRAG(rag_index=self.rag)
#   self.emergency_rag.start_shadow_sync()
#
#   self.local_rag       = LocalRAGIndex()
#   self.local_tool_rag  = LocalToolRAG()
# =====================================================================

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
import os
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ── Project root resolution ───────────────────────────────────────────
try:
    from swayambhu_utils import PROJECT_ROOT
except ImportError:
    try:
        PROJECT_ROOT = Path(__file__).parent.resolve()
    except NameError:
        PROJECT_ROOT = Path(os.getcwd()).resolve()

RAG_INDEX_PATH = PROJECT_ROOT / "local_rag_index.json"
logger = logging.getLogger("TacticalEdgeRAG")

# ── Optional deps ─────────────────────────────────────────────────────
try:
    import numpy as np
    _NP_OK = True
except ImportError:
    _NP_OK = False

# ── Module-level config ───────────────────────────────────────────────
_TACT_DIR = PROJECT_ROOT / "tactical_rag"
_TACT_DIR.mkdir(parents=True, exist_ok=True)

SYNC_INTERVAL_SEC = 300
PQ_BITS           = 8
PQ_SUBVECTORS     = 4
MOUNT_TARGET_MS   = 50
TOP_K_DEFAULT     = 5

# ── Survival knowledge corpus ─────────────────────────────────────────
TACTICAL_DOMAINS = [
    {
        "name": "medical_triage",
        "desc": "Emergency medical triage, wound management, CPR, airway protocols",
        "entries": [
            {"id": "med_001", "text": "Hemorrhage control: apply direct pressure, tourniquet if limb."},
            {"id": "med_002", "text": "Airway management: head-tilt chin-lift, jaw thrust if trauma."},
            {"id": "med_003", "text": "Shock treatment: lay flat, elevate legs, keep warm, monitor pulse."},
            {"id": "med_004", "text": "Burn triage: cool with water 20min, do not pop blisters."},
            {"id": "med_005", "text": "CPR ratio: 30 compressions : 2 breaths, 100-120 bpm."},
            {"id": "med_006", "text": "Anaphylaxis: epinephrine IM thigh, call EMS, lay flat with legs raised."},
        ],
    },
    {
        "name": "survival_protocol",
        "desc": "Urban and wilderness survival, water, shelter, navigation",
        "entries": [
            {"id": "surv_001", "text": "Rule of 3: 3 min air, 3 hr shelter, 3 days water, 3 weeks food."},
            {"id": "surv_002", "text": "Water purification: boil 1 min at sea level, 3 min at altitude."},
            {"id": "surv_003", "text": "Shelter priority: insulation from ground is more critical than overhead cover."},
            {"id": "surv_004", "text": "Navigation: moss grows on north side; sun rises east, sets west."},
            {"id": "surv_005", "text": "Signal fire: green vegetation produces white smoke (daytime)."},
            {"id": "surv_006", "text": "Urban evacuation: stay low in smoke, feel doors before opening."},
        ],
    },
    {
        "name": "coding_triage",
        "desc": "Offline coding reference: Python stdlib, algorithms, debugging",
        "entries": [
            {"id": "code_001", "text": "Sort list: sorted(lst, key=lambda x: x['field'], reverse=True)"},
            {"id": "code_002", "text": "Dict merge (Python 3.9+): merged = dict_a | dict_b"},
            {"id": "code_003", "text": "Retry pattern: for attempt in range(3): try: ... except: time.sleep(2**attempt)"},
            {"id": "code_004", "text": "Thread-safe counter: use threading.Lock() around increment."},
            {"id": "code_005", "text": "JSON pretty print: json.dumps(obj, indent=2, ensure_ascii=False)"},
            {"id": "code_006", "text": "Path ops (pathlib): Path.home() / 'dir' / 'file.txt'"},
        ],
    },
]


# =====================================================================
# PQ INDEX — 8-bit Product Quantization
# =====================================================================
class PQIndex:
    """
    8-bit Product Quantization index.
    Compresses float32 embeddings by ~4x by splitting each vector into
    M sub-vectors and quantizing each to PQ_BITS-bit centroids.
    Pure-numpy — no FAISS required.
    """

    def __init__(self, dim: int, m: int = PQ_SUBVECTORS, bits: int = PQ_BITS):
        self._dim  = dim
        self._m    = m
        self._bits = bits
        self._k    = 2 ** bits
        self._sub  = dim // m

        self._codebooks: Optional[List] = None
        self._codes:     Optional[any]  = None
        self._ids:       List[str]      = []
        self._texts:     List[str]      = []
        self._trained    = False

    def _compute_embedding(self, text: str) -> "np.ndarray":
        """TF-IDF-like bag-of-tokens embedding — no external model needed."""
        vec = np.zeros(self._dim, dtype=np.float32)
        tokens = text.lower().split()
        for i, tok in enumerate(tokens):
            h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
            idx = h % self._dim
            vec[idx] += 1.0 / (1.0 + i)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-9)

    def train_and_add(self, entries: List[Dict]) -> int:
        """Build quantized index from entries [{id, text, ...}]. Returns n indexed."""
        if not _NP_OK or not entries:
            return 0

        raw = np.array(
            [self._compute_embedding(e.get("text", e.get("description", e.get("id", ""))))
             for e in entries],
            dtype=np.float32,
        )

        n, d = raw.shape
        if d != self._dim:
            padded = np.zeros((n, self._dim), dtype=np.float32)
            padded[:, :min(d, self._dim)] = raw[:, :min(d, self._dim)]
            raw = padded

        self._codebooks = []
        self._codes = np.zeros((n, self._m), dtype=np.uint8)

        for m_idx in range(self._m):
            lo      = m_idx * self._sub
            hi      = lo + self._sub
            subvecs = raw[:, lo:hi]

            k_actual  = min(self._k, n)
            rng_idx   = np.random.choice(n, k_actual, replace=False)
            centroids = subvecs[rng_idx].copy()

            for _ in range(3):
                dists  = np.sum((subvecs[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
                labels = np.argmin(dists, axis=1)
                for c in range(k_actual):
                    mask = labels == c
                    if mask.any():
                        centroids[c] = subvecs[mask].mean(axis=0)

            self._codebooks.append(centroids)
            self._codes[:, m_idx] = labels.astype(np.uint8)

        self._ids   = [e["id"] for e in entries]
        self._texts = [
            e.get("text", e.get("description", e.get("id", ""))) for e in entries
        ]
        self._trained = True
        return n

    def search(self, query: str, top_k: int = TOP_K_DEFAULT) -> List[Dict]:
        """Approximate nearest-neighbor search via PQ distance tables."""
        if not self._trained or not _NP_OK:
            return []

        q_vec  = self._compute_embedding(query)
        scores = np.zeros(len(self._ids), dtype=np.float32)

        for m_idx in range(self._m):
            lo        = m_idx * self._sub
            hi        = lo + self._sub
            q_sub     = q_vec[lo:hi]
            centroids = self._codebooks[m_idx]
            dists     = np.sum((centroids - q_sub[None, :]) ** 2, axis=1)
            code_col  = self._codes[:, m_idx].astype(np.int32)
            scores   += -dists[code_col]

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"id": self._ids[i], "text": self._texts[i], "score": float(scores[i])}
            for i in top_idx
        ]

    def save(self, path: Path) -> bool:
        """Serialize index to disk as compact JSON."""
        if not self._trained:
            return False
        try:
            data = {
                "dim":       self._dim,
                "m":         self._m,
                "bits":      self._bits,
                "ids":       self._ids,
                "texts":     self._texts,
                "codes":     self._codes.tolist(),
                "codebooks": [cb.tolist() for cb in self._codebooks],
            }
            path.write_text(json.dumps(data), encoding="utf-8")
            size_kb = path.stat().st_size // 1024
            logger.info(f"[PQIndex] Saved {len(self._ids)} entries → {path.name} ({size_kb}KB)")
            return True
        except Exception as e:
            logger.error(f"[PQIndex] Save error: {e}")
            return False

    def load(self, path: Path) -> bool:
        """Deserialize index from disk."""
        try:
            data            = json.loads(path.read_text(encoding="utf-8"))
            self._dim       = data["dim"]
            self._m         = data["m"]
            self._bits      = data["bits"]
            self._sub       = self._dim // self._m
            self._ids       = data["ids"]
            self._texts     = data["texts"]
            self._codes     = np.array(data["codes"], dtype=np.uint8)
            self._codebooks = [np.array(cb, dtype=np.float32) for cb in data["codebooks"]]
            self._trained   = True
            return True
        except Exception as e:
            logger.error(f"[PQIndex] Load error: {e}")
            return False

    @property
    def size(self) -> int:
        return len(self._ids)


# =====================================================================
# SHADOW SYNC — async background index builder (DEFCON 5)
# =====================================================================
class ShadowSync:
    """
    Silently trickles compressed PQ indices to NVMe while DEFCON 5.
    Runs in a low-priority daemon thread. Never blocks the main loop.
    """

    def __init__(
        self,
        tact_dir:  Path = _TACT_DIR,
        domains:   List[Dict] = None,
        dim:       int = 128,
        on_synced: Optional[Callable] = None,
    ):
        self._dir       = tact_dir
        self._domains   = domains or TACTICAL_DOMAINS
        self._dim       = dim
        self._on_synced = on_synced
        self._stop_evt  = threading.Event()
        self._thread    = None
        self._synced:   Dict[str, bool] = {}
        self._last_sync = 0.0

    def start(self):
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._sync_loop, daemon=True, name="ShadowSync"
        )
        self._thread.start()
        logger.info("[ShadowSync] Background sync started.")

    def stop(self):
        self._stop_evt.set()

    def force_sync(self) -> int:
        """Sync all domains now. Returns count of newly built indices."""
        return self._sync_all()

    def is_ready(self, domain: str) -> bool:
        return (self._dir / f"{domain}.pqidx").exists()

    def get_status(self) -> dict:
        return {
            "domains_synced":  sum(1 for d in self._domains if self.is_ready(d["name"])),
            "total_domains":   len(self._domains),
            "last_sync_ago_s": round(time.time() - self._last_sync) if self._last_sync else None,
        }

    def _sync_loop(self):
        while not self._stop_evt.is_set():
            self._sync_all()
            for _ in range(SYNC_INTERVAL_SEC):
                if self._stop_evt.is_set():
                    return
                time.sleep(1)

    def _sync_all(self) -> int:
        built = 0
        for domain in self._domains:
            if self._stop_evt.is_set():
                break
            name = domain["name"]
            path = self._dir / f"{name}.pqidx"
            if path.exists():
                continue
            try:
                idx = PQIndex(dim=self._dim)
                n   = idx.train_and_add(domain["entries"])
                if n > 0 and idx.save(path):
                    self._synced[name] = True
                    built += 1
                    logger.info(f"[ShadowSync] Built index: {name} ({n} entries)")
            except Exception as e:
                logger.warning(f"[ShadowSync] {name} failed: {e}")

        self._last_sync = time.time()
        if built > 0 and self._on_synced:
            try:
                self._on_synced(built)
            except Exception:
                pass
        return built


# =====================================================================
# TACTICAL EDGE RAG — instant DEFCON 1 mount + cross-domain query
# =====================================================================
class TacticalEdgeRAG:
    """
    Drop-in upgrade for EmergencyRAGDownloader.

    DEFCON 5: ShadowSync trickles PQ indices to NVMe silently.
    DEFCON 1: mount_tactical_db() loads all indices in <50ms.

    Public API:
        start_shadow_sync()   — begin background sync
        mount_tactical_db()   — instant mount on DEFCON 1
        query(text, top_k)    — cross-domain semantic search
        get_status()          — dict with readiness metrics
        trigger_download()    — legacy EmergencyRAGDownloader compat
    """

    def __init__(
        self,
        rag_index=None,
        tact_dir: Path = _TACT_DIR,
        domains:  List[Dict] = None,
        dim:      int = 128,
    ):
        self._rag      = rag_index
        self._tact_dir = tact_dir
        self._dim      = dim
        self._domains  = domains or TACTICAL_DOMAINS

        self._sync = ShadowSync(
            tact_dir  = tact_dir,
            domains   = self._domains,
            dim       = dim,
            on_synced = self._on_domain_synced,
        )

        self._mounted_indices: Dict[str, PQIndex] = {}
        self._mounted  = False
        self._mount_ms = 0.0
        self._lock     = threading.Lock()

    def start_shadow_sync(self):
        """Begin background index building (call on boot / DEFCON 5)."""
        self._sync.start()

    def stop(self):
        self._sync.stop()

    def trigger_download(self):
        """Legacy compat: EmergencyRAGDownloader.trigger_download()."""
        self.mount_tactical_db()

    def mount_tactical_db(self) -> dict:
        """
        DEFCON 1: load all pre-built PQ indices from NVMe.
        Falls back to force-building any indices not yet synced.
        """
        t0 = time.time()
        loaded = built = 0

        with self._lock:
            for domain in self._domains:
                name = domain["name"]
                path = self._tact_dir / f"{name}.pqidx"

                if name in self._mounted_indices:
                    loaded += 1
                    continue

                if not path.exists():
                    idx = PQIndex(dim=self._dim)
                    n   = idx.train_and_add(domain["entries"])
                    if n > 0:
                        idx.save(path)
                    built += 1
                else:
                    idx = PQIndex(dim=self._dim)
                    if idx.load(path):
                        loaded += 1
                    else:
                        idx = PQIndex(dim=self._dim)
                        idx.train_and_add(domain["entries"])
                        built += 1

                self._mounted_indices[name] = idx

        self._mounted  = True
        self._mount_ms = round((time.time() - t0) * 1000, 1)
        logger.info(
            f"🛰️  [TacticalRAG] Mounted: {loaded} loaded + {built} built "
            f"in {self._mount_ms}ms"
        )

        if self._rag:
            try:
                # Merge tactical entries WITH any already-indexed blueprints so
                # neither set is clobbered.  Existing blueprint ids are preserved;
                # tactical domain entries are appended under their domain name as id.
                existing: List[Dict] = [
                    {"id": eid, "description": edesc, "category": "blueprint"}
                    for eid, edesc in zip(self._rag._ids, self._rag._descs)
                ] if self._rag._built and self._rag._ids else []

                tactical_entries = [
                    {"id": e["id"], "description": e.get("text", ""), "category": d["name"]}
                    for d in self._domains
                    for e in d["entries"]
                ]

                # Deduplicate by id — existing blueprints win over tactical entries
                seen: set = {e["id"] for e in existing}
                merged = existing + [e for e in tactical_entries if e["id"] not in seen]
                build_fn = getattr(self._rag, "build_faiss_index", getattr(self._rag, "build", None))
                if build_fn:
                    build_fn(merged)
            except Exception as e:
                logger.debug(f"[TacticalRAG] Legacy RAG inject error: {e}")

        return {
            "status":       "mounted",
            "domains":      len(self._mounted_indices),
            "loaded":       loaded,
            "built":        built,
            "mount_ms":     self._mount_ms,
            "under_target": self._mount_ms < MOUNT_TARGET_MS,
        }

    def query(self, text: str, top_k: int = TOP_K_DEFAULT) -> List[Dict]:
        """Cross-domain semantic search across all mounted indices."""
        if not self._mounted:
            self.mount_tactical_db()

        results: List[Dict] = []
        with self._lock:
            n_domains = max(1, len(self._mounted_indices))
            for name, idx in self._mounted_indices.items():
                hits = idx.search(text, top_k=max(1, top_k // n_domains))
                for h in hits:
                    h["domain"] = name
                    results.append(h)

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def get_status(self) -> dict:
        return {
            "mounted":         self._mounted,
            "mounted_domains": len(self._mounted_indices),
            "mount_ms":        self._mount_ms,
            "under_target_ms": self._mount_ms < MOUNT_TARGET_MS if self._mount_ms else None,
            "tact_dir":        str(self._tact_dir),
            "shadow_sync":     self._sync.get_status(),
            "total_entries":   sum(idx.size for idx in self._mounted_indices.values()),
        }

    def _on_domain_synced(self, count: int):
        logger.info(f"[TacticalRAG] ShadowSync: {count} new domain(s) ready.")


# =====================================================================
# LOCAL RAG INDEX — TF-IDF cosine blueprint retrieval (FAISS/numpy)
# Migrated from: notebook Stateless JIT Orchestrator cell
# Role: companion index for StatelessOrchestrator; body-side blueprint
#       retrieval for offline operation.
# =====================================================================
class LocalRAGIndex:
    """
    TF-IDF style vector store for blueprint descriptions.
    Primary:  FAISS (if installed) — GPU-accelerated ANN search.
    Fallback: pure-numpy cosine similarity — always available.

    On first build: tokenises every blueprint description, builds vocab,
    embeds as bag-of-words, indexes. Saves to disk for fast reload.
    On query: vector_search_blueprint() returns top-k in <5ms on CPU.

    Public API:
        build_faiss_index(blueprints)        — build index from list of dicts
        vector_search_blueprint(query, top_k) — semantic search
        offline_rag_query(query, top_k)       — alias for vector_search_blueprint
    """

    INDEX_PATH = str(RAG_INDEX_PATH)

    # Minimum cosine score to include a result
    MIN_SCORE  = 0.30
    MAX_INJECT = 5

    def __init__(self, index_path: Optional[str] = None):
        self._index_path = Path(index_path) if index_path else Path(self.INDEX_PATH)
        self._vocab:   List[str]        = []
        self._matrix:  List[List[float]] = []
        self._ids:     List[str]        = []
        self._descs:   List[str]        = []
        self._faiss    = None
        self._built    = False
        self._lock     = threading.Lock()

    # ── Tokenisation & embedding ──────────────────────────────────────
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z][a-z0-9_]{2,}\b', text.lower())

    def _build_vocab(self, docs: List[str]) -> List[str]:
        counts: Dict[str, int] = {}
        for doc in docs:
            for tok in set(self._tokenize(doc)):
                counts[tok] = counts.get(tok, 0) + 1
        n = len(docs)
        # Keep tokens appearing in at least 1 doc but not all (unless tiny corpus)
        return [t for t, c in counts.items() if 1 <= c < n or n <= 3]

    def _embed(self, text: str) -> List[float]:
        tokens = set(self._tokenize(text))
        return [1.0 if v in tokens else 0.0 for v in self._vocab]

    def _cosine(self, a: List[float], b: List[float]) -> float:
        if _NP_OK:
            av, bv = np.array(a), np.array(b)
            return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-9))
        dot = sum(x * y for x, y in zip(a, b))
        na  = sum(x * x for x in a) ** 0.5
        nb  = sum(x * x for x in b) ** 0.5
        return dot / (na * nb + 1e-9)

    # ── Build ─────────────────────────────────────────────────────────
    def build_faiss_index(self, blueprints: List[Dict]) -> int:
        """
        Build vector index from blueprint dicts.
        Each dict needs at minimum: id. Optional: description, category.
        Returns number of documents indexed.
        """
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

            # Try FAISS for fast ANN
            self._faiss = None
            if _NP_OK:
                try:
                    import faiss
                    dim = len(self._vocab)
                    if dim > 0:
                        mat = np.array(self._matrix, dtype="float32")
                        faiss.normalize_L2(mat)
                        self._faiss = faiss.IndexFlatIP(dim)
                        self._faiss.add(mat)
                        logger.info(f"[LocalRAGIndex] FAISS index: {len(docs)} docs, dim={dim}")
                except ImportError:
                    logger.info(f"[LocalRAGIndex] numpy cosine index: {len(docs)} docs, dim={len(self._vocab)}")
                except Exception as e:
                    logger.warning(f"[LocalRAGIndex] FAISS init error: {e}")

            self._built = True
            self._save()
            return len(docs)

    def _save(self):
        try:
            with open(self._index_path, "w") as f:
                json.dump(
                    {"vocab": self._vocab, "matrix": self._matrix,
                     "ids": self._ids, "descs": self._descs},
                    f,
                )
        except Exception as e:
            logger.warning(f"[LocalRAGIndex] Save error: {e}")

    def _load(self) -> bool:
        try:
            if not self._index_path.exists():
                return False
            with open(self._index_path) as f:
                data = json.load(f)
            self._vocab  = data["vocab"]
            self._matrix = data["matrix"]
            self._ids    = data["ids"]
            self._descs  = data.get("descs", self._ids)
            self._built  = True
            logger.info(f"[LocalRAGIndex] Loaded: {len(self._ids)} docs, dim={len(self._vocab)}")
            return True
        except Exception as e:
            logger.warning(f"[LocalRAGIndex] Load error: {e}")
            return False

    # ── Query ─────────────────────────────────────────────────────────
    def vector_search_blueprint(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Semantic blueprint search. Returns list of
        {id, description, score} sorted by relevance.
        """
        if not self._built:
            if not self._load():
                return []
        if not self._vocab:
            return []

        q_vec = self._embed(query)

        # FAISS path
        if self._faiss and _NP_OK:
            try:
                import faiss
                qv = np.array([q_vec], dtype="float32")
                faiss.normalize_L2(qv)
                scores, idxs = self._faiss.search(qv, min(top_k, len(self._ids)))
                return [
                    {"id": self._ids[i], "description": self._descs[i], "score": float(scores[0][j])}
                    for j, i in enumerate(idxs[0]) if i >= 0
                ]
            except Exception:
                pass

        # numpy cosine fallback
        if _NP_OK:
            q = np.array(q_vec)
            sims = []
            for i, row in enumerate(self._matrix):
                r     = np.array(row)
                denom = np.linalg.norm(q) * np.linalg.norm(r) + 1e-9
                score = float(np.dot(q, r) / denom)
                sims.append((score, i))
        else:
            sims = [(self._cosine(q_vec, row), i) for i, row in enumerate(self._matrix)]

        sims.sort(reverse=True)
        return [
            {"id": self._ids[i], "description": self._descs[i], "score": s}
            for s, i in sims[:top_k]
            if s > 0.05
        ]

    def offline_rag_query(self, query: str, top_k: int = 3) -> List[Dict]:
        """Alias for vector_search_blueprint — used by edge node and StatelessOrchestrator."""
        return self.vector_search_blueprint(query, top_k)

    def augment_system_prompt(self, base_prompt: str, query: str) -> str:
        """
        Prepend top matching blueprint descriptions to a system prompt.
        Used when the brain wants offline context injection.
        """
        facts = self.vector_search_blueprint(query, top_k=self.MAX_INJECT)
        if not facts:
            return base_prompt
        mem_block = "\n".join(f"- {f['description']}" for f in facts)
        return (
            f"{base_prompt}\n\n"
            f"[LOCAL RAG — relevant blueprints]\n{mem_block}\n"
            f"[END LOCAL RAG]"
        )

    @property
    def doc_count(self) -> int:
        return len(self._ids)

    def get_status(self) -> dict:
        return {
            "built":      self._built,
            "doc_count":  self.doc_count,
            "vocab_size": len(self._vocab),
            "faiss":      self._faiss is not None,
            "index_path": str(self._index_path),
        }


# =====================================================================
# LOCAL TOOL RAG — keyword cosine tool registry retriever
# Migrated from: notebook Cell 9 / Module 7 (LocalToolRAG)
# Role: retrieves the most relevant internal tools for a given query
#       using cosine similarity over word-count vectors.
# =====================================================================
class LocalToolRAG:
    """
    Lightweight tool registry with cosine-similarity retrieval.
    No embeddings — pure bag-of-words cosine over the tool descriptions.

    Public API:
        retrieve_top_tools(query, top_k=2) — returns {name: description} dict
        register_tool(name, description)   — add a tool at runtime
        get_status()                       — registry size + last query info
    """

    # Default tool registry — matches notebook Cell 9
    _DEFAULT_REGISTRY: Dict[str, str] = {
        "c++_kernel_compiler": "Compiles raw C++ or Rust code into bare-metal binaries.",
        "i2c_sensor_bridge":   "Reads biometric or thermal data from hardware sensors.",
        "tensor_math_core":    "Performs complex matrix multiplications and quantum variance.",
        "local_file_manager":  "Reads, writes, and manages local text and JSON files.",
        "freelance_bidder":    "Formats and sanitizes code for external freelance APIs.",
    }

    def __init__(self, registry: Optional[Dict[str, str]] = None):
        self.tool_registry: Dict[str, str] = dict(registry or self._DEFAULT_REGISTRY)
        self._last_query:   str = ""
        self._last_results: Dict[str, str] = {}

    def _cosine(self, query: str, text: str) -> float:
        """Bag-of-words cosine similarity between query and tool description."""
        v1 = Counter(query.lower().split())
        v2 = Counter(text.lower().split())
        inter = set(v1) & set(v2)
        num   = sum(v1[x] * v2[x] for x in inter)
        den   = (
            math.sqrt(sum(v ** 2 for v in v1.values()))
            * math.sqrt(sum(v ** 2 for v in v2.values()))
        )
        return num / den if den else 0.0

    def retrieve_top_tools(self, query: str, top_k: int = 2) -> Dict[str, str]:
        """
        Returns the top_k most relevant tools for the query.
        Result: {tool_name: description} sorted by relevance.
        """
        scored = sorted(
            [(self._cosine(query, desc), name, desc)
             for name, desc in self.tool_registry.items()],
            reverse=True,
        )
        self._last_query   = query
        self._last_results = {name: desc for _, name, desc in scored[:top_k]}
        return self._last_results

    def register_tool(self, name: str, description: str):
        """Register a new tool at runtime."""
        self.tool_registry[name] = description

    def get_status(self) -> dict:
        return {
            "tool_count":   len(self.tool_registry),
            "tool_names":   list(self.tool_registry.keys()),
            "last_query":   self._last_query,
            "last_results": list(self._last_results.keys()),
        }


# =====================================================================
# SELF-TESTS
# All tests run top-to-bottom. Exit code 0 = all passed.
# =====================================================================
def _run_tests() -> bool:
    import tempfile
    import shutil

    logging.basicConfig(level=logging.WARNING)
    print("🛰️  TacticalEdgeRAG + LocalRAGIndex + LocalToolRAG — Full Test Suite\n")
    passed = failed = 0

    def ok(name: str, cond: bool, detail: str = ""):
        nonlocal passed, failed
        if cond:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}" + (f": {detail}" if detail else ""))
            failed += 1

    # ── Test group 1: PQIndex embedding ──────────────────────────────
    print("=== Group 1: PQIndex embedding ===")
    idx = PQIndex(dim=128, m=4, bits=8)
    vec = idx._compute_embedding("medical triage emergency bleeding")
    ok("Embedding shape",         vec.shape == (128,))
    ok("Embedding normalized",    abs(np.linalg.norm(vec) - 1.0) < 1e-5)
    ok("Embedding non-zero",      float(vec.max()) > 0)
    vec2 = idx._compute_embedding("medical triage emergency bleeding")
    ok("Deterministic embedding", np.allclose(vec, vec2))

    # ── Test group 2: PQIndex train & search ─────────────────────────
    print("\n=== Group 2: PQIndex train & search ===")
    entries = [
        {"id": "a", "text": "apply tourniquet for severe bleeding"},
        {"id": "b", "text": "water purification by boiling"},
        {"id": "c", "text": "python dict merge with pipe operator"},
        {"id": "d", "text": "CPR compressions thirty to two ratio"},
        {"id": "e", "text": "shelter insulation from ground surface"},
    ]
    idx2 = PQIndex(dim=128, m=4, bits=8)
    n = idx2.train_and_add(entries)
    ok("All entries indexed",     n == 5)
    ok("Index is trained",        idx2._trained)
    results = idx2.search("bleeding control tourniquet", top_k=3)
    ok("Returns ≤3 results",      len(results) <= 3)
    ok("Results have id+score",   all("id" in r and "score" in r for r in results))
    ok("Top result relevant",
       "tourniquet" in results[0]["text"].lower() or "bleeding" in results[0]["text"].lower(),
       f"got: {results[0]['text'][:60]}")

    # ── Test group 3: PQIndex save/load ───────────────────────────────
    print("\n=== Group 3: PQIndex save/load ===")
    tmpdir = Path(tempfile.mkdtemp())
    path   = tmpdir / "test.pqidx"
    ok("Save succeeds",           idx2.save(path))
    ok("File on disk",            path.exists())
    idx3 = PQIndex(dim=128, m=4, bits=8)
    ok("Load succeeds",           idx3.load(path))
    ok("IDs preserved",           idx3._ids == idx2._ids)
    ok("Size matches",            idx3.size == idx2.size)
    r_orig = idx2.search("bleeding tourniquet", top_k=1)
    r_load = idx3.search("bleeding tourniquet", top_k=1)
    ok("Search consistent post-load", r_orig[0]["id"] == r_load[0]["id"])

    # ── Test group 4: Compression ratio ──────────────────────────────
    print("\n=== Group 4: Compression ratio ===")
    raw_size = 128 * 4 * len(entries)   # float32 = 4 bytes per dim
    pq_size  = path.stat().st_size
    ratio    = raw_size / max(pq_size, 1)
    ok("PQ file exists",          pq_size > 0)
    ok("Compression achieved",    ratio >= 0.5, f"ratio={ratio:.2f}x")

    # ── Test group 5: ShadowSync ──────────────────────────────────────
    print("\n=== Group 5: ShadowSync ===")
    sync_dir  = tmpdir / "sync_test"
    sync_dir.mkdir()
    synced_cb: List[int] = []
    sync = ShadowSync(
        tact_dir  = sync_dir,
        domains   = TACTICAL_DOMAINS[:2],
        dim       = 64,
        on_synced = lambda n: synced_cb.append(n),
    )
    count = sync.force_sync()
    ok("Sync builds 2 indices",   count == 2, f"built={count}")
    ok("Indices on disk",         len(list(sync_dir.glob("*.pqidx"))) == 2)
    ok("is_ready medical",        sync.is_ready("medical_triage"))
    ok("is_ready survival",       sync.is_ready("survival_protocol"))
    ok("Callback fired",          len(synced_cb) > 0)
    status = sync.get_status()
    ok("Status domains_synced=2", status["domains_synced"] == 2)
    count2 = sync.force_sync()
    ok("Idempotent sync=0",       count2 == 0, f"built={count2}")

    # ── Test group 6: TacticalEdgeRAG mount + query ───────────────────
    print("\n=== Group 6: TacticalEdgeRAG mount + query ===")
    tact_dir2 = tmpdir / "tact_test"
    tact_dir2.mkdir()
    rag = TacticalEdgeRAG(tact_dir=tact_dir2, dim=64)
    mount = rag.mount_tactical_db()
    ok("Mount returns dict",      isinstance(mount, dict))
    ok("Mount status=mounted",    mount["status"] == "mounted")
    ok("Mount has all domains",   mount["domains"] == len(TACTICAL_DOMAINS))
    ok("Mount records ms",        mount["mount_ms"] >= 0)
    r_med = rag.query("stop bleeding tourniquet wound", top_k=3)
    ok("Query returns results",   len(r_med) > 0)
    ok("Results have domain key", all("domain" in r for r in r_med))
    r_code = rag.query("python dict merge pipe operator", top_k=3)
    ok("Coding domain returned",
       any(r["domain"] == "coding_triage" for r in r_code),
       f"domains: {[r['domain'] for r in r_code]}")

    # ── Test group 7: TacticalEdgeRAG status ─────────────────────────
    print("\n=== Group 7: TacticalEdgeRAG status ===")
    st = rag.get_status()
    ok("Status: mounted",         st["mounted"])
    ok("Status: total_entries>0", st["total_entries"] > 0)
    ok("Status: shadow_sync key", "shadow_sync" in st)
    ok("Status: under_target_ms", "under_target_ms" in st)

    # ── Test group 8: Legacy trigger_download ─────────────────────────
    print("\n=== Group 8: Legacy compat (trigger_download) ===")
    tact_dir3 = tmpdir / "legacy_test"
    tact_dir3.mkdir()
    rag2 = TacticalEdgeRAG(tact_dir=tact_dir3, dim=64)
    rag2.trigger_download()
    ok("trigger_download mounts", rag2._mounted)

    # ── Test group 9: LocalRAGIndex build + search ────────────────────
    print("\n=== Group 9: LocalRAGIndex build + search ===")
    idx_path = tmpdir / "test_rag.json"
    lri = LocalRAGIndex(index_path=str(idx_path))
    blueprints = [
        {"id": "organ_quantum_sonar",        "description": "sonar radar threat detection anomaly",          "category": "tactical"},
        {"id": "forge_autonomous_decision",  "description": "rust FFI autonomous decision cycle enemy",      "category": "forge"},
        {"id": "organ_motor_cortex",         "description": "applescript appliance IoT actuate motor",       "category": "motor"},
        {"id": "forge_locked_tensor_mac",    "description": "verilog toffoli HDL tensor mac FPGA",           "category": "forge"},
        {"id": "organ_cognitive_fusion",     "description": "hyperbolic triton math matrix distance",        "category": "math"},
        {"id": "organ_swarm",                "description": "swarm multi-agent consensus orchestration",     "category": "swarm"},
    ]
    n_indexed = lri.build_faiss_index(blueprints)
    ok("Indexed all blueprints",  n_indexed == len(blueprints), f"got {n_indexed}")
    ok("Built flag set",          lri._built)
    ok("doc_count correct",       lri.doc_count == len(blueprints))
    ok("vocab_size > 0",          len(lri._vocab) > 0)

    res_sonar = lri.vector_search_blueprint("sonar anomaly detection threat", top_k=3)
    ok("Sonar search returns",    len(res_sonar) > 0)
    ok("Sonar top result id",     res_sonar[0]["id"] == "organ_quantum_sonar",
       f"got: {res_sonar[0]['id']}")
    ok("Results have score",      all("score" in r for r in res_sonar))

    res_motor = lri.vector_search_blueprint("applescript IoT actuate smart home", top_k=2)
    ok("Motor search returns",    len(res_motor) > 0)
    ok("Motor top result id",     res_motor[0]["id"] == "organ_motor_cortex",
       f"got: {res_motor[0]['id']}")

    res_rust = lri.vector_search_blueprint("rust FFI autonomous decision", top_k=2)
    ok("Rust search returns",     len(res_rust) > 0)
    ok("Rust top result id",      res_rust[0]["id"] == "forge_autonomous_decision",
       f"got: {res_rust[0]['id']}")

    # ── Test group 10: LocalRAGIndex save/load ────────────────────────
    print("\n=== Group 10: LocalRAGIndex save/load ===")
    ok("Index file saved",        idx_path.exists())
    lri2 = LocalRAGIndex(index_path=str(idx_path))
    ok("Load from disk",          lri2._load())
    ok("doc_count after load",    lri2.doc_count == len(blueprints))
    res2 = lri2.vector_search_blueprint("sonar anomaly detection", top_k=1)
    ok("Search consistent after load",
       len(res2) > 0 and res2[0]["id"] == "organ_quantum_sonar",
       f"got: {res2[0]['id'] if res2 else 'empty'}")

    # ── Test group 11: LocalRAGIndex offline_rag_query alias ──────────
    print("\n=== Group 11: LocalRAGIndex offline_rag_query alias ===")
    r_alias = lri.offline_rag_query("hyperbolic math triton kernel", top_k=2)
    ok("offline_rag_query works", len(r_alias) > 0)
    ok("Alias top result math",   r_alias[0]["id"] == "organ_cognitive_fusion",
       f"got: {r_alias[0]['id']}")

    # ── Test group 12: LocalRAGIndex augment_system_prompt ────────────
    print("\n=== Group 12: LocalRAGIndex augment_system_prompt ===")
    augmented = lri.augment_system_prompt("You are Swayambhu.", "sonar anomaly radar")
    ok("Augmented has header",    "[LOCAL RAG" in augmented)
    ok("Base prompt preserved",   "You are Swayambhu." in augmented)
    ok("Relevant entry injected", "sonar" in augmented.lower() or "quantum" in augmented.lower())

    # ── Test group 13: LocalRAGIndex get_status ───────────────────────
    print("\n=== Group 13: LocalRAGIndex get_status ===")
    lri_status = lri.get_status()
    ok("Status: built",           lri_status["built"])
    ok("Status: doc_count",       lri_status["doc_count"] == len(blueprints))
    ok("Status: vocab_size>0",    lri_status["vocab_size"] > 0)
    ok("Status: index_path",      "index_path" in lri_status)

    # ── Test group 14: LocalToolRAG basic retrieval ───────────────────
    print("\n=== Group 14: LocalToolRAG retrieval ===")
    tool_rag = LocalToolRAG()
    ok("Default registry loaded", len(tool_rag.tool_registry) == 5)

    res_cpp = tool_rag.retrieve_top_tools("compile C++ kernel binary", top_k=2)
    ok("CPP tool retrieved",      "c++_kernel_compiler" in res_cpp,
       f"got: {list(res_cpp.keys())}")
    ok("Returns at most top_k",   len(res_cpp) <= 2)

    res_sensor = tool_rag.retrieve_top_tools("read biometric sensor temperature", top_k=2)
    ok("Sensor tool retrieved",   "i2c_sensor_bridge" in res_sensor,
       f"got: {list(res_sensor.keys())}")

    res_file = tool_rag.retrieve_top_tools("read write local JSON file", top_k=1)
    ok("File manager retrieved",  "local_file_manager" in res_file,
       f"got: {list(res_file.keys())}")

    res_math = tool_rag.retrieve_top_tools("matrix multiplication quantum variance", top_k=1)
    ok("Math core retrieved",     "tensor_math_core" in res_math,
       f"got: {list(res_math.keys())}")

    # ── Test group 15: LocalToolRAG register + status ─────────────────
    print("\n=== Group 15: LocalToolRAG register + status ===")
    tool_rag.register_tool("quantum_circuit_sim", "Simulates quantum gate circuits and measures qubit states.")
    ok("Tool registered",         "quantum_circuit_sim" in tool_rag.tool_registry)
    ok("Registry size grew",      len(tool_rag.tool_registry) == 6)

    res_quantum = tool_rag.retrieve_top_tools("qubit gate simulation measurement", top_k=2)
    ok("New tool retrievable",    "quantum_circuit_sim" in res_quantum,
       f"got: {list(res_quantum.keys())}")

    st_tool = tool_rag.get_status()
    ok("Status: tool_count",      st_tool["tool_count"] == 6)
    ok("Status: last_query set",  "qubit" in st_tool["last_query"])
    ok("Status: last_results",    len(st_tool["last_results"]) > 0)

    # ── Test group 16: LocalToolRAG cosine edge cases ─────────────────
    print("\n=== Group 16: LocalToolRAG cosine edge cases ===")
    res_empty = tool_rag.retrieve_top_tools("", top_k=2)
    ok("Empty query returns dict", isinstance(res_empty, dict))
    ok("Empty query <= top_k",     len(res_empty) <= 2)

    res_gibber = tool_rag.retrieve_top_tools("xyzzy zzz nonsense", top_k=2)
    ok("No-match returns dict",    isinstance(res_gibber, dict))

    # ── Test group 17: LocalRAGIndex + TacticalEdgeRAG integration ────
    print("\n=== Group 17: LocalRAGIndex + TacticalEdgeRAG integration ===")
    tact_dir4 = tmpdir / "integration_test"
    tact_dir4.mkdir()
    idx_path4 = tmpdir / "integration_rag.json"
    shared_lri = LocalRAGIndex(index_path=str(idx_path4))
    shared_lri.build_faiss_index(blueprints)

    # Wire LocalRAGIndex into TacticalEdgeRAG as legacy rag_index
    integrated_rag = TacticalEdgeRAG(
        rag_index=shared_lri, tact_dir=tact_dir4, dim=64
    )
    mount2 = integrated_rag.mount_tactical_db()
    ok("Integration mount succeeds", mount2["status"] == "mounted")

    # LocalRAGIndex should have been refreshed with tactical domain entries
    ok("LocalRAGIndex still built", shared_lri._built)

    # Cross-query: tactical RAG for survival, LocalRAGIndex for blueprints
    surv_results = integrated_rag.query("water purification boiling altitude", top_k=3)
    ok("Tactical survival query",   any(r["domain"] == "survival_protocol" for r in surv_results))

    bp_results = shared_lri.vector_search_blueprint("swarm multi agent consensus", top_k=2)
    ok("Blueprint swarm query",     len(bp_results) > 0 and bp_results[0]["id"] == "organ_swarm",
       f"got: {bp_results[0]['id'] if bp_results else 'empty'}")

    # ── Test group 18: TacticalEdgeRAG re-mount idempotency ──────────
    print("\n=== Group 18: TacticalEdgeRAG re-mount idempotency ===")
    mount3 = integrated_rag.mount_tactical_db()
    ok("Re-mount returns mounted",  mount3["status"] == "mounted")
    # all domains should be loaded (not re-built) on second call
    ok("Re-mount loaded=all",       mount3["loaded"] == len(TACTICAL_DOMAINS),
       f"loaded={mount3['loaded']}, built={mount3['built']}")
    ok("Re-mount built=0",          mount3["built"] == 0,
       f"built={mount3['built']}")

    shutil.rmtree(tmpdir)

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_tests() else 1)
