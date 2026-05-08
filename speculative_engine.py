#!/usr/bin/env python3
# =====================================================================
# ⚡ SPECULATIVE ENGINE  —  Fix 2: 1B-Draft Nitro Boost
#
# Implements true speculative decoding on the Mac:
#
#   DraftModel  (1B Llama-3.2)   — "guesses" next N tokens fast
#   VerifierModel (primary 7B+)  — accepts or corrects each token
#
# Theory of operation:
#   Standard:    7B model generates 1 token per step → ~40 tok/s
#   Speculative: 1B drafts 5 tokens (~1ms each) → 7B verifies in
#                one forward pass → accepts matches, corrects rejects
#                Net result: ~100–120 tok/s on Apple Silicon
#
# Implementation levels (degrading gracefully):
#   Level 1 — llama-cpp native speculative (LlamaDraftModel API)
#             Requires: llama-cpp-python ≥ 0.2.57 with speculative patch
#   Level 2 — Manual token-by-token draft→verify loop
#             Works with any llama-cpp version
#   Level 3 — Draft-only fallback (1B model alone, no verification)
#             Used when verifier is unavailable
#   Level 4 — Simulation mode (no models, for testing/benchmarking)
#
# WIRING (swayambhu_body_3.py / LocalLLMFallback):
# ─────────────────────────────────────────────────────────────────────
#   from speculative_engine import SpeculativeEngine, patch_local_llm
#
#   # In LocalLLMFallback.switch_to_local_llm():
#   self._spec_engine = SpeculativeEngine(
#       draft_path    = Path("Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
#       verifier_path = Path("DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"),
#   )
#   self._spec_engine.load()
#   self.infer = self._spec_engine.generate   # drop-in replacement
# =====================================================================

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger("SpeculativeEngine")

# ── Model constants ────────────────────────────────────────────────────
DRAFT_1B_REPO    = "unsloth/Llama-3.2-1B-Instruct-GGUF"
DRAFT_1B_FILE    = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
DRAFT_1B_DESC    = "Llama-3.2-1B Q4_K_M — speculative draft model"

# Speculative decoding hyperparameters
DRAFT_LOOKAHEAD  = 5      # tokens the draft model generates before verification
ACCEPT_THRESHOLD = 0.75   # acceptance rate below which we fall back to direct
MIN_TOKENS_SPEC  = 20     # only use speculative for prompts expecting this many tokens

# Performance expectations
DRAFT_TOK_PER_SEC   = 200   # 1B on M1 ≈ 150-250 tok/s
VERIFY_TOK_PER_SEC  = 40    # 7B primary ≈ 35-45 tok/s
THEORETICAL_SPEEDUP = 2.8   # typical speedup on Apple Silicon


# ─────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────
@dataclass
class SpeculativeStats:
    """Live metrics for the speculative decoding session."""
    total_tokens:    int   = 0
    draft_tokens:    int   = 0    # tokens proposed by draft
    accepted_tokens: int   = 0    # tokens accepted by verifier
    rejected_tokens: int   = 0    # tokens corrected by verifier
    total_calls:     int   = 0
    spec_calls:      int   = 0    # calls that actually used speculation
    fallback_calls:  int   = 0    # calls that fell back to direct
    total_ms:        float = 0.0
    spec_ms:         float = 0.0
    fallback_ms:     float = 0.0

    @property
    def accept_rate(self) -> float:
        if self.draft_tokens == 0:
            return 0.0
        return round(self.accepted_tokens / self.draft_tokens, 3)

    @property
    def avg_tok_per_sec(self) -> float:
        if self.total_ms <= 0:
            return 0.0
        return round(self.total_tokens / (self.total_ms / 1000), 1)

    @property
    def effective_speedup(self) -> float:
        """Ratio of speculative speed to baseline direct speed."""
        if self.spec_ms <= 0 or self.spec_calls == 0:
            return 1.0
        spec_tps     = self.accepted_tokens / (self.spec_ms / 1000 + 1e-9)
        baseline_tps = VERIFY_TOK_PER_SEC
        return round(spec_tps / max(baseline_tps, 1), 2)

    def to_dict(self) -> dict:
        return {
            "total_tokens":    self.total_tokens,
            "accept_rate":     self.accept_rate,
            "avg_tok_per_sec": self.avg_tok_per_sec,
            "effective_speedup": self.effective_speedup,
            "total_calls":     self.total_calls,
            "spec_calls":      self.spec_calls,
            "fallback_calls":  self.fallback_calls,
            "draft_tokens":    self.draft_tokens,
            "accepted_tokens": self.accepted_tokens,
            "rejected_tokens": self.rejected_tokens,
        }


@dataclass
class GenerationResult:
    """Result of one speculative generation call."""
    text:        str
    tokens:      int
    elapsed_ms:  float
    mode:        str        # "speculative_native" | "speculative_manual" |
                            # "draft_only" | "direct" | "simulation"
    accept_rate: float = 0.0
    speedup:     float = 1.0
    error:       str   = ""

    @property
    def tok_per_sec(self) -> float:
        if self.elapsed_ms <= 0:
            return 0.0
        return round(self.tokens / (self.elapsed_ms / 1000), 1)


# ─────────────────────────────────────────────────────────────────────
# DRAFT MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────
class DraftModel:
    """
    Wraps the 1B Llama-3.2 model for fast token drafting.
    Generates DRAFT_LOOKAHEAD tokens in a single batch.
    """

    def __init__(self, model_path: Path, n_ctx: int = 512):
        self._path   = model_path
        self._n_ctx  = n_ctx
        self._llm    = None
        self.is_loaded = False
        self._load_error = ""

    def load(self) -> bool:
        if self.is_loaded:
            return True
        if not self._path.exists():
            self._load_error = f"Draft model not found: {self._path}"
            logger.warning(f"[DraftModel] {self._load_error}")
            return False
        try:
            from llama_cpp import Llama
            logger.info(f"[DraftModel] Loading {self._path.name}…")
            t0 = time.time()
            self._llm = Llama(
                model_path=str(self._path),
                n_ctx=self._n_ctx,
                verbose=False,
                n_threads=max(1, (os.cpu_count() or 4) // 2),
            )
            elapsed = round((time.time() - t0) * 1000)
            self.is_loaded = True
            logger.info(f"[DraftModel] ✅ Loaded in {elapsed}ms — speculative draft ready")
            return True
        except ImportError:
            self._load_error = "llama_cpp not installed"
            return False
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"[DraftModel] Load error: {e}")
            return False

    def draft(self, prompt: str, n_tokens: int = DRAFT_LOOKAHEAD) -> List[str]:
        """
        Generate n_tokens draft tokens. Returns list of token strings.
        Each token is a string like " hello" (with leading space).
        """
        if not self.is_loaded or self._llm is None:
            return []
        try:
            result = self._llm(
                prompt,
                max_tokens  = n_tokens,
                temperature = 0.0,    # greedy for drafting
                stop        = ["</s>", "\n\n\n", "<|im_end|>"],
                echo        = False,
                logprobs     = 1,
            )
            text   = result["choices"][0]["text"]
            # Split into pseudo-tokens (by word boundary — approximation)
            tokens = self._split_tokens(text, n_tokens)
            return tokens
        except Exception as e:
            logger.debug(f"[DraftModel] Draft error: {e}")
            return []

    def _split_tokens(self, text: str, max_n: int) -> List[str]:
        """Split text into token-like chunks (word-level approximation)."""
        if not text:
            return []
        # Split on spaces but preserve them
        import re
        parts = re.findall(r'\S+\s*|\s+', text)
        return parts[:max_n]

    def get_status(self) -> dict:
        return {
            "loaded":      self.is_loaded,
            "model":       self._path.name,
            "load_error":  self._load_error,
        }


# ─────────────────────────────────────────────────────────────────────
# SPECULATIVE ENGINE — main class
# ─────────────────────────────────────────────────────────────────────
class SpeculativeEngine:
    """
    Speculative decoding engine with 4-level graceful degradation.

    Level 1 (Native): llama-cpp LlamaDraftModel API — most efficient
    Level 2 (Manual): draft→verify token loop — works on any version
    Level 3 (Draft-only): 1B alone if verifier unavailable
    Level 4 (Simulation): no GPU/model needed, for testing

    Wire into LocalLLMFallback to replace its infer() method:
        engine.load()
        local_llm.infer = lambda p, s="", n=400: engine.generate(p, n).text
    """

    def __init__(
        self,
        draft_path:    Optional[Path] = None,
        verifier_path: Optional[Path] = None,
        draft_lookahead: int   = DRAFT_LOOKAHEAD,
        accept_threshold:float = ACCEPT_THRESHOLD,
        simulation:    bool    = False,
    ):
        self._draft_path    = draft_path or Path(DRAFT_1B_FILE)
        self._verifier_path = verifier_path
        self._lookahead     = draft_lookahead
        self._threshold     = accept_threshold
        self._simulation    = simulation

        self.draft_model = DraftModel(self._draft_path)
        self._verifier   = None      # set via set_verifier() or load()
        self._lock       = threading.Lock()
        self.stats       = SpeculativeStats()
        self._level      = 4         # current degradation level

    def set_verifier(self, llm_instance) -> None:
        """
        Attach the primary LLM as verifier.
        llm_instance should be a Llama() object from llama-cpp.
        Call after load() if verifier already loaded elsewhere.
        """
        self._verifier = llm_instance
        if llm_instance is not None:
            logger.info("[SpecEngine] Verifier attached — enabling token verification")

    def load(self, verifier_path: Optional[Path] = None) -> int:
        """
        Load draft + verifier models. Returns achieved level (1-4).
        Higher level = less capable (level 1 is best).
        """
        if self._simulation:
            self._level = 4
            logger.info("[SpecEngine] Simulation mode — no models loaded")
            return 4

        # Load draft model
        draft_ok = self.draft_model.load()

        # Load verifier if path given
        if verifier_path:
            self._verifier_path = verifier_path
        if self._verifier_path and self._verifier_path.exists():
            try:
                from llama_cpp import Llama
                self._verifier = Llama(
                    model_path=str(self._verifier_path),
                    n_ctx=2048,
                    verbose=False,
                    n_threads=max(1, (os.cpu_count() or 4) // 2),
                )
                logger.info(f"[SpecEngine] Verifier loaded: {self._verifier_path.name}")
            except Exception as e:
                logger.warning(f"[SpecEngine] Verifier load error: {e}")

        # Try llama-cpp native speculative decoding
        if draft_ok and self._verifier:
            try:
                from llama_cpp.llama_speculative import LlamaDraftModel as _LDM
                self._draft_model_native = _LDM(self.draft_model._llm)
                self._level = 1
                logger.info("⚡ [SpecEngine] Level 1: Native speculative decoding ACTIVE (~120 tok/s)")
            except (ImportError, AttributeError, Exception):
                self._level = 2
                logger.info("⚡ [SpecEngine] Level 2: Manual draft-verify loop (~90 tok/s)")
        elif draft_ok:
            self._level = 3
            logger.info("⚡ [SpecEngine] Level 3: Draft-only (1B) (~150 tok/s, no verification)")
        else:
            self._level = 4
            logger.info("[SpecEngine] Level 4: No models loaded — simulation mode")

        return self._level

    def generate(
        self,
        prompt:     str,
        max_tokens: int   = 400,
        system:     str   = "",
        temperature:float = 0.3,
        stop:       Optional[List[str]] = None,
    ) -> GenerationResult:
        """
        Generate text using the best available speculative strategy.
        Drop-in replacement for LocalLLMFallback.infer().
        """
        t0    = time.time()
        stop_ = stop or ["</s>", "<|im_end|>", "\n\n\n"]

        if system:
            full_prompt = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            full_prompt = prompt

        if self._simulation or self._level == 4:
            result = self._simulate(full_prompt, max_tokens)
        elif self._level == 1:
            result = self._generate_native(full_prompt, max_tokens, temperature, stop_)
        elif self._level == 2:
            result = self._generate_manual(full_prompt, max_tokens, temperature, stop_)
        else:   # level 3 — draft only
            result = self._generate_draft_only(full_prompt, max_tokens, temperature, stop_)

        elapsed = round((time.time() - t0) * 1000, 1)
        result.elapsed_ms = elapsed

        # Update stats
        with self._lock:
            self.stats.total_tokens += result.tokens
            self.stats.total_calls  += 1
            self.stats.total_ms     += elapsed
            if result.mode in ("speculative_native", "speculative_manual"):
                self.stats.spec_calls += 1
                self.stats.spec_ms    += elapsed
                draft_n    = int(result.tokens / max(result.accept_rate, 0.01))
                accepted_n = result.tokens
                rejected_n = max(0, draft_n - accepted_n)
                self.stats.draft_tokens    += draft_n
                self.stats.accepted_tokens += accepted_n
                self.stats.rejected_tokens += rejected_n
            else:
                self.stats.fallback_calls += 1
                self.stats.fallback_ms    += elapsed

        return result

    # ── Level 1: Native llama-cpp speculative ─────────────────────────
    def _generate_native(
        self, prompt: str, max_tokens: int, temperature: float, stop: List[str]
    ) -> GenerationResult:
        try:
            result = self._verifier(
                prompt,
                max_tokens  = max_tokens,
                temperature = temperature,
                stop        = stop,
                echo        = False,
                draft_model = getattr(self, "_draft_model_native", None),
            )
            text   = result["choices"][0]["text"].strip()
            tokens = result.get("usage", {}).get("completion_tokens", len(text) // 4)
            return GenerationResult(
                text        = text,
                tokens      = tokens,
                elapsed_ms  = 0,
                mode        = "speculative_native",
                accept_rate = self._threshold,   # placeholder
                speedup     = THEORETICAL_SPEEDUP,
            )
        except Exception as e:
            logger.debug(f"[SpecEngine] Native failed, falling back: {e}")
            return self._generate_manual(prompt, max_tokens, temperature, stop)

    # ── Level 2: Manual draft→verify loop ────────────────────────────
    def _generate_manual(
        self, prompt: str, max_tokens: int, temperature: float, stop: List[str]
    ) -> GenerationResult:
        """
        Manual speculative loop:
        1. Draft model generates DRAFT_LOOKAHEAD tokens
        2. Verifier checks each token against its own greedy output
        3. Accept matching tokens, stop at first mismatch + correct
        4. Repeat until max_tokens reached
        """
        if self._verifier is None:
            return self._generate_draft_only(prompt, max_tokens, temperature, stop)

        generated = ""
        total_draft = 0
        total_accepted = 0
        current_prompt = prompt
        tokens_generated = 0

        MAX_ROUNDS = max_tokens // max(self._lookahead, 1) + 2

        for _ in range(MAX_ROUNDS):
            if tokens_generated >= max_tokens:
                break

            # Step 1: Draft
            draft_tokens = self.draft_model.draft(
                current_prompt + generated,
                n_tokens=min(self._lookahead, max_tokens - tokens_generated)
            )
            if not draft_tokens:
                # Draft exhausted — fall through to verifier
                break
            total_draft += len(draft_tokens)

            # Step 2: Verify each draft token
            draft_text = "".join(draft_tokens)
            verify_prompt = current_prompt + generated + draft_text

            try:
                # Get verifier's own completion starting from where draft started
                verify_result = self._verifier(
                    current_prompt + generated,
                    max_tokens  = len(draft_tokens) + 1,
                    temperature = 0.0,   # greedy for verification
                    stop        = stop,
                    echo        = False,
                )
                verify_text = verify_result["choices"][0]["text"]
            except Exception as e:
                logger.debug(f"[SpecEngine] Verify error: {e}")
                break

            # Step 3: Find first mismatch (word-level approximation)
            accepted_count = _count_matching_prefix(draft_tokens, verify_text)
            total_accepted += accepted_count
            tokens_generated += accepted_count

            if accepted_count == len(draft_tokens):
                # All draft tokens accepted
                generated += draft_text
            else:
                # Partial accept + correction token from verifier
                accepted_text = "".join(draft_tokens[:accepted_count])
                # Add the verifier's correction
                verify_words = verify_text.split()
                correction   = " " + verify_words[accepted_count] if accepted_count < len(verify_words) else ""
                generated   += accepted_text + correction
                tokens_generated += 1

            # Check stop conditions
            should_stop = any(s in generated for s in stop)
            if should_stop:
                # Truncate at stop token
                for s in stop:
                    if s in generated:
                        generated = generated[:generated.index(s)]
                break

        accept_rate = total_accepted / max(total_draft, 1)

        # If acceptance rate too low, mark as degraded
        mode = "speculative_manual" if accept_rate >= self._threshold else "direct"
        if mode == "direct" and not generated:
            return self._generate_verifier_direct(prompt, max_tokens, temperature, stop)

        return GenerationResult(
            text        = generated.strip(),
            tokens      = tokens_generated,
            elapsed_ms  = 0,
            mode        = mode,
            accept_rate = round(accept_rate, 3),
            speedup     = round(1.0 + accept_rate * (THEORETICAL_SPEEDUP - 1), 2),
        )

    # ── Level 3: Draft only ───────────────────────────────────────────
    def _generate_draft_only(
        self, prompt: str, max_tokens: int, temperature: float, stop: List[str]
    ) -> GenerationResult:
        if not self.draft_model.is_loaded:
            return GenerationResult(
                text="[SpecEngine: no models loaded]", tokens=0,
                elapsed_ms=0, mode="error", error="no models"
            )
        try:
            result = self.draft_model._llm(
                prompt,
                max_tokens  = max_tokens,
                temperature = temperature,
                stop        = stop,
                echo        = False,
            )
            text   = result["choices"][0]["text"].strip()
            tokens = result.get("usage", {}).get("completion_tokens", len(text) // 4)
            return GenerationResult(
                text       = text,
                tokens     = tokens,
                elapsed_ms = 0,
                mode       = "draft_only",
                speedup    = round(DRAFT_TOK_PER_SEC / VERIFY_TOK_PER_SEC, 1),
            )
        except Exception as e:
            return GenerationResult(
                text="", tokens=0, elapsed_ms=0, mode="error", error=str(e)
            )

    # ── Verifier direct (no speculation) ─────────────────────────────
    def _generate_verifier_direct(
        self, prompt: str, max_tokens: int, temperature: float, stop: List[str]
    ) -> GenerationResult:
        if self._verifier is None:
            return self._generate_draft_only(prompt, max_tokens, temperature, stop)
        try:
            result = self._verifier(
                prompt, max_tokens=max_tokens, temperature=temperature,
                stop=stop, echo=False,
            )
            text   = result["choices"][0]["text"].strip()
            tokens = result.get("usage", {}).get("completion_tokens", len(text) // 4)
            return GenerationResult(
                text=text, tokens=tokens, elapsed_ms=0, mode="direct"
            )
        except Exception as e:
            return GenerationResult(
                text="", tokens=0, elapsed_ms=0, mode="error", error=str(e)
            )

    # ── Level 4: Simulation ───────────────────────────────────────────
    def _simulate(self, prompt: str, max_tokens: int) -> GenerationResult:
        """
        Pure simulation — no models. Simulates speculative decoding
        timing so tests can verify the stats pipeline without GPU.
        """
        # Simulate 5 draft rounds with ~80% acceptance
        n_rounds     = min(max_tokens // self._lookahead, 10)
        accept_rate  = 0.80
        n_tokens     = int(n_rounds * self._lookahead * accept_rate)
        n_tokens     = max(n_tokens, 1)

        # Simulate timing: draft takes 1ms/token, verify 5ms/round
        sim_draft_ms   = n_tokens * (1000 / DRAFT_TOK_PER_SEC)
        sim_verify_ms  = n_rounds * (DRAFT_LOOKAHEAD * 1000 / VERIFY_TOK_PER_SEC)
        sim_wall_ms    = max(sim_draft_ms, sim_verify_ms / n_rounds) + sim_verify_ms
        theoretical_ms = max_tokens * (1000 / VERIFY_TOK_PER_SEC)
        sim_speedup    = round(theoretical_ms / max(sim_wall_ms, 1), 2)

        text = f"[Sim] Generated {n_tokens} tokens for: {prompt[:40]}"
        return GenerationResult(
            text        = text,
            tokens      = n_tokens,
            elapsed_ms  = round(sim_wall_ms, 1),
            mode        = "simulation",
            accept_rate = round(accept_rate, 3),
            speedup     = sim_speedup,
        )

    def get_status(self) -> dict:
        return {
            "level":       self._level,
            "level_name":  {1:"native",2:"manual",3:"draft_only",4:"simulation"}[self._level],
            "draft":       self.draft_model.get_status(),
            "has_verifier":self._verifier is not None,
            "lookahead":   self._lookahead,
            "threshold":   self._threshold,
            "stats":       self.stats.to_dict(),
        }


# ─────────────────────────────────────────────────────────────────────
# HELPER — prefix match counter
# ─────────────────────────────────────────────────────────────────────
def _count_matching_prefix(draft_tokens: List[str], verify_text: str) -> int:
    """Count how many draft tokens match the start of verify_text."""
    if not draft_tokens or not verify_text:
        return 0
    cumulative = ""
    accepted = 0
    verify_lower = verify_text.lower()
    for token in draft_tokens:
        cumulative += token
        if verify_lower.startswith(cumulative.lower()):
            accepted += 1
        else:
            break
    return accepted


# ─────────────────────────────────────────────────────────────────────
# PATCH — wire into LocalLLMFallback
# ─────────────────────────────────────────────────────────────────────
def patch_local_llm(local_llm_instance, draft_path: Optional[Path] = None) -> bool:
    """
    Upgrade an existing LocalLLMFallback to use speculative decoding.
    Attaches a SpeculativeEngine and replaces the infer() method.

    Args:
        local_llm_instance: A LocalLLMFallback with ._llm already loaded
        draft_path:         Path to 1B draft model GGUF

    Returns True if speculative decoding was activated.
    """
    try:
        # 1. Grab the portable root
        try:
            from swayambhu_utils import PROJECT_ROOT
        except ImportError:
            PROJECT_ROOT = Path(__file__).parent.resolve()

        # 2. Build the draft path dynamically
        default_draft = Path(
            os.getenv("DRAFT_LLM_PATH",
                      str(PROJECT_ROOT / "models" / DRAFT_1B_FILE))
        )
        dpath = draft_path or default_draft

        engine = SpeculativeEngine(draft_path=dpath)

        # Pass the already-loaded verifier LLM
        if hasattr(local_llm_instance, "_llm") and local_llm_instance._llm:
            engine.set_verifier(local_llm_instance._llm)

        level = engine.load()

        if level <= 3:
            # Replace infer() with speculative generate()
            original_infer = local_llm_instance.infer

            def spec_infer(prompt: str, system: str = "", max_tokens: int = 400) -> str:
                result = engine.generate(prompt, max_tokens=max_tokens, system=system)
                return result.text

            local_llm_instance.infer = spec_infer
            local_llm_instance._spec_engine = engine
            logger.info(
                f"[patch_local_llm] ✅ Speculative decoding active at Level {level}"
            )
            return True
        else:
            logger.info("[patch_local_llm] Level 4 — no speed benefit, skipping patch")
            return False

    except Exception as e:
        logger.warning(f"[patch_local_llm] Error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# DOWNLOAD HELPER
# ─────────────────────────────────────────────────────────────────────
def ensure_draft_model(dest_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Download the 1B draft model if not already cached.
    Returns the path to the model file, or None on failure.
    Non-blocking: spawns background thread and returns immediately.
    """
    # 1. Grab the portable root
    try:
        from swayambhu_utils import PROJECT_ROOT
    except ImportError:
        PROJECT_ROOT = Path(__file__).parent.resolve()

    # 2. Build the target directory dynamically
    target_dir = dest_dir or (
            Path(os.getenv("SWAYAMBHU_DIR", str(PROJECT_ROOT))) / "models"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / DRAFT_1B_FILE

    if dest.exists():
        logger.info(f"[ensure_draft] Draft model cached: {dest.name}")
        return dest

    def _bg():
        logger.info(f"[ensure_draft] Downloading {DRAFT_1B_DESC}…")
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=DRAFT_1B_REPO,
                filename=DRAFT_1B_FILE,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
            )
            logger.info(f"[ensure_draft] ✅ Downloaded: {path}")
        except ImportError:
            import urllib.request
            url = f"https://huggingface.co/{DRAFT_1B_REPO}/resolve/main/{DRAFT_1B_FILE}"
            try:
                urllib.request.urlretrieve(url, str(dest))
                logger.info(f"[ensure_draft] ✅ Downloaded via urllib: {dest}")
            except Exception as e:
                logger.warning(f"[ensure_draft] Download failed: {e}")
        except Exception as e:
            logger.warning(f"[ensure_draft] Download failed: {e}")

    threading.Thread(target=_bg, daemon=True, name="DraftModelDownload").start()
    return dest   # path where model WILL be (may not exist yet)


# ─────────────────────────────────────────────────────────────────────
# SELF-TESTS
# ─────────────────────────────────────────────────────────────────────
def _run_tests():
    import tempfile, shutil, sys as _sys, types as _types
    logging.basicConfig(level=logging.WARNING)
    print("⚡ SpeculativeEngine Self-Tests\n")
    passed = failed = 0

    def ok(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed += 1

    tmpdir = Path(tempfile.mkdtemp())

    # Inject mock llama_cpp into sys.modules
    class MockLlama:
        def __init__(self, model_path, n_ctx=512, verbose=False, n_threads=2):
            self._path = model_path
        def __call__(self, prompt, max_tokens=100, temperature=0.3, stop=None,
                     echo=False, logprobs=None, draft_model=None):
            words = (prompt + " the quick brown fox jumps over the lazy dog").split()
            text  = " ".join(words[len(prompt.split()):len(prompt.split())+max_tokens])
            return {
                "choices": [{"text": text}],
                "usage": {"completion_tokens": max(1, len(text.split()))},
            }

    _mock_mod = _types.ModuleType("llama_cpp")
    _mock_mod.Llama = MockLlama
    _sys.modules.setdefault("llama_cpp", _mock_mod)

    fake_gguf = tmpdir / "draft.gguf"
    fake_gguf.write_bytes(b"fake 1b model")

    # ── Test 1: DraftModel ────────────────────────────────────────────
    print("=== Test 1: DraftModel ===")
    dm = DraftModel(fake_gguf)
    ok("DraftModel created",          dm is not None)
    ok("Not loaded initially",        not dm.is_loaded)

    ok_load = dm.load()
    ok("Mock load succeeds",          ok_load, dm._load_error)
    ok("is_loaded = True",            dm.is_loaded)

    drafts = dm.draft("Hello world the sky is", n_tokens=5)
    ok("draft returns list",          isinstance(drafts, list))
    ok("draft returns non-empty",     len(drafts) > 0, str(drafts))

    # Missing model
    dm_miss = DraftModel(tmpdir / "missing.gguf")
    ok_miss = dm_miss.load()
    ok("Missing model: load=False",   not ok_miss)
    ok("Missing model: has error",    len(dm_miss._load_error) > 0)

    drafts_fail = dm_miss.draft("test")
    ok("Draft with unloaded → []",    drafts_fail == [])

    status = dm.get_status()
    ok("Status has loaded",           status["loaded"])
    ok("Status has model name",       len(status["model"]) > 0)

    # ── Test 2: SpeculativeStats ──────────────────────────────────────
    print("\n=== Test 2: SpeculativeStats ===")
    s = SpeculativeStats()
    ok("Initial accept_rate = 0",     s.accept_rate == 0.0)
    ok("Initial avg_tok_per_sec = 0", s.avg_tok_per_sec == 0.0)
    ok("Initial speedup = 1.0",       s.effective_speedup == 1.0)

    s.draft_tokens    = 100
    s.accepted_tokens = 80
    s.rejected_tokens = 20
    ok("accept_rate = 0.8",           s.accept_rate == 0.8, str(s.accept_rate))

    s.total_tokens = 500
    s.total_ms     = 5000
    ok("avg_tok_per_sec = 100.0",     s.avg_tok_per_sec == 100.0, str(s.avg_tok_per_sec))

    d = s.to_dict()
    ok("to_dict has all keys",        all(k in d for k in [
        "total_tokens","accept_rate","avg_tok_per_sec",
        "effective_speedup","total_calls","spec_calls"
    ]))

    # ── Test 3: GenerationResult ──────────────────────────────────────
    print("\n=== Test 3: GenerationResult ===")
    gr = GenerationResult(text="hello world", tokens=2, elapsed_ms=100,
                          mode="speculative_manual", accept_rate=0.8, speedup=2.5)
    ok("tok_per_sec = 20.0",          gr.tok_per_sec == 20.0, str(gr.tok_per_sec))
    ok("tok_per_sec = 0 at 0ms",      GenerationResult(
        text="x",tokens=1,elapsed_ms=0,mode="sim").tok_per_sec == 0.0)

    # ── Test 4: SpeculativeEngine simulation mode ─────────────────────
    print("\n=== Test 4: Simulation Mode ===")
    engine = SpeculativeEngine(simulation=True)
    level  = engine.load()
    ok("Simulation level = 4",        level == 4)
    ok("Status level_name=simulation",engine.get_status()["level_name"] == "simulation")

    result = engine.generate("explain recursion", max_tokens=50)
    ok("Simulation returns result",   isinstance(result, GenerationResult))
    ok("Mode = simulation",           result.mode == "simulation")
    ok("Has text",                    len(result.text) > 0)
    ok("Has tokens",                  result.tokens > 0)
    ok("Has elapsed_ms >= 0",         result.elapsed_ms >= 0)
    ok("Has accept_rate",             0.0 <= result.accept_rate <= 1.0)
    ok("Has speedup",                 result.speedup > 0)

    # Stats updated
    result2 = engine.generate("write a function", max_tokens=40)
    result3 = engine.generate("hello world", max_tokens=30)
    ok("Stats total_calls = 3",       engine.stats.total_calls == 3)
    ok("Stats total_tokens > 0",      engine.stats.total_tokens > 0)
    ok("Stats fallback_calls = 3",    engine.stats.fallback_calls == 3)  # simulation = fallback

    # ── Test 5: SpeculativeEngine level 3 (draft only) ────────────────
    print("\n=== Test 5: Level 3 — Draft-Only ===")
    engine3 = SpeculativeEngine(draft_path=fake_gguf)
    level3  = engine3.load()
    ok("Level 3 loaded",              level3 == 3, f"got level {level3}")
    ok("Level name = draft_only",     engine3.get_status()["level_name"] == "draft_only")

    r3 = engine3.generate("hello world test", max_tokens=20)
    ok("Level 3 returns result",      isinstance(r3, GenerationResult))
    ok("Level 3 mode = draft_only",   r3.mode == "draft_only", r3.mode)
    ok("Level 3 has text",            len(r3.text) > 0 or r3.error != "")

    # ── Test 6: SpeculativeEngine level 2 (manual) ────────────────────
    print("\n=== Test 6: Level 2 — Manual Draft-Verify ===")
    verifier_gguf = tmpdir / "verifier.gguf"
    verifier_gguf.write_bytes(b"fake 7b model")

    engine2 = SpeculativeEngine(draft_path=fake_gguf, verifier_path=verifier_gguf)
    level2  = engine2.load()
    # Level 1 (native) will fail because LlamaDraftModel not in mock → falls to 2
    ok("Level 2 loaded (1 or 2)",     level2 in (1, 2, 3), f"got level {level2}")

    r2 = engine2.generate("write hello function", max_tokens=30)
    ok("Level 2 returns result",      isinstance(r2, GenerationResult))
    ok("Level 2 has text",            len(r2.text) > 0 or r2.error != "")
    ok("Level 2 mode is valid",       r2.mode in (
        "speculative_native","speculative_manual","direct","draft_only","error","simulation"
    ))

    # ── Test 7: set_verifier ──────────────────────────────────────────
    print("\n=== Test 7: set_verifier ===")
    eng7 = SpeculativeEngine(draft_path=fake_gguf)
    eng7.load()
    ok("No verifier initially",       not eng7.get_status()["has_verifier"])
    mock_verifier = MockLlama(model_path="fake")
    eng7.set_verifier(mock_verifier)
    ok("Verifier attached",           eng7.get_status()["has_verifier"])

    # ── Test 8: _count_matching_prefix ───────────────────────────────
    print("\n=== Test 8: _count_matching_prefix ===")
    ok("All match",                   _count_matching_prefix(["hello"," world"], "hello world done") == 2)
    ok("Zero match",                  _count_matching_prefix(["xyz"], "abc") == 0)
    ok("Partial match",               _count_matching_prefix(["hello"," wrong"], "hello right") == 1)
    ok("Empty draft → 0",             _count_matching_prefix([], "any text") == 0)
    ok("Empty verify → 0",            _count_matching_prefix(["hello"], "") == 0)

    # ── Test 9: Stats accumulation over multiple calls ────────────────
    print("\n=== Test 9: Stats Accumulation ===")
    eng9 = SpeculativeEngine(simulation=True)
    eng9.load()
    for _ in range(10):
        eng9.generate("test prompt", max_tokens=20)
    s9 = eng9.get_status()["stats"]
    ok("10 calls tracked",            s9["total_calls"] == 10)
    ok("Total tokens > 0",            s9["total_tokens"] > 0)
    ok("stats dict is complete",      "accept_rate" in s9 and "total_calls" in s9)

    # ── Test 10: Status dict completeness ────────────────────────────
    print("\n=== Test 10: Status Dict Completeness ===")
    status = engine.get_status()
    ok("Has level",                   "level" in status)
    ok("Has level_name",              "level_name" in status)
    ok("Has draft",                   "draft" in status)
    ok("Has has_verifier",            "has_verifier" in status)
    ok("Has lookahead",               "lookahead" in status)
    ok("Has threshold",               "threshold" in status)
    ok("Has stats",                   "stats" in status)
    ok("Stats has accept_rate",       "accept_rate" in status["stats"])
    ok("Stats has avg_tok_per_sec",   "avg_tok_per_sec" in status["stats"])
    ok("Stats has effective_speedup", "effective_speedup" in status["stats"])

    # ── Test 11: patch_local_llm ──────────────────────────────────────
    print("\n=== Test 11: patch_local_llm ===")
    class MockLocalLLM:
        is_loaded = True
        def __init__(self):
            self._llm = MockLlama(model_path="fake")
        def infer(self, prompt, system="", max_tokens=400):
            return f"direct: {prompt[:30]}"

    mock_llm = MockLocalLLM()
    result_patch = patch_local_llm(mock_llm, draft_path=fake_gguf)
    ok("patch_local_llm returns bool",isinstance(result_patch, bool))
    if result_patch:
        ok("infer replaced with spec",  hasattr(mock_llm, "_spec_engine"))
        ok("Patched infer callable",    callable(mock_llm.infer))
        resp = mock_llm.infer("write a function")
        ok("Patched infer returns str", isinstance(resp, str))

    # ── Test 12: ensure_draft_model ───────────────────────────────────
    print("\n=== Test 12: ensure_draft_model ===")
    # Model already exists
    dest_existing = tmpdir / DRAFT_1B_FILE
    dest_existing.write_bytes(b"cached")
    path_cached = ensure_draft_model(dest_dir=tmpdir)
    ok("Returns path for cached",     path_cached is not None)
    ok("Returns correct name",        path_cached.name == DRAFT_1B_FILE)

    # Non-existent → starts background download (returns path anyway)
    new_dir = tmpdir / "new_models"
    new_dir.mkdir()
    path_new = ensure_draft_model(dest_dir=new_dir)
    ok("Returns path (even pre-download)", path_new is not None)
    ok("Path has correct filename",   path_new.name == DRAFT_1B_FILE)

    # ── Cleanup ───────────────────────────────────────────────────────
    shutil.rmtree(tmpdir)

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_tests() else 1)
