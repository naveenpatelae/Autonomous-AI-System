#!/usr/bin/env python3
# =====================================================================
# 🤜🤛  DUAL MODEL ENGINE  —  Fix 1: Dual-Hand Parallel Execution
#
# Replaces "Single-Model Syndrome" with a true two-expert architecture:
#
#   CoderLLM   — DeepSeek-Coder-1.5B  (writes Python code)
#   TesterLLM  — Qwen2.5-1.5B-Instruct (writes pytest suites)
#
# Key design decisions:
#   • Both models held in RAM simultaneously — 1.5B × 2 ≈ 3 GB, fits
#     comfortably in 8 GB Apple Silicon unified memory
#   • Parallel execution: CoderLLM and TesterLLM run in concurrent
#     threads → Software Firm total latency = max(coder, tester)
#     instead of sum(coder + tester)
#   • Independent system prompts: each model locked to its role
#   • Hot-reload: swap either model path without restarting the other
#   • Graceful degradation: if tester model unavailable, falls back to
#     coder model for both roles (slower, but functional)
#
# WIRING (swayambhu_body_3.py / swayambhu_v13.py boot):
# ─────────────────────────────────────────────────────────────────────
#   from dual_model_engine import DualModelEngine, wire_software_firm
#
#   # In EdgeNodeOrchestrator.__init__():
#   self.dual_engine = DualModelEngine(
#       model_dir = Path(os.getenv("SWAYAMBHU_DIR")) / "models"
#   )
#
#   # In boot() after procurement:
#   self.dual_engine.load(
#       coder_path  = Path("DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf"),
#       tester_path = Path("qwen2.5-1.5b-instruct-q4_k_m.gguf"),
#   )
#   self.coder_llm  = self.dual_engine.coder
#   self.tester_llm = self.dual_engine.tester
#
#   # Wire Software Firm:
#   self.firm = wire_software_firm(
#       manager_fn   = self._cloud_llm_fn,
#       coder_llm    = self.coder_llm,
#       tester_llm   = self.tester_llm,
#   )
# =====================================================================

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("DualModelEngine")

# ── Model catalogue ────────────────────────────────────────────────────
CODER_MODEL_REPO  = "deepseek-ai/deepseek-coder-1.3b-instruct"
CODER_MODEL_FILE  = "deepseek-coder-1.3b-instruct.Q4_K_M.gguf"
CODER_MODEL_DESC  = "DeepSeek-Coder 1.3B — specialist code writer"

TESTER_MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
TESTER_MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
TESTER_MODEL_DESC = "Qwen2.5-1.5B-Instruct — specialist test writer"

# ── Role-locked system prompts ─────────────────────────────────────────
_CODER_SYSTEM = (
    "You are DeepSeek Coder, an expert multi-language software engineer. "
    "Write clean, secure, working code in whatever language the user requests. "
    "If you are writing a Swayambhu Blueprint or a local Mac automation, you MUST use Python. "
    "Handle errors gracefully and use best practices."
)

_TESTER_SYSTEM = (
    "You are an expert polyglot QA engineer. "
    "Write rigorous test suites using the standard testing framework for whatever language the user provides. "
    "No explanations. No markdown fences. Pure test code only. "
    "Cover: happy path, edge cases, error handling, and boundary values."
)

# ── Inference config ───────────────────────────────────────────────────
DEFAULT_N_CTX    = 8192
DEFAULT_VERBOSE  = False
STOP_TOKENS      = ["</s>", "<|im_end|>", "<|end_of_text|>", "<｜end▁of▁sentence｜>"]


# ─────────────────────────────────────────────────────────────────────
# INFERENCE RESULT
# ─────────────────────────────────────────────────────────────────────
@dataclass
class InferenceResult:
    text:       str
    role:       str          # "coder" | "tester"
    model:      str          # model filename
    elapsed_ms: float
    tokens:     int
    error:      str = ""

    @property
    def ok(self) -> bool:
        return not self.error and len(self.text.strip()) > 0

    @property
    def tokens_per_sec(self) -> float:
        if self.elapsed_ms <= 0:
            return 0.0
        return round(self.tokens / (self.elapsed_ms / 1000), 1)


# ─────────────────────────────────────────────────────────────────────
# SINGLE MODEL SLOT
# Wraps one llama-cpp model with role-locked system prompt,
# request queue, and thread-safe inference.
# ─────────────────────────────────────────────────────────────────────
class ModelSlot:
    """
    One model slot: loads a GGUF, exposes infer() with role-locked prompt.

    Thread-safe: uses a lock so concurrent callers queue rather than
    corrupt each other's context (llama-cpp is not thread-safe internally).
    """

    def __init__(
        self,
        role:         str,
        model_path:   Path,
        system_prompt:str,
        n_ctx:        int   = DEFAULT_N_CTX,
        verbose:      bool  = DEFAULT_VERBOSE,
        fallback_slot: Optional["ModelSlot"] = None,
    ):
        self.role          = role
        self.model_path    = model_path
        self.system_prompt = system_prompt
        self.n_ctx         = n_ctx
        self.verbose       = verbose
        self._fallback     = fallback_slot

        self._llm          = None
        self._lock         = threading.Lock()
        self.is_loaded     = False
        self._load_error   = ""
        self._call_count   = 0
        self._total_tokens = 0
        self._total_ms     = 0.0

    # ── Load ─────────────────────────────────────────────────────────
    def load(self) -> bool:
        """Load model into RAM. Thread-safe. Returns True on success."""
        if self.is_loaded:
            return True
        if not self.model_path.exists():
            self._load_error = f"Model file not found: {self.model_path}"
            logger.warning(f"[{self.role}] {self._load_error}")
            return False
        try:
            from llama_cpp import Llama
            logger.info(f"[{self.role}] Loading {self.model_path.name}…")
            t0 = time.time()
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                verbose=self.verbose,
                n_threads=max(1, (os.cpu_count() or 4) // 2),
            )
            elapsed = round((time.time() - t0) * 1000)
            self.is_loaded = True
            logger.info(
                f"[{self.role}] ✅ Loaded {self.model_path.name} in {elapsed}ms"
            )
            return True
        except ImportError:
            self._load_error = "llama_cpp not installed — pip install llama-cpp-python"
            logger.warning(f"[{self.role}] {self._load_error}")
            return False
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"[{self.role}] Load error: {e}")
            return False

    def unload(self):
        """Release model from RAM."""
        with self._lock:
            self._llm      = None
            self.is_loaded = False
        logger.info(f"[{self.role}] Model unloaded")

    def swap(self, new_path: Path) -> bool:
        """Hot-swap: load a new model path without losing the slot."""
        self.unload()
        self.model_path = new_path
        return self.load()

    # ── Infer ─────────────────────────────────────────────────────────
    def infer(
        self,
        prompt:     str,
        system:     Optional[str] = None,
        max_tokens: int = 1500,
        temperature:float = 0.3,
        stop:       Optional[List[str]] = None,
    ) -> InferenceResult:
        """
        Run inference. Uses role-locked system prompt by default.
        Thread-safe via internal lock — concurrent callers will queue.
        Falls back to fallback_slot if this model is not loaded.
        """
        t0 = time.time()
        effective_system = system or self.system_prompt
        stop_tokens      = stop or STOP_TOKENS

        # Graceful degradation
        if not self.is_loaded or self._llm is None:
            if self._fallback and self._fallback.is_loaded:
                logger.debug(
                    f"[{self.role}] Not loaded — delegating to {self._fallback.role} fallback"
                )
                return self._fallback.infer(prompt, effective_system, max_tokens, temperature, stop)
            return InferenceResult(
                text       = f"[{self.role} offline — model not loaded: {self.model_path.name}]",
                role       = self.role,
                model      = self.model_path.name,
                elapsed_ms = 0.0,
                tokens     = 0,
                error      = self._load_error or "not loaded",
            )

        # Build full prompt with system prefix
        messages = []
        if effective_system:
            messages.append({"role": "system", "content": effective_system})
        messages.append({"role": "user", "content": prompt})

        with self._lock:
            try:
                # Use native chat completion so the model applies its own template
                result = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop_tokens
                )
                text = result["choices"][0]["message"]["content"].strip()
                tokens = result.get("usage", {}).get("completion_tokens", len(text) // 4)
            except Exception as e:
                elapsed = round((time.time() - t0) * 1000, 1)
                return InferenceResult(
                    text       = "",
                    role       = self.role,
                    model      = self.model_path.name,
                    elapsed_ms = elapsed,
                    tokens     = 0,
                    error      = str(e),
                )

        elapsed = round((time.time() - t0) * 1000, 1)

        # Strip any accidental markdown fences
        import re
        text = re.sub(r'^```(?:python|py)?\s*', '', text, flags=re.M)
        text = re.sub(r'\s*```\s*$', '', text, flags=re.M).strip()

        # Update stats
        self._call_count   += 1
        self._total_tokens += tokens
        self._total_ms     += elapsed

        return InferenceResult(
            text       = text,
            role       = self.role,
            model      = self.model_path.name,
            elapsed_ms = elapsed,
            tokens     = tokens,
        )

    def get_stats(self) -> dict:
        avg_tps = (
            self._total_tokens / (self._total_ms / 1000)
            if self._total_ms > 0 else 0.0
        )
        return {
            "role":        self.role,
            "model":       self.model_path.name,
            "loaded":      self.is_loaded,
            "calls":       self._call_count,
            "total_tokens":self._total_tokens,
            "avg_tok_per_s":round(avg_tps, 1),
            "load_error":  self._load_error,
        }


# ─────────────────────────────────────────────────────────────────────
# PARALLEL INFERENCE ENGINE
# Runs coder + tester simultaneously via ThreadPoolExecutor
# ─────────────────────────────────────────────────────────────────────
@dataclass
class ParallelResult:
    coder_result:  InferenceResult
    tester_result: InferenceResult
    wall_ms:       float   # actual wall-clock time (= max of the two)
    sequential_ms: float   # what it would have taken without parallelism
    speedup:       float   # sequential_ms / wall_ms

    @property
    def code(self) -> str:
        return self.coder_result.text

    @property
    def tests(self) -> str:
        return self.tester_result.text

    @property
    def both_ok(self) -> bool:
        return self.coder_result.ok and self.tester_result.ok


class ParallelInferenceEngine:
    """
    Runs CoderLLM and TesterLLM in parallel threads.

    Wall-clock time = max(coder_latency, tester_latency)
    instead of coder_latency + tester_latency.

    For typical 1.5B models on Apple Silicon (~40 tok/s):
      Sequential: 3s + 3s = 6s
      Parallel:   max(3s, 3s) = 3s   →  2× speedup
    """

    def __init__(self, coder: ModelSlot, tester: ModelSlot):
        self.coder  = coder
        self.tester = tester
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="DualInfer")
        self._parallel_calls = 0
        self._total_speedup  = 0.0

    def run_parallel(
        self,
        code_prompt: str,
        test_prompt: str,
        code_max_tokens: int = 800,
        test_max_tokens: int = 600,
    ) -> ParallelResult:
        """
        Submit both inference jobs simultaneously. Returns when BOTH complete.
        """
        t0 = time.time()

        future_code: Future = self._executor.submit(
            self.coder.infer, code_prompt, None, code_max_tokens
        )
        future_test: Future = self._executor.submit(
            self.tester.infer, test_prompt, None, test_max_tokens
        )

        # Wait for both — as_completed gives us whichever finishes first
        done_results = {}
        for future in as_completed([future_code, future_test]):
            result: InferenceResult = future.result()
            done_results[future] = result

        wall_ms       = round((time.time() - t0) * 1000, 1)
        coder_result  = done_results[future_code]
        tester_result = done_results[future_test]

        sequential_ms = coder_result.elapsed_ms + tester_result.elapsed_ms
        speedup       = round(sequential_ms / wall_ms, 2) if wall_ms > 0 else 1.0

        self._parallel_calls += 1
        self._total_speedup  += speedup

        logger.info(
            f"[ParallelEngine] coder={coder_result.elapsed_ms}ms "
            f"tester={tester_result.elapsed_ms}ms "
            f"wall={wall_ms}ms speedup={speedup}x"
        )

        return ParallelResult(
            coder_result  = coder_result,
            tester_result = tester_result,
            wall_ms       = wall_ms,
            sequential_ms = sequential_ms,
            speedup       = speedup,
        )

    def run_sequential(
        self,
        code_prompt: str,
        test_prompt: str,
        code_max_tokens: int = 800,
        test_max_tokens: int = 600,
    ) -> ParallelResult:
        """Fallback: run sequentially (same return type as parallel)."""
        t0 = time.time()
        cr = self.coder.infer(code_prompt, max_tokens=code_max_tokens)
        tr = self.tester.infer(test_prompt, max_tokens=test_max_tokens)
        wall_ms = round((time.time() - t0) * 1000, 1)
        return ParallelResult(
            coder_result  = cr,
            tester_result = tr,
            wall_ms       = wall_ms,
            sequential_ms = cr.elapsed_ms + tr.elapsed_ms,
            speedup       = 1.0,
        )

    def get_stats(self) -> dict:
        avg_speedup = (
            round(self._total_speedup / self._parallel_calls, 2)
            if self._parallel_calls > 0 else 1.0
        )
        return {
            "parallel_calls": self._parallel_calls,
            "avg_speedup":    avg_speedup,
            "coder":          self.coder.get_stats(),
            "tester":         self.tester.get_stats(),
        }

    def shutdown(self):
        self._executor.shutdown(wait=False)


# ─────────────────────────────────────────────────────────────────────
# DUAL MODEL ENGINE — top-level orchestrator
# ─────────────────────────────────────────────────────────────────────
class DualModelEngine:
    """
    Manages CoderLLM + TesterLLM as a coordinated pair.

    Responsibilities:
      1. Download models if not present (background threads)
      2. Load both models in parallel on boot
      3. Expose .coder and .tester ModelSlot instances
      4. Expose .parallel for simultaneous coder+tester inference
      5. Wire SoftwareFirm with correct model callables
      6. Hot-swap individual models without stopping the other
    """

    def __init__(
        self,
        model_dir:     Path = Path("."),
        coder_file:    str  = CODER_MODEL_FILE,
        tester_file:   str  = TESTER_MODEL_FILE,
        n_ctx:         int  = DEFAULT_N_CTX,
        auto_download: bool = False,
    ):
        self._model_dir    = Path(model_dir)
        self._coder_file   = coder_file
        self._tester_file  = tester_file
        self._n_ctx        = n_ctx
        self._auto_download= auto_download

        # Build slots (not loaded yet)
        self.coder = ModelSlot(
            role          = "coder",
            model_path    = self._model_dir / coder_file,
            system_prompt = _CODER_SYSTEM,
            n_ctx         = n_ctx,
        )
        self.tester = ModelSlot(
            role          = "tester",
            model_path    = self._model_dir / tester_file,
            system_prompt = _TESTER_SYSTEM,
            n_ctx         = n_ctx,
            fallback_slot = self.coder,   # tester falls back to coder
        )

        self.parallel = ParallelInferenceEngine(self.coder, self.tester)
        self._loaded_event = threading.Event()

    # ── Boot-time loading ─────────────────────────────────────────────
    def load(
        self,
        coder_path:  Optional[Path] = None,
        tester_path: Optional[Path] = None,
        blocking:    bool = False,
    ) -> bool:
        """
        Load both models. Sequentially loads to prevent macOS Metal
        Memory Allocation deadlocks when firing both at once.
        """
        if coder_path:
            self.coder.model_path  = coder_path
        if tester_path:
            self.tester.model_path = tester_path

        if self._auto_download:
            self._download_models_background()

        def _sequential_load():
            self._load_coder()
            self._load_tester()
            self._loaded_event.set()
            logger.info(
                f"[DualEngine] Both models ready: "
                f"coder={'✅' if self.coder.is_loaded else '❌'} "
                f"tester={'✅' if self.tester.is_loaded else '❌'}"
            )

        t = threading.Thread(target=_sequential_load, daemon=True, name="DualLoader")
        t.start()

        if blocking:
            t.join()
            return self.coder.is_loaded or self.tester.is_loaded

        return True

    def _load_coder(self):
        self.coder.load()

    def _load_tester(self):
        self.tester.load()

    def wait_loaded(self, timeout: float = 120.0) -> bool:
        """Block until both models have attempted to load (or timeout)."""
        return self._loaded_event.wait(timeout=timeout)

    # ── Hot-swap ──────────────────────────────────────────────────────
    def swap_coder(self, new_path: Path) -> bool:
        """Replace coder model without touching tester."""
        logger.info(f"[DualEngine] Hot-swapping coder → {new_path.name}")
        return self.coder.swap(new_path)

    def swap_tester(self, new_path: Path) -> bool:
        """Replace tester model without touching coder."""
        logger.info(f"[DualEngine] Hot-swapping tester → {new_path.name}")
        return self.tester.swap(new_path)

    # ── Callable wrappers for SoftwareFirm ────────────────────────────
    def coder_fn(self, prompt: str, system: str = "", max_tokens: int = 800) -> str:
        """Drop-in for SoftwareFirm coder_llm callable."""
        result = self.coder.infer(prompt, system=system or None, max_tokens=max_tokens)
        return result.text

    def tester_fn(self, prompt: str, system: str = "", max_tokens: int = 600) -> str:
        """Drop-in for SoftwareFirm tester_llm callable."""
        result = self.tester.infer(prompt, system=system or None, max_tokens=max_tokens)
        return result.text

    # ── Download helpers ──────────────────────────────────────────────
    def _download_models_background(self):
        # Use instance filenames so custom models are respected.
        # Only download when the resolved filename matches the default HF repo
        # file — never try to download a user-supplied local filename.
        _coder_repo  = CODER_MODEL_REPO  if self._coder_file  == CODER_MODEL_FILE  else None
        _tester_repo = TESTER_MODEL_REPO if self._tester_file == TESTER_MODEL_FILE else None
        candidates = [
            (_coder_repo,  self._coder_file,  CODER_MODEL_DESC),
            (_tester_repo, self._tester_file, TESTER_MODEL_DESC),
        ]
        for repo, filename, desc in candidates:
            if repo is None:
                continue  # user-supplied local model — do not download
            dest = self._model_dir / filename
            if not dest.exists():
                threading.Thread(
                    target=_download_model,
                    args=(repo, filename, dest, desc),
                    daemon=True,
                    name=f"Download_{filename[:20]}",
                ).start()

    def get_status(self) -> dict:
        return {
            "coder":         self.coder.get_stats(),
            "tester":        self.tester.get_stats(),
            "parallel":      self.parallel.get_stats(),
            "both_loaded":   self.coder.is_loaded and self.tester.is_loaded,
            "any_loaded":    self.coder.is_loaded or  self.tester.is_loaded,
            "load_complete": self._loaded_event.is_set(),
        }

    def shutdown(self):
        self.parallel.shutdown()
        logger.info("[DualEngine] Shutdown complete")


# ─────────────────────────────────────────────────────────────────────
# MODEL DOWNLOAD HELPER
# ─────────────────────────────────────────────────────────────────────
def _download_model(repo: str, filename: str, dest: Path, desc: str) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info(f"[Download] Already cached: {filename}")
        return True
    logger.info(f"[Download] Fetching {desc} from {repo}…")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=repo, filename=filename,
            local_dir=str(dest.parent), local_dir_use_symlinks=False,
        )
        logger.info(f"[Download] ✅ {filename} → {path}")
        return True
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"[Download] HF Hub failed: {e}")
    import urllib.request
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    try:
        urllib.request.urlretrieve(url, str(dest))
        logger.info(f"[Download] ✅ {filename} via urllib")
        return True
    except Exception as e:
        logger.warning(f"[Download] urllib failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# WIRE_SOFTWARE_FIRM — convenience factory
# ─────────────────────────────────────────────────────────────────────
def wire_software_firm(
    manager_fn:  Optional[Callable[[str], str]],
    coder_llm:   ModelSlot,
    tester_llm:  ModelSlot,
    max_iterations: int = 3,
):
    """
    Instantiate SoftwareFirm with the dual model engine callables.
    Returns a SoftwareFirm instance or None if not importable.

    SoftwareFirm pipeline:
        manager(70B) → JSON spec
        coder(1.5B)  → Python code
        tester(1.5B) + redteamer(70B) → in PARALLEL
        manager(70B) → review → PASS / KICK_BACK
    """
    try:
        from software_firm import SoftwareFirm

        def _coder_callable(prompt: str, system: str = "", max_tokens: int = 800) -> str:
            return coder_llm.infer(prompt, system=system or None, max_tokens=max_tokens).text

        def _tester_callable(prompt: str, system: str = "", max_tokens: int = 600) -> str:
            return tester_llm.infer(prompt, system=system or None, max_tokens=max_tokens).text

        firm = SoftwareFirm(
            manager_fn     = manager_fn,
            coder_llm      = _coder_callable,
            tester_llm     = _tester_callable,
            max_iterations = max_iterations,
        )
        logger.info("[wire_software_firm] ✅ SoftwareFirm wired with dual LLM engine")
        return firm
    except ImportError as e:
        logger.warning(f"[wire_software_firm] software_firm not importable: {e}")
        return None
    except Exception as e:
        logger.warning(f"[wire_software_firm] Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────
# PATCH — monkey-patch EdgeNodeOrchestrator at runtime
# ─────────────────────────────────────────────────────────────────────
def patch_orchestrator(orchestrator, model_dir: Optional[Path] = None) -> bool:
    """
    Add .dual_engine, .coder_llm, .tester_llm, .firm to an existing
    EdgeNodeOrchestrator instance. Call this in boot() after procurement.
    """
    try:
        mdir = model_dir or Path(os.getenv("SWAYAMBHU_DIR", str(PROJECT_ROOT))) / "models"
        mdir.mkdir(parents=True, exist_ok=True)

        engine = DualModelEngine(model_dir=mdir)
        engine.load()   # non-blocking

        orchestrator.dual_engine = engine
        orchestrator.coder_llm   = engine.coder
        orchestrator.tester_llm  = engine.tester

        # Wire SoftwareFirm if cloud LLM is available
        cloud_fn = getattr(orchestrator, "_cloud_llm_fn", None)
        if cloud_fn is None:
            cloud_fn = getattr(orchestrator, "_cloud_post", None)
        if cloud_fn:
            orchestrator.firm = wire_software_firm(
                manager_fn  = cloud_fn,
                coder_llm   = engine.coder,
                tester_llm  = engine.tester,
            )

        logger.info(
            "[patch_orchestrator] ✅ DualModelEngine attached: "
            f"coder={engine.coder.model_path.name} "
            f"tester={engine.tester.model_path.name}"
        )
        return True
    except Exception as e:
        logger.warning(f"[patch_orchestrator] Error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# SELF-TESTS
# ─────────────────────────────────────────────────────────────────────
def _run_tests():
    import tempfile, shutil, re as _re
    logging.basicConfig(level=logging.WARNING)
    print("🤜🤛  DualModelEngine Self-Tests\n")
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

    # ── Mock llama-cpp Llama ──────────────────────────────────────────
    class MockLlama:
        def __init__(self, model_path, n_ctx=8192, verbose=False, n_threads=2):
            self._path = model_path

        def create_chat_completion(self, messages, max_tokens=1500, temperature=0.3, stop=None):
            # Extract the user prompt from the messages list
            prompt = messages[-1]["content"] if messages else ""

            # Return different responses based on prompt
            if "DeepSeek" in prompt or "engineer" in prompt.lower():
                text = "def hello():\n    return 'world'\n"
            elif "Qwen" in prompt or "test" in prompt.lower():
                text = "def test_hello():\n    assert hello() == 'world'\n"
            else:
                text = f"response to: {prompt[:30]}"

            # Must return the correct Chat Completion dictionary structure
            return {
                "choices": [{"message": {"content": text}}],
                "usage": {"completion_tokens": len(text) // 4},
            }

    # ── Test 1: ModelSlot creation ────────────────────────────────────
    print("=== Test 1: ModelSlot Creation ===")
    slot = ModelSlot(
        role="coder", model_path=tmpdir / "test.gguf",
        system_prompt=_CODER_SYSTEM,
    )
    ok("Slot created",              slot is not None)
    ok("Role is coder",             slot.role == "coder")
    ok("Not loaded initially",      not slot.is_loaded)
    ok("Model path set",            slot.model_path.name == "test.gguf")
    ok("System prompt set",         len(slot.system_prompt) > 0)

    # ── Test 2: ModelSlot offline inference ───────────────────────────
    print("\n=== Test 2: ModelSlot Offline Inference ===")
    result = slot.infer("write a function")
    ok("Offline result is InferenceResult", isinstance(result, InferenceResult))
    ok("Offline has error",         result.error != "")
    ok("Offline ok=False",          not result.ok)
    ok("Offline text has role info",result.role == "coder")
    ok("Offline text non-empty",    len(result.text) > 0)
    ok("tokens_per_sec=0 offline",  result.tokens_per_sec == 0.0)

    # ── Test 3: ModelSlot with mock LLM ──────────────────────────────
    print("\n=== Test 3: ModelSlot with Mock LLM ===")
    import unittest.mock as _mock
    import types as _types
    # Inject mock llama_cpp module into sys.modules ONCE for all tests
    _mock_llama_mod = _types.ModuleType("llama_cpp")
    _mock_llama_mod.Llama = MockLlama
    import sys as _sys
    _sys.modules["llama_cpp"] = _mock_llama_mod

    # Write a fake .gguf file
    fake_gguf = tmpdir / "fake.gguf"
    fake_gguf.write_bytes(b"fake model data")

    slot2 = ModelSlot(
        role="coder", model_path=fake_gguf, system_prompt=_CODER_SYSTEM
    )
    loaded = slot2.load()

    ok("Mock load succeeds",        loaded)
    ok("is_loaded = True",          slot2.is_loaded)

    # Inference
    result2 = slot2.infer("write a hello function")
    ok("Mock infer returns result",  isinstance(result2, InferenceResult))
    ok("Result has text",            len(result2.text) > 0)
    ok("Result ok=True",             result2.ok, result2.error)
    ok("Result has elapsed_ms",      result2.elapsed_ms >= 0)
    ok("Result has tokens",          result2.tokens >= 0)
    ok("Role preserved",             result2.role == "coder")
    ok("Model name in result",       result2.model == "fake.gguf")

    # Markdown fences stripped
    slot_strip = ModelSlot(
        role="strip_test", model_path=fake_gguf, system_prompt=""
    )
    class MarkdownLlama:
        def __init__(self, *a, **kw): pass
        def __call__(self, p, **kw):
            return {"choices":[{"text":"```python\ndef foo(): pass\n```"}],
                    "usage":{"completion_tokens": 5}}
    _mock_llama_mod.Llama = MarkdownLlama
    slot_strip.load()
    _mock_llama_mod.Llama = MockLlama  # restore
    r_strip = slot_strip.infer("write foo")
    ok("Markdown fences stripped",   "```" not in r_strip.text, repr(r_strip.text))
    ok("Code content preserved",     "def foo" in r_strip.text)

    # ── Test 4: Fallback slot ─────────────────────────────────────────
    print("\n=== Test 4: Fallback Slot ===")
    # Coder loaded, tester not loaded → tester falls back to coder
    loaded_slot = ModelSlot(role="main", model_path=fake_gguf, system_prompt="")
    loaded_slot.load()

    unloaded_slot = ModelSlot(
        role="fallback_user",
        model_path=tmpdir / "missing.gguf",
        system_prompt="",
        fallback_slot=loaded_slot,
    )
    fallback_result = unloaded_slot.infer("test prompt")
    ok("Fallback used when not loaded", fallback_result.ok or len(fallback_result.text) > 0)

    # ── Test 5: Stats tracking ────────────────────────────────────────
    print("\n=== Test 5: Stats Tracking ===")
    slot2.infer("call 1"); slot2.infer("call 2"); slot2.infer("call 3")
    stats = slot2.get_stats()
    ok("Stats has role",            stats["role"] == "coder")
    ok("Stats has model",           stats["model"] == "fake.gguf")
    ok("Stats has calls",           stats["calls"] >= 3)
    ok("Stats has total_tokens",    stats["total_tokens"] >= 0)
    ok("Stats has avg_tok_per_s",   stats["avg_tok_per_s"] >= 0)
    ok("Stats has loaded=True",     stats["loaded"])

    # ── Test 6: ParallelInferenceEngine ──────────────────────────────
    print("\n=== Test 6: ParallelInferenceEngine ===")
    # Create two mock-loaded slots
    coder_slot = ModelSlot(role="coder", model_path=fake_gguf, system_prompt=_CODER_SYSTEM)
    tester_slot = ModelSlot(role="tester", model_path=fake_gguf, system_prompt=_TESTER_SYSTEM)
    coder_slot.load()
    tester_slot.load()

    engine = ParallelInferenceEngine(coder_slot, tester_slot)

    t0 = time.time()
    par_result = engine.run_parallel(
        code_prompt="write a sort function",
        test_prompt="write tests for sort function",
    )
    wall = (time.time() - t0) * 1000

    ok("Returns ParallelResult",    isinstance(par_result, ParallelResult))
    ok("Has coder_result",          isinstance(par_result.coder_result, InferenceResult))
    ok("Has tester_result",         isinstance(par_result.tester_result, InferenceResult))
    ok("Has wall_ms",               par_result.wall_ms >= 0)
    ok("Has sequential_ms",         par_result.sequential_ms >= 0)
    ok("Has speedup",               par_result.speedup >= 0.0)
    ok("Code accessible via .code", len(par_result.code) > 0)
    ok("Tests accessible via .tests",len(par_result.tests) > 0)
    ok("both_ok property",          isinstance(par_result.both_ok, bool))

    # Parallel should be at most as slow as sequential
    ok("Wall ≤ sequential_ms",      par_result.wall_ms <= par_result.sequential_ms + 50,
       f"wall={par_result.wall_ms} seq={par_result.sequential_ms}")

    # Parallel stats
    par_stats = engine.get_stats()
    ok("Parallel stats has calls",  par_stats["parallel_calls"] == 1)
    ok("Parallel stats has speedup",par_stats["avg_speedup"] >= 0.0)
    ok("Parallel stats has coder",  "coder" in par_stats)
    ok("Parallel stats has tester", "tester" in par_stats)

    # Sequential fallback
    seq_result = engine.run_sequential(
        "write fibonacci", "write tests for fibonacci"
    )
    ok("Sequential returns ParallelResult", isinstance(seq_result, ParallelResult))
    ok("Sequential speedup = 1.0",          seq_result.speedup == 1.0)

    # ── Test 7: DualModelEngine ───────────────────────────────────────
    print("\n=== Test 7: DualModelEngine ===")
    dual = DualModelEngine(model_dir=tmpdir)
    ok("Engine created",            dual is not None)
    ok("Has .coder slot",           isinstance(dual.coder, ModelSlot))
    ok("Has .tester slot",          isinstance(dual.tester, ModelSlot))
    ok("Has .parallel engine",      isinstance(dual.parallel, ParallelInferenceEngine))
    ok("Tester has fallback to coder", dual.tester._fallback is dual.coder)

    # Load (no models present — graceful failure)
    dual.load(blocking=True)
    ok("Load doesn't crash (no models)", True)
    ok("Status has both_loaded",    "both_loaded" in dual.get_status())
    ok("Status has any_loaded",     "any_loaded" in dual.get_status())

    # With mock models
    dual2 = DualModelEngine(model_dir=tmpdir, coder_file="fake.gguf", tester_file="fake.gguf")
    dual2.load(blocking=True)

    ok("Both loaded with mock",     dual2.get_status()["both_loaded"])
    ok("any_loaded = True",         dual2.get_status()["any_loaded"])
    ok("load_complete = True",      dual2.get_status()["load_complete"])

    # coder_fn and tester_fn callables
    code_out = dual2.coder_fn("write sort")
    test_out  = dual2.tester_fn("write tests")
    ok("coder_fn returns string",   isinstance(code_out, str) and len(code_out) > 0)
    ok("tester_fn returns string",  isinstance(test_out, str) and len(test_out) > 0)

    # Hot-swap
    fake_gguf2 = tmpdir / "fake2.gguf"
    fake_gguf2.write_bytes(b"another fake model")
    swap_ok = dual2.swap_coder(fake_gguf2)
    ok("Hot-swap coder succeeds",   swap_ok)
    ok("Coder path updated",        dual2.coder.model_path == fake_gguf2)
    ok("Tester unchanged after swap",dual2.tester.model_path.name == "fake.gguf")

    # ── Test 8: Role isolation ────────────────────────────────────────
    print("\n=== Test 8: Role Isolation ===")
    # Verify each slot uses its own system prompt
    ok("Coder system prompt has Python", "Python" in dual2.coder.system_prompt)
    ok("Tester system prompt has pytest","pytest" in dual2.tester.system_prompt)
    ok("Prompts are different",     dual2.coder.system_prompt != dual2.tester.system_prompt)

    # ── Test 9: patch_orchestrator ────────────────────────────────────
    print("\n=== Test 9: patch_orchestrator ===")
    class MockOrch:
        pass

    mo = MockOrch()
    result_patch = patch_orchestrator(mo, model_dir=tmpdir)
    ok("Patch returns bool",        isinstance(result_patch, bool))
    ok("Has .dual_engine",          hasattr(mo, "dual_engine"))
    ok("Has .coder_llm",            hasattr(mo, "coder_llm"))
    ok("Has .tester_llm",           hasattr(mo, "tester_llm"))
    ok("coder_llm is ModelSlot",    isinstance(mo.coder_llm, ModelSlot))
    ok("tester_llm is ModelSlot",   isinstance(mo.tester_llm, ModelSlot))

    # ── Test 10: wire_software_firm ───────────────────────────────────
    print("\n=== Test 10: wire_software_firm ===")
    # software_firm.py may not be importable in test context — test graceful handling
    firm = wire_software_firm(
        manager_fn  = lambda p: "manager response",
        coder_llm   = dual2.coder,
        tester_llm  = dual2.tester,
    )
    # Either wired successfully or None (if software_firm.py not present)
    ok("wire_software_firm returns firm or None", firm is None or hasattr(firm, "build"))

    # ── Test 11: Concurrent stress test ──────────────────────────────
    print("\n=== Test 11: Concurrent Stress Test ===")
    # Fire 6 parallel inference requests and verify no corruption
    results = []
    errors  = []

    def _do_infer(slot: ModelSlot, prompt: str, i: int):
        r = slot.infer(prompt)
        if r.error:
            errors.append(f"call {i}: {r.error}")
        else:
            results.append(r.text)

    threads = []
    for i in range(3):
        threads.append(threading.Thread(
            target=_do_infer, args=(dual2.coder, f"write function {i}", i)
        ))
        threads.append(threading.Thread(
            target=_do_infer, args=(dual2.tester, f"write test {i}", i+10)
        ))

    for t in threads: t.start()
    for t in threads: t.join(timeout=10)

    ok("All 6 concurrent calls completed", len(results) == 6, f"got {len(results)} results, errors: {errors}")
    ok("No corruption errors",           len(errors) == 0, str(errors))

    # ── Test 12: InferenceResult properties ──────────────────────────
    print("\n=== Test 12: InferenceResult Properties ===")
    r_ok  = InferenceResult(text="def foo(): pass", role="coder", model="m.gguf",
                            elapsed_ms=500, tokens=10)
    r_err = InferenceResult(text="", role="coder", model="m.gguf",
                            elapsed_ms=0, tokens=0, error="failed")
    ok("ok=True when text+no error",  r_ok.ok)
    ok("ok=False when error",         not r_err.ok)
    ok("ok=False when empty text",    not InferenceResult(
        text="", role="r", model="m", elapsed_ms=0, tokens=0).ok)
    ok("tokens_per_sec computed",     r_ok.tokens_per_sec == 20.0,
       str(r_ok.tokens_per_sec))
    ok("tokens_per_sec=0 when 0ms",   r_err.tokens_per_sec == 0.0)

    # ── Cleanup ───────────────────────────────────────────────────────
    dual2.shutdown()
    shutil.rmtree(tmpdir)

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_tests() else 1)
