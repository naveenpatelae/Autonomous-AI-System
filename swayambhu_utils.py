#!/usr/bin/env python3
# =====================================================================
# 🛠️  SWAYAMBHU UTILS  —  Safe Parsing, Env Helpers, Schema Guards
# FIX-7: "Truncated Translator" — complete safe_json_parse + helpers
# =====================================================================

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

logger = logging.getLogger("SwayambhuUtils")
F = TypeVar("F", bound=Callable[..., Any])


try:
    PROJECT_ROOT = Path(__file__).parent.resolve()
except NameError:
    # Fallback for interactive shells
    PROJECT_ROOT = Path(os.getcwd()).resolve()

_JUDGE_RUBRIC = """\
You are a strict quality judge for AI training data.
Rate this response 0.0-1.0 on: CORRECTNESS, COMPLETENESS, CONCISENESS, SAFETY.
PROMPT: {prompt}
RESPONSE: {response}
Return ONLY: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}"""

# ─────────────────────────────────────────────────────────────────────
# safe_json_parse — 6-layer crash-proof parser
# ─────────────────────────────────────────────────────────────────────

def safe_json_parse(raw, fallback=None, expected_type=dict, label=""):
    if fallback is None:
        fallback = {} if expected_type is dict else []
    if not raw or not isinstance(raw, str):
        return fallback
    text = raw.strip()
    tag = f"[safe_json_parse{':'+label if label else ''}]"

    # Layer 1: direct
    try:
        result = json.loads(text)
        if isinstance(result, expected_type):
            return result
        if expected_type is dict and isinstance(result, list):
            return {"items": result}
        if expected_type is list and isinstance(result, dict):
            return [result]
        return result
    except json.JSONDecodeError:
        pass

    # Layer 2: strip markdown fences
    stripped = re.sub(r'^```(?:json)?\s*', '', text, flags=re.M)
    stripped = re.sub(r'\s*```\s*$', '', stripped, flags=re.M).strip()
    if stripped != text:
        try:
            result = json.loads(stripped)
            if isinstance(result, expected_type):
                return result
        except json.JSONDecodeError:
            pass

    # Layer 3: extract first balanced JSON block
    opener = '{' if expected_type is dict else '['
    closer = '}' if expected_type is dict else ']'
    extracted = _extract_json_block(text, opener, closer)
    if extracted:
        try:
            result = json.loads(extracted)
            if isinstance(result, expected_type):
                return result
        except json.JSONDecodeError:
            pass

    alt_opener = '[' if expected_type is dict else '{'
    alt_closer = ']' if expected_type is dict else '}'
    alt_extracted = _extract_json_block(text, alt_opener, alt_closer)
    if alt_extracted:
        try:
            result = json.loads(alt_extracted)
            if expected_type is dict and isinstance(result, list):
                return {"items": result}
            if expected_type is list and isinstance(result, dict):
                return [result]
        except json.JSONDecodeError:
            pass

    # Layer 4: fix common LLM mistakes
    fixable = extracted or stripped or text
    fixed = _fix_llm_json(fixable)
    if fixed != fixable:
        try:
            result = json.loads(fixed)
            if isinstance(result, expected_type):
                return result
        except json.JSONDecodeError:
            pass

    # Layer 5: repair truncation
    repaired = _repair_truncated(fixable, expected_type)
    if repaired:
        try:
            result = json.loads(repaired)
            if isinstance(result, expected_type):
                logger.debug(f"{tag} Recovered via truncation repair")
                return result
        except json.JSONDecodeError:
            pass

    # Layer 6: fallback
    if text:
        logger.warning(f"{tag} All JSON parse attempts failed. Raw: {repr(text[:120])}")
    return fallback


def _extract_json_block(text, opener, closer):
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False; continue
        if ch == '\\' and in_string:
            escape_next = True; continue
        if ch == '"' and not escape_next:
            in_string = not in_string; continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def _fix_llm_json(text):
    fixed = text
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    fixed = re.sub(r'\bTrue\b',  'true',  fixed)
    fixed = re.sub(r'\bFalse\b', 'false', fixed)
    fixed = re.sub(r'\bNone\b',  'null',  fixed)
    if '"' not in fixed and "'" in fixed:
        fixed = fixed.replace("'", '"')
    fixed = fixed.replace('...', 'null')
    fixed = re.sub(r'//[^\n]*', '', fixed)
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    return fixed


def _repair_truncated(text, expected_type):
    if not text:
        return None
    stack = []
    in_string = False
    escape_next = False
    close_map = {'{': '}', '[': ']'}
    open_set = set('{[')
    close_set = set('}]')
    for ch in text:
        if escape_next:
            escape_next = False; continue
        if ch == '\\' and in_string:
            escape_next = True; continue
        if ch == '"':
            in_string = not in_string; continue
        if in_string:
            continue
        if ch in open_set:
            stack.append(ch)
        elif ch in close_set:
            if stack and close_map.get(stack[-1]) == ch:
                stack.pop()
    if not stack:
        return None
    closing = "".join(close_map[ch] for ch in reversed(stack))
    repaired = text + closing
    if in_string:
        repaired = text + '"' + closing
    return repaired


# ─────────────────────────────────────────────────────────────────────
# Response normalisers
# ─────────────────────────────────────────────────────────────────────

def safe_extract_message(response, default=""):
    if response is None:
        return default
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        for key in ("message", "response", "text", "content", "answer"):
            val = response.get(key)
            if isinstance(val, str) and val:
                return val
        return json.dumps(response)
    return str(response) if response else default


def safe_extract_plan(response):
    if not isinstance(response, dict):
        return []
    plan = response.get("plan", [])
    if not isinstance(plan, list):
        return []
    return [s for s in plan if isinstance(s, dict) and s]


def flatten_response(response):
    if response is None:
        return {"message": "", "plan": [], "high_stakes": False, "source": "none", "error": "null response"}
    if isinstance(response, str):
        parsed = safe_json_parse(response, fallback=None)
        if parsed and isinstance(parsed, dict):
            response = parsed
        else:
            return {"message": response, "plan": [], "high_stakes": False, "source": "str", "error": ""}
    if isinstance(response, dict):
        return {
            "message":     safe_extract_message(response),
            "plan":        safe_extract_plan(response),
            "high_stakes": bool(response.get("high_stakes", False)),
            "source":      str(response.get("source", response.get("mode", "unknown"))),
            "error":       str(response.get("error", "")),
        }
    return {"message": str(response), "plan": [], "high_stakes": False,
            "source": "unknown", "error": f"unexpected type: {type(response).__name__}"}


# ─────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────

def env_require(*names, hint=""):
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        lines = [f"Missing required environment variables{' for '+hint if hint else ''}:"]
        for var in missing:
            lines.append(f"  export {var}=<your_value>")
        lines.append("\nAdd these to your .env file or shell profile.")
        raise EnvironmentError("\n".join(lines))
    return {n: os.environ[n] for n in names}


def env_get(name, default="", cast=str):
    raw = os.getenv(name, "")
    if not raw:
        return default
    if cast is bool:
        return raw.lower() in ("1", "true", "yes", "on")
    try:
        return cast(raw)
    except (ValueError, TypeError):
        return default


def load_dotenv(path=None):
    candidates = []
    if path:
        candidates.append(Path(path))
    swayambhu_dir = os.getenv("SWAYAMBHU_DIR", "")
    if swayambhu_dir:
        candidates.append(Path(swayambhu_dir) / ".env")
    candidates.append(PROJECT_ROOT / ".env")
    candidates.append(Path(".env"))
    for candidate in candidates:
        if candidate.exists():
            return _parse_dotenv(candidate)
    return 0


def _parse_dotenv(path):
    loaded = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
                loaded += 1
    except Exception as e:
        logger.warning(f"[load_dotenv] Error reading {path}: {e}")
    return loaded


# ─────────────────────────────────────────────────────────────────────
# String utilities
# ─────────────────────────────────────────────────────────────────────

def truncate(text, max_len=200, suffix="…"):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    if len(text) <= max_len:
        return text
    cut = text[:max_len - len(suffix)]
    last_space = cut.rfind(" ")
    if last_space > max_len * 0.7:
        cut = cut[:last_space]
    return cut + suffix


def redact_secrets(text):
    patterns = [
        (r'sk-[A-Za-z0-9]{20,}',                        '[OPENAI_KEY_REDACTED]'),
        (r'AKIA[0-9A-Z]{16}',                            '[AWS_KEY_REDACTED]'),
        (r'(?i)(password|passwd|secret|token|api_key)\s*[:=]\s*\S+', r'\1=[REDACTED]'),
        (r'Bearer\s+[A-Za-z0-9._\-]{20,}',              'Bearer [TOKEN_REDACTED]'),
    ]
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    return result


# ─────────────────────────────────────────────────────────────────────
# Retry decorator
# ─────────────────────────────────────────────────────────────────────

def retry(max_attempts=3, delay=1.0, backoff=2.0, exceptions=(Exception,), logger_name=""):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _log = logging.getLogger(logger_name or func.__module__ or "retry")
            current_delay = delay
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_attempts:
                        _log.warning(f"[retry] {func.__name__} attempt {attempt}/{max_attempts} failed: {e}. Retrying in {current_delay:.1f}s…")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        _log.error(f"[retry] {func.__name__} failed after {max_attempts} attempts: {e}")
            raise last_exc
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────
# Typed config
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SwayambhuConfig:
    brain_url:             str  = ""
    client_id:             str  = "default"
    swayambhu_dir: Path = field(default_factory=lambda: PROJECT_ROOT)
    local_llm_path:        Path = field(default_factory=lambda: Path("swayambhu_local.gguf"))
    draft_llm_path:        Path = field(default_factory=lambda: Path("swayambhu_draft_1b.gguf"))
    edge_server_port:      int  = 8003
    v13_api_port:          int  = 8004
    avatar_port:           int  = 8007
    elevenlabs_api_key:    str  = ""
    elevenlabs_voice_id:   str  = "21m00Tcm4TlvDq8ikWAM"
    tailscale_authkey:     str  = ""
    firebase_b64:          str  = ""
    debug:                 bool = False
    headless:              bool = True

    @classmethod
    def from_env(cls, auto_load_dotenv=True):
        if auto_load_dotenv:
            load_dotenv()
        swayambhu_dir = Path(os.getenv("SWAYAMBHU_DIR", str(PROJECT_ROOT)))

        return cls(
            brain_url           = os.getenv("BRAIN_URL", ""),
            client_id           = os.getenv("SWAYAMBHU_CLIENT_ID", "default"),
            swayambhu_dir       = swayambhu_dir,
            local_llm_path      = Path(os.getenv("LOCAL_LLM_PATH",
                                       str(swayambhu_dir / "models" / "swayambhu_local.gguf"))),
            draft_llm_path      = Path(os.getenv("DRAFT_LLM_PATH",
                                       str(swayambhu_dir / "models" / "swayambhu_draft_1b.gguf"))),
            edge_server_port    = env_get("EDGE_SERVER_PORT", 8003, int),
            v13_api_port        = env_get("V13_API_PORT", 8004, int),
            avatar_port         = env_get("AVATAR_PORT", 8007, int),
            elevenlabs_api_key  = os.getenv("ELEVENLABS_API_KEY", ""),
            elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            tailscale_authkey   = os.getenv("TAILSCALE_AUTHKEY", ""),
            firebase_b64        = os.getenv("FIREBASE_B64", ""),
            debug               = env_get("DEBUG", False, bool),
            headless            = env_get("HEADLESS", True, bool),
        )

    def is_cloud_configured(self):
        return bool(self.brain_url)

    def is_tts_configured(self):
        return bool(self.elevenlabs_api_key)

    def to_dict(self):
        return {
            "brain_url":        self.brain_url,
            "client_id":        self.client_id,
            "swayambhu_dir":    str(self.swayambhu_dir),
            "local_llm_path":   str(self.local_llm_path),
            "draft_llm_path":   str(self.draft_llm_path),
            "edge_server_port": self.edge_server_port,
            "avatar_port":      self.avatar_port,
            "cloud_configured": self.is_cloud_configured(),
            "tts_configured":   self.is_tts_configured(),
            "debug":            self.debug,
        }


# ─────────────────────────────────────────────────────────────────────
# SELF-TESTS
# ─────────────────────────────────────────────────────────────────────
def _run_tests():
    import tempfile, shutil
    logging.basicConfig(level=logging.WARNING)
    print("🛠️  SwayambhuUtils Self-Tests\n")
    passed = failed = 0

    def ok(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed += 1

    # Layer 1: direct
    print("=== safe_json_parse: Layer 1 — Direct ===")
    ok("Parse valid dict",          safe_json_parse('{"a": 1}') == {"a": 1})
    ok("Parse valid list",          safe_json_parse('[1,2,3]', expected_type=list) == [1,2,3])
    ok("Parse nested",              safe_json_parse('{"a":{"b":2}}') == {"a":{"b":2}})
    ok("Returns fallback on None",  safe_json_parse(None, fallback={"x":1}) == {"x":1})
    ok("Returns fallback on empty", safe_json_parse("", fallback={"x":1}) == {"x":1})
    ok("List→dict wrap",            safe_json_parse('[1,2]') == {"items":[1,2]})

    # Layer 2: fences
    print("\n=== Layer 2 — Markdown Fences ===")
    ok("json fence stripped",       safe_json_parse('```json\n{"a":1}\n```')["a"] == 1)
    ok("bare fence stripped",       safe_json_parse('```\n{"a":1}\n```')["a"] == 1)

    # Layer 3: extract
    print("\n=== Layer 3 — Extract Block ===")
    ok("Prefix text handled",       safe_json_parse('Here:\n{"steps":["a"]}').get("steps") == ["a"])
    ok("Suffix text handled",       safe_json_parse('{"result":42} Done')["result"] == 42)

    # Layer 4: LLM mistakes
    print("\n=== Layer 4 — LLM Mistakes ===")
    ok("Trailing comma dict",       safe_json_parse('{"a":1,"b":2,}')["a"] == 1)
    ok("Trailing comma list",       safe_json_parse('{"items":[1,2,3,]}')["items"] == [1,2,3])
    ok("Python True/False/None",    safe_json_parse('{"a":True,"b":False,"c":None}') == {"a":True,"b":False,"c":None})

    # Layer 5: truncation
    print("\n=== Layer 5 — Truncation Repair ===")
    r = safe_json_parse('{"message":"hello","plan":[{"action":"open"}')
    ok("Truncated dict repaired",   isinstance(r, dict))
    ok("Message preserved",         r.get("message") == "hello", str(r))
    rl = safe_json_parse('[{"a":1},{"b":2', expected_type=list)
    ok("Truncated list repaired",   isinstance(rl, list))

    # Layer 6: fallback
    print("\n=== Layer 6 — Fallback ===")
    ok("Garbage→fallback",          safe_json_parse("not json", fallback={"ok":1}) == {"ok":1})
    ok("Empty→empty dict",          safe_json_parse("") == {})
    ok("No crash on garbage",       safe_json_parse("totally garbage") == {})

    # Real LLM outputs
    print("\n=== Real LLM Output Shapes ===")
    plan_output = '```json\n{"message":"Opening Safari.","plan":[{"action":"actuate","params":{"script":"open -a Safari"}}],"high_stakes":false}\n```'
    r2 = safe_json_parse(plan_output)
    ok("Real plan output",          "message" in r2)
    ok("Plan is list",              isinstance(r2.get("plan"), list))

    complex_plan = '{"message":"Done","plan":[{"action":"actuate","params":{"script":"open Safari"}},{"action":"speak","params":{"text":"done"}}],"high_stakes":false}'
    r3 = safe_json_parse(complex_plan)
    ok("Complex plan parsed",       len(r3.get("plan",[])) == 2)

    # _extract_json_block
    print("\n=== _extract_json_block ===")
    ok("Extracts dict",             _extract_json_block('abc {"x":1} xyz', '{', '}') == '{"x":1}')
    ok("Extracts list",             _extract_json_block('result: [1,2,3] end', '[', ']') == '[1,2,3]')
    ok("Returns None if absent",    _extract_json_block('no json', '{', '}') is None)
    ok("Handles nested",            _extract_json_block('{"a":{"b":1}}', '{', '}') == '{"a":{"b":1}}')

    # _fix_llm_json
    print("\n=== _fix_llm_json ===")
    ok("Fixes trailing comma",      _fix_llm_json('{"a":1,}') == '{"a":1}')
    ok("Fixes Python True",         'true' in _fix_llm_json('{"x":True}'))
    ok("Fixes Python None",         'null' in _fix_llm_json('{"x":None}'))

    # _repair_truncated
    print("\n=== _repair_truncated ===")
    rep = _repair_truncated('{"a":1,"b":[1,2', dict)
    ok("Repairs unclosed dict+list", rep is not None and rep.endswith(']}'), str(rep))
    ok("Returns None for balanced",  _repair_truncated('{"a":1}', dict) is None)

    # safe_extract_message
    print("\n=== safe_extract_message ===")
    ok("From message key",          safe_extract_message({"message":"hi"}) == "hi")
    ok("From response key",         safe_extract_message({"response":"hi"}) == "hi")
    ok("Passthrough str",           safe_extract_message("hi") == "hi")
    ok("Default on None",           safe_extract_message(None, "def") == "def")

    # safe_extract_plan
    print("\n=== safe_extract_plan ===")
    ok("Extracts plan",             safe_extract_plan({"plan":[{"action":"x"}]}) == [{"action":"x"}])
    ok("Empty on missing",          safe_extract_plan({}) == [])
    ok("Filters non-dict",          safe_extract_plan({"plan":[{"a":1},"bad",None]}) == [{"a":1}])

    # flatten_response
    print("\n=== flatten_response ===")
    r4 = flatten_response({"message":"hi","plan":[],"high_stakes":True})
    ok("Flatten dict",              r4["message"] == "hi" and r4["high_stakes"] is True)
    ok("Flatten str",               flatten_response("plain")["message"] == "plain")
    ok("Flatten None",              flatten_response(None)["error"] != "")
    ok("Flatten JSON str",          flatten_response('{"message":"hello"}')["message"] == "hello")

    # truncate
    print("\n=== truncate ===")
    ok("Short unchanged",           truncate("hello", 100) == "hello")
    ok("Long truncated",            len(truncate("x"*300, 100)) <= 102)
    ok("Ends with ellipsis",        truncate("x "*200, 50).endswith("…"))
    ok("None handled",              isinstance(truncate(None, 50), str))

    # redact_secrets
    print("\n=== redact_secrets ===")
    ok("OpenAI key redacted",       "REDACTED" in redact_secrets("key=sk-abcdefghijklmnop1234567890"))
    ok("AWS key redacted",          "REDACTED" in redact_secrets("AKIAIOSFODNN7EXAMPLE"))
    ok("Normal text unchanged",     redact_secrets("hello world") == "hello world")

    # retry
    print("\n=== retry decorator ===")
    attempts = [0]
    @retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
    def flaky():
        attempts[0] += 1
        if attempts[0] < 3: raise ValueError("not yet")
        return "success"
    ok("Retries and succeeds",      flaky() == "success")
    ok("Took 3 attempts",           attempts[0] == 3)
    fail_count = [0]
    @retry(max_attempts=2, delay=0.01, exceptions=(RuntimeError,))
    def always_fails():
        fail_count[0] += 1
        raise RuntimeError("always fails")
    try:
        always_fails()
        ok("Should have raised",    False)
    except RuntimeError:
        ok("Raises after max",      True)
    ok("Made 2 attempts",           fail_count[0] == 2)

    # env_get
    print("\n=== env_get ===")
    os.environ["TEST_PORT_UTILS"] = "9999"
    ok("Int cast",                  env_get("TEST_PORT_UTILS", 0, int) == 9999)
    os.environ["TEST_BOOL_UTILS"] = "true"
    ok("Bool cast true",            env_get("TEST_BOOL_UTILS", False, bool) is True)
    ok("Missing→default",           env_get("DEFINITELY_NOT_SET_XYZ999", "default") == "default")

    # env_require
    print("\n=== env_require ===")
    os.environ["TEST_REQ_A"] = "val_a"
    ok("Returns values",            env_require("TEST_REQ_A")["TEST_REQ_A"] == "val_a")
    try:
        env_require("MISSING_VAR_UTILS_XYZ")
        ok("Missing var raises",    False)
    except EnvironmentError as e:
        ok("EnvironmentError raised",True)
        ok("Error names the var",    "MISSING_VAR_UTILS_XYZ" in str(e))

    # SwayambhuConfig
    print("\n=== SwayambhuConfig ===")
    cfg = SwayambhuConfig.from_env(auto_load_dotenv=False)
    ok("Config loads",              isinstance(cfg, SwayambhuConfig))
    ok("Has edge_server_port",      cfg.edge_server_port > 0)
    ok("Has avatar_port",           cfg.avatar_port > 0)
    ok("to_dict has keys",          "client_id" in cfg.to_dict())
    ok("swayambhu_dir is Path",     isinstance(cfg.swayambhu_dir, Path))
    ok("is_cloud_configured False", not cfg.is_cloud_configured() or bool(os.getenv("BRAIN_URL")))

    # load_dotenv
    print("\n=== load_dotenv ===")
    tmpdir = Path(tempfile.mkdtemp())
    env_file = tmpdir / ".env"
    env_file.write_text("TEST_DOTENV_VAR=hello\n# comment\nTEST_DOTENV_NUM=42\nTEST_DOTENV_QUOTED=\"quoted value\"\n")
    n_loaded = _parse_dotenv(env_file)
    ok("Loads 3 vars",              n_loaded == 3, f"loaded {n_loaded}")
    ok("Value correct",             os.getenv("TEST_DOTENV_VAR") == "hello")
    ok("Quoted unquoted",           os.getenv("TEST_DOTENV_QUOTED") == "quoted value")
    os.environ["TEST_ALREADY"] = "original"
    env2 = tmpdir / ".env2"
    env2.write_text("TEST_ALREADY=overwritten\n")
    _parse_dotenv(env2)
    ok("Existing not overwritten",  os.getenv("TEST_ALREADY") == "original")

    shutil.rmtree(tmpdir)

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run_tests() else 1)
