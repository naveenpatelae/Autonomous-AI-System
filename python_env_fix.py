#!/usr/bin/env python3
# =====================================================================
# 🐍 PYTHON_ENV_FIX.PY  —  Fix 1: Python 3.14 Dependency Wall
#
# Problem: mediapipe, torch MPS, llama-cpp-python all require Python
#          3.12 or below. Python 3.14 (pre-release) breaks them.
#
# This script:
#   1. Detects your current Python version
#   2. Checks which critical libraries are currently broken
#   3. Prints the exact brew/pyenv commands to install Python 3.12
#   4. Generates a requirements_3_12.txt with pinned versions
#   5. Validates the new environment after switching
#   6. Patches swayambhu_body_3.py's gesture_tracker import guard
#      to print a human-readable error instead of silently failing
#
# Run this BEFORE running setup.sh if you are on Python 3.14.
# =====================================================================

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────
# COMPATIBILITY MATRIX
# Python version → compatible: (min_py, max_py_exclusive)
# ─────────────────────────────────────────────────────────────────────
COMPATIBILITY_MATRIX: Dict[str, dict] = {
    "mediapipe": {
        "min_py": (3, 8),
        "max_py": (3, 13),   # 3.13 is the last wheel on PyPI as of 2026
        "reason": "No pre-built wheels for Python 3.14 yet.",
        "install": "pip install mediapipe>=0.10.14",
        "critical": True,
    },
    "torch": {
        "min_py": (3, 9),
        "max_py": (3, 13),
        "reason": "PyTorch MPS (Apple Silicon GPU) not yet compiled for 3.14.",
        "install": "pip install torch torchvision torchaudio",
        "critical": False,   # MLX is the primary path; torch is fallback
    },
    "llama-cpp-python": {
        "min_py": (3, 8),
        "max_py": (3, 13),
        "reason": "C++ extension build fails on 3.14 due to ABI changes.",
        "install": 'CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python',
        "critical": True,
    },
    "mlx": {
        "min_py": (3, 9),
        "max_py": (3, 14),   # mlx already supports 3.13, 3.14 in progress
        "reason": "MLX is Apple-native, tracks Python closely.",
        "install": "pip install mlx mlx-lm",
        "critical": False,
    },
    "pyaudio": {
        "min_py": (3, 7),
        "max_py": (3, 13),
        "reason": "PortAudio binding breaks on 3.14.",
        "install": "pip install pyaudio",
        "critical": False,
    },
}

TARGET_PYTHON = "3.12"   # The sweet spot


# ─────────────────────────────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────────────────────────────

def get_current_python() -> Tuple[int, int, int]:
    """Return (major, minor, micro) of the running Python."""
    v = sys.version_info
    return (v.major, v.minor, v.micro)


def is_python_compatible(min_py: tuple, max_py: tuple) -> bool:
    """Return True if running Python is in [min_py, max_py)."""
    current = get_current_python()[:2]
    return min_py <= current < max_py


def check_library_importable(lib_name: str) -> Tuple[bool, str]:
    """Try to import a library. Returns (importable, version_or_error)."""
    import_name = lib_name.replace("-", "_").replace("llama_cpp_python", "llama_cpp")
    if lib_name == "llama-cpp-python":
        import_name = "llama_cpp"
    elif lib_name == "mediapipe":
        import_name = "mediapipe"
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"import error: {e}"


def audit_environment() -> dict:
    """
    Comprehensive environment audit.
    Returns dict with:
      - python_version: str
      - is_compatible: bool
      - broken_libs: list of {lib, reason, critical}
      - ok_libs: list of {lib, version}
      - recommendations: list of str
    """
    current     = get_current_python()
    ver_str     = f"{current[0]}.{current[1]}.{current[2]}"
    is_313_plus = current[:2] >= (3, 13)
    is_314_plus = current[:2] >= (3, 14)

    broken: List[dict] = []
    ok:     List[dict] = []
    recs:   List[str]  = []

    for lib, info in COMPATIBILITY_MATRIX.items():
        py_ok = is_python_compatible(info["min_py"], info["max_py"])
        importable, ver_or_err = check_library_importable(lib)

        if not py_ok:
            broken.append({
                "lib":      lib,
                "reason":   info["reason"],
                "critical": info["critical"],
                "install":  info["install"],
                "py_ok":    False,
            })
        elif not importable:
            broken.append({
                "lib":      lib,
                "reason":   f"Not installed: {ver_or_err}",
                "critical": info["critical"],
                "install":  info["install"],
                "py_ok":    True,
            })
        else:
            ok.append({"lib": lib, "version": ver_or_err})

    # Recommendations
    critical_broken = [b for b in broken if b["critical"] and not b["py_ok"]]
    if critical_broken and is_314_plus:
        recs.append(
            f"⚠️  CRITICAL: Python {ver_str} is incompatible with "
            f"{', '.join(b['lib'] for b in critical_broken)}. "
            f"Downgrade to Python {TARGET_PYTHON}."
        )

    if is_314_plus:
        recs.append(
            f"Run: brew install python@{TARGET_PYTHON}  "
            f"(or: pyenv install {TARGET_PYTHON})"
        )
        recs.append(
            f"Then: python{TARGET_PYTHON} -m venv ~/.venv_swayambhu && "
            f"source ~/.venv_swayambhu/bin/activate"
        )

    return {
        "python_version":     ver_str,
        "python_tuple":       current,
        "is_safe":            not is_314_plus,
        "is_py314_plus":      is_314_plus,
        "broken_libs":        broken,
        "ok_libs":            ok,
        "recommendations":    recs,
        "target_python":      TARGET_PYTHON,
    }


# ─────────────────────────────────────────────────────────────────────
# INSTALL GUIDE GENERATOR
# ─────────────────────────────────────────────────────────────────────

def generate_install_guide(audit: dict) -> str:
    """Generate a human-readable install guide from audit results."""
    lines = [
        "=" * 60,
        f"  🐍 SWAYAMBHU PYTHON ENVIRONMENT FIX",
        f"  Current Python: {audit['python_version']}",
        f"  Target Python:  {TARGET_PYTHON}",
        "=" * 60,
        "",
    ]

    if audit["is_safe"]:
        lines.append(f"  ✅ Python {audit['python_version']} is COMPATIBLE with all libraries.")
        lines.append("  No downgrade needed.\n")
    else:
        lines.append(f"  ❌ Python {audit['python_version']} has COMPATIBILITY ISSUES:\n")
        for b in audit["broken_libs"]:
            icon = "🔴" if b["critical"] else "🟡"
            lines.append(f"  {icon} {b['lib']}: {b['reason']}")
        lines.append("")

    if audit["recommendations"]:
        lines.append("  RECOMMENDED ACTIONS:")
        for rec in audit["recommendations"]:
            lines.append(f"  → {rec}")
        lines.append("")

    if audit["is_py314_plus"]:
        lines += [
            "  STEP-BY-STEP FIX (macOS / Homebrew):",
            "  " + "─" * 50,
            f"  1. Install Python {TARGET_PYTHON}:",
            f"     brew install python@{TARGET_PYTHON}",
            "",
            f"  2. Create a dedicated virtualenv:",
            f"     python{TARGET_PYTHON} -m venv ~/.venv_swayambhu",
            "",
            "  3. Activate it:",
            "     source ~/.venv_swayambhu/bin/activate",
            "",
            "  4. Install requirements:",
            "     pip install -r requirements_3_12.txt",
            "",
            f"  5. Verify: python --version   # should print {TARGET_PYTHON}.x",
            "",
            "  ALTERNATIVE (pyenv — recommended for multiple Python versions):",
            "  " + "─" * 50,
            f"  brew install pyenv",
            f"  pyenv install {TARGET_PYTHON}",
            f"  pyenv local {TARGET_PYTHON}",
            f"  python -m venv .venv && source .venv/bin/activate",
            "",
        ]

    if audit["ok_libs"]:
        lines.append("  ALREADY WORKING:")
        for lib in audit["ok_libs"]:
            lines.append(f"  ✅ {lib['lib']} ({lib['version']})")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# REQUIREMENTS FILE GENERATOR
# ─────────────────────────────────────────────────────────────────────

REQUIREMENTS_3_12 = """\
# =====================================================================
# requirements_3_12.txt  —  Pinned for Python 3.12 (AI sweet spot)
# Generated by python_env_fix.py
# Install: pip install -r requirements_3_12.txt
# =====================================================================

# ── Core API / Server ─────────────────────────────────────────────────
fastapi>=0.110,<1.0
uvicorn[standard]>=0.29
pydantic>=2.0,<3
requests>=2.31
nest_asyncio>=1.6
httpx>=0.27

# ── Local LLM (Metal GPU on Apple Silicon) ───────────────────────────
# Install with Metal support:
# CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
llama-cpp-python>=0.2.57

# ── MLX (Apple Silicon native — 3-10x faster than PyTorch on M-series)
mlx>=0.12
mlx-lm>=0.12

# ── Computer Vision + Gesture Tracking ───────────────────────────────
mediapipe>=0.10.14
opencv-python>=4.9
Pillow>=10.0

# ── Voice / Audio ─────────────────────────────────────────────────────
pyttsx3>=2.90
SpeechRecognition>=3.10
pyaudio>=0.2.14

# ── ML / Data ─────────────────────────────────────────────────────────
numpy>=1.26,<2.0
torch>=2.2        ; sys_platform == "darwin"

# ── System ────────────────────────────────────────────────────────────
psutil>=5.9
setproctitle>=1.3

# ── Optional: HuggingFace Hub (for model downloads) ──────────────────
huggingface-hub>=0.23

# ── Optional: Firebase (for cloud sync) ──────────────────────────────
# firebase-admin>=6.0

# ── Optional: ChromaDB (for episodic memory) ─────────────────────────
# chromadb>=0.4

# ── Optional: ElevenLabs (for TTS) ───────────────────────────────────
# elevenlabs>=1.0
"""


def write_requirements(path: Optional[Path] = None) -> Path:
    if path is None:
        try:
            root = Path(__file__).parent.resolve()
        except NameError:
            root = Path(os.getcwd()).resolve()
        path = root / "requirements_3_12.txt"

    path.write_text(REQUIREMENTS_3_12)
    return path


# ─────────────────────────────────────────────────────────────────────
# GESTURE TRACKER IMPORT GUARD PATCHER
#
# The original swayambhu_body_3.py has:
#     from gesture_tracker import GestureTracker, start_gesture_tracking
#     _GESTURE_OK = True
#   except ImportError:
#     _GESTURE_OK = False
#     print("⚠️  gesture_tracker not found — hand tracking disabled.")
#
# With Python 3.14, mediapipe fails to import silently.
# This patch adds a version check so the user gets a clear error.
# ─────────────────────────────────────────────────────────────────────

GESTURE_GUARD_PATCH = '''\
# ── FIX-1: Python version gate for gesture_tracker ────────────────────
import sys as _sys
if _sys.version_info >= (3, 14):
    _GESTURE_OK = False
    print(
        "⚠️  [GestureTracker] Python 3.14+ detected. mediapipe requires Python ≤3.12.\\n"
        "   Fix: brew install python@3.12 && python3.12 -m venv .venv && source .venv/bin/activate\\n"
        "   Then re-run: python swayambhu_v13.py"
    )
else:
    try:
        from gesture_tracker import GestureTracker, start_gesture_tracking
        _GESTURE_OK = True
    except ImportError:
        _GESTURE_OK = False
        print("⚠️  gesture_tracker not found — hand tracking disabled.")
'''


def patch_body_gesture_guard(body_path: Path) -> Tuple[bool, str]:
    """
    Patch swayambhu_body_3.py to add Python version gate for gesture_tracker.
    Returns (success, message).
    """
    if not body_path.exists():
        return False, f"File not found: {body_path}"

    content = body_path.read_text(encoding="utf-8")

    OLD_GUARD = (
        "try:\n"
        "    from gesture_tracker import GestureTracker, start_gesture_tracking\n"
        "    _GESTURE_OK = True\n"
        "except ImportError:\n"
        "    _GESTURE_OK = False\n"
        '    print("⚠️  gesture_tracker not found — hand tracking disabled.")'
    )

    if "FIX-1: Python version gate" in content:
        return True, "Already patched"

    if OLD_GUARD not in content:
        return False, "Old guard pattern not found — file may have been modified"

    # Write backup
    backup = body_path.with_suffix(".py.bak_fix1")
    backup.write_text(content, encoding="utf-8")

    # Apply patch
    patched = content.replace(OLD_GUARD, GESTURE_GUARD_PATCH.strip())
    body_path.write_text(patched, encoding="utf-8")

    return True, f"Patched successfully. Backup at {backup.name}"


# ─────────────────────────────────────────────────────────────────────
# SELF-TESTS
# ─────────────────────────────────────────────────────────────────────

def _run_tests():
    import tempfile, shutil
    print("🐍 PythonEnvFix Self-Tests\n")
    passed = failed = 0

    def ok(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed += 1

    current = get_current_python()

    # ── Test 1: Detection ─────────────────────────────────────────────
    print("=== Test 1: Python Version Detection ===")
    ok("get_current_python returns tuple",  isinstance(current, tuple))
    ok("Tuple has 3 elements",             len(current) == 3)
    ok("Major version is 3",               current[0] == 3)
    ok("Minor version is integer",         isinstance(current[1], int))
    ok("Version string readable",          f"{current[0]}.{current[1]}.{current[2]}")

    # ── Test 2: Compatibility matrix ────────────────────────────────
    print("\n=== Test 2: Compatibility Matrix ===")
    ok("mediapipe in matrix",       "mediapipe" in COMPATIBILITY_MATRIX)
    ok("llama-cpp-python in matrix","llama-cpp-python" in COMPATIBILITY_MATRIX)
    ok("mlx in matrix",             "mlx" in COMPATIBILITY_MATRIX)
    ok("All entries have min_py",   all("min_py" in v for v in COMPATIBILITY_MATRIX.values()))
    ok("All entries have max_py",   all("max_py" in v for v in COMPATIBILITY_MATRIX.values()))
    ok("All entries have install",  all("install" in v for v in COMPATIBILITY_MATRIX.values()))
    ok("All entries have critical", all("critical" in v for v in COMPATIBILITY_MATRIX.values()))

    # ── Test 3: is_python_compatible ──────────────────────────────────
    print("\n=== Test 3: is_python_compatible ===")
    ok("3.12 compatible with (3.8)→(3.14)",
       is_python_compatible((3, 8), (3, 14)))
    ok("3.14 NOT compatible with (3.8)→(3.13)",
       not is_python_compatible((3, 8), (3, 13)) or current[:2] < (3, 14))
    ok("Current Python: correct compat check",
       is_python_compatible((3, 0), (4, 0)))   # always true for 3.x
    ok("Incompatible range returns False",
       not is_python_compatible((3, 99), (3, 100)))

    # ── Test 4: check_library_importable ─────────────────────────────
    print("\n=== Test 4: check_library_importable ===")
    # json is always available
    importable, ver = check_library_importable("json")
    # json doesn't have __version__ but should be importable
    ok("json: importable",          importable, ver)

    # os is always available
    importable_os, ver_os = check_library_importable("os")
    ok("os: importable",            importable_os, ver_os)

    # fake library
    importable_fake, err_fake = check_library_importable("definitely_not_a_real_library_xyz123")
    ok("Fake lib: not importable",  not importable_fake)
    ok("Fake lib: has error msg",   len(err_fake) > 0)

    # ── Test 5: audit_environment ────────────────────────────────────
    print("\n=== Test 5: audit_environment ===")
    audit = audit_environment()
    ok("Returns dict",              isinstance(audit, dict))
    ok("Has python_version",        "python_version" in audit)
    ok("Has is_safe",               "is_safe" in audit)
    ok("Has broken_libs",           "broken_libs" in audit)
    ok("Has ok_libs",               "ok_libs" in audit)
    ok("Has recommendations",       "recommendations" in audit)
    ok("Has target_python",         "target_python" in audit)
    ok("Version string format",     re.match(r'\d+\.\d+\.\d+', audit["python_version"]))
    ok("Broken libs is list",       isinstance(audit["broken_libs"], list))
    ok("OK libs is list",           isinstance(audit["ok_libs"], list))

    # On current Python:
    if current[:2] >= (3, 14):
        ok("3.14+: is_safe=False",  not audit["is_safe"])
        ok("3.14+: has recs",       len(audit["recommendations"]) > 0)
    else:
        ok("3.12/3.13: is_safe=True", audit["is_safe"])

    # ── Test 6: generate_install_guide ────────────────────────────────
    print("\n=== Test 6: generate_install_guide ===")
    guide = generate_install_guide(audit)
    ok("Guide is string",           isinstance(guide, str))
    ok("Guide has content",         len(guide) > 100)
    ok("Guide mentions Python ver", audit["python_version"] in guide)
    ok("Guide mentions target",     TARGET_PYTHON in guide)

    if audit["is_py314_plus"]:
        ok("Guide has brew command",    "brew install" in guide)
        ok("Guide has venv command",    "venv" in guide)
        ok("Guide has requirements",    "requirements_3_12.txt" in guide)
    else:
        ok("Safe guide has compat msg", "COMPATIBLE" in guide)

    # ── Test 7: write_requirements ───────────────────────────────────
    print("\n=== Test 7: write_requirements ===")
    tmpdir = Path(tempfile.mkdtemp())
    req_path = tmpdir / "requirements_3_12.txt"
    written = write_requirements(req_path)
    ok("Requirements file written",    req_path.exists())
    ok("File non-empty",               req_path.stat().st_size > 0)
    content = req_path.read_text()
    ok("Has llama-cpp-python",         "llama-cpp-python" in content)
    ok("Has mediapipe",                "mediapipe" in content)
    ok("Has mlx",                      "mlx" in content)
    ok("Has fastapi",                  "fastapi" in content)
    ok("Has numpy",                    "numpy" in content)
    ok("Has version pins",             ">=" in content)
    ok("Has comments",                 "#" in content)

    # ── Test 8: GESTURE_GUARD_PATCH content ──────────────────────────
    print("\n=== Test 8: Gesture Guard Patch ===")
    ok("Patch has version check",      "3, 14" in GESTURE_GUARD_PATCH)
    ok("Patch has clear error msg",    "brew install python@3.12" in GESTURE_GUARD_PATCH)
    ok("Patch has fallback import",    "from gesture_tracker import" in GESTURE_GUARD_PATCH)
    ok("Patch has _GESTURE_OK=False",  "_GESTURE_OK = False" in GESTURE_GUARD_PATCH)

    # ── Test 9: patch_body_gesture_guard ─────────────────────────────
    print("\n=== Test 9: patch_body_gesture_guard ===")
    # Create a mock body file with the old guard
    mock_body = tmpdir / "mock_body.py"
    original_content = (
        "import sys\n"
        "# some code\n"
        "try:\n"
        "    from gesture_tracker import GestureTracker, start_gesture_tracking\n"
        "    _GESTURE_OK = True\n"
        "except ImportError:\n"
        "    _GESTURE_OK = False\n"
        '    print("⚠️  gesture_tracker not found — hand tracking disabled.")\n'
        "\n# more code\n"
    )
    mock_body.write_text(original_content)

    success, msg = patch_body_gesture_guard(mock_body)
    ok("Patch succeeds",               success, msg)
    ok("Patch message returned",       len(msg) > 0)

    patched_content = mock_body.read_text()
    ok("Patch added version check",    "3, 14" in patched_content or "FIX-1" in patched_content)
    ok("Backup was created",           mock_body.with_suffix(".py.bak_fix1").exists())

    # Idempotent — second patch should return 'Already patched'
    success2, msg2 = patch_body_gesture_guard(mock_body)
    ok("Idempotent — already patched", success2 and "Already" in msg2)

    # File not found case
    success3, msg3 = patch_body_gesture_guard(tmpdir / "nonexistent.py")
    ok("Missing file returns False",   not success3)
    ok("Missing file has error msg",   "not found" in msg3.lower())

    # ── Test 10: COMPATIBILITY_MATRIX completeness ───────────────────
    print("\n=== Test 10: Matrix Completeness ===")
    for lib, info in COMPATIBILITY_MATRIX.items():
        ok(f"{lib}: min_py < max_py",  info["min_py"] < info["max_py"])
        ok(f"{lib}: reason non-empty", len(info["reason"]) > 10)
        ok(f"{lib}: install non-empty",len(info["install"]) > 5)

    shutil.rmtree(tmpdir)

    print(f"\n{'='*55}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fix Python 3.14 dependency wall")
    parser.add_argument("--audit",     action="store_true", help="Show environment audit")
    parser.add_argument("--guide",     action="store_true", help="Show full install guide")
    parser.add_argument("--reqs",      action="store_true", help="Write requirements_3_12.txt")
    parser.add_argument("--patch",     metavar="PATH",      help="Patch a swayambhu_body*.py file")
    parser.add_argument("--test",      action="store_true", help="Run self-tests")
    args = parser.parse_args()

    if args.test or len(sys.argv) == 1:
        sys.exit(0 if _run_tests() else 1)

    audit = audit_environment()

    if args.audit:
        print(json.dumps(audit, indent=2, default=str))

    if args.guide or not any([args.audit, args.reqs, args.patch]):
        print(generate_install_guide(audit))

    if args.reqs:
        path = write_requirements()
        print(f"✅ Written: {path}")

    if args.patch:
        success, msg = patch_body_gesture_guard(Path(args.patch))
        print(f"{'✅' if success else '❌'} {msg}")
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
