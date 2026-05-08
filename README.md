# 🌌 Swayambhu: Sovereign Edge-to-Cloud AI Orchestration Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Edge_Server-009688?style=for-the-badge&logo=fastapi)
![Architecture](https://img.shields.io/badge/Architecture-Hybrid_Cloud%2FEdge-success?style=for-the-badge)
![Security](https://img.shields.io/badge/Security-AST_Execution_Shield-red?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-macOS-lightgrey?style=for-the-badge&logo=apple)
![LLM](https://img.shields.io/badge/LLM-70B_Cloud_%2B_1.5B_Local-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-🚧_Active_Development-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A deterministic, multi-agent AI operating system for macOS.**  
Hybrid local/cloud inference · AST security shield · Real-time 3D avatar · Voice control

[Architecture](#-architecture) · [Features](#-core-features) · [Quick Start](#-quick-start) · [File Reference](#-complete-file-reference) · [API Docs](#-api-reference) · [Security](#-security-model) · [Roadmap](#-roadmap)

</div>

---

> ## 🚧 Project Status: Active Development
>
> **Swayambhu is a work in progress.** The core text-prompt pipeline is functional end-to-end, but most of the advanced organ modules are still being built, integrated, and tested.
>
> **What works today (v14):**
> - ✅ Text prompt → Cloud Brain (70B LLM) → macOS action execution
> - ✅ 30+ built-in blueprints (OS control, apps, system info)
> - ✅ SecureShield AST execution gate
> - ✅ Local RAG index (TF-IDF + FAISS)
> - ✅ Air-gap survival mode with offline command queue
> - ✅ Firebase Firestore state sync + Brain URL auto-discovery
> - ✅ Particle Avatar WebGL UI with WebSocket state bridge
> - ✅ Edge FastAPI server (port 8003)
>
> **What is being actively developed:**
> - 🔧 Voice input pipeline (wake word → acoustic gate → STT → intent)
> - 🔧 Gesture recognition (webcam-based command input)
> - 🔧 On-device DPO fine-tuning flywheel (MLX + distillation factory)
> - 🔧 Affective/emotion engine integration with the command router
> - 🔧 S-LoRA multi-adapter routing for specialised task domains
> - 🔧 Meta-agent factory (dynamic specialised agent spawning)
>
> **Contributions, issues, and feedback are welcome.** If you are interested in helping build any of the modules listed above, please open an issue first so we can coordinate.

---

## Executive Summary

Swayambhu is a production-grade, multi-agent AI orchestration system that turns a Mac into a **sovereign AI workstation**. It intelligently bridges local edge computing with cloud-based inference:

- **Routine OS tasks** (screenshots, volume control, app launching) execute locally with zero latency via a library of 30+ pre-compiled blueprints
- **Complex reasoning** (code generation, multi-step planning) routes to a cloud-hosted 70B parameter LLM cluster via encrypted ngrok tunnels
- **All LLM output** passes through a strict `SecureShield` AST execution envelope before touching the OS — hallucinations cannot cause destructive actions
- **Survives network loss** via an air-gap mode that queues commands and flushes them on reconnect
- **Learns over time** via a nocturnal distillation flywheel that harvests execution logs, scores them with a 70B judge, and prepares DPO preference pairs for offline fine-tuning

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                         │
│   Voice (wake_detector → acoustic_gate → STT)                       │
│   Text prompt · Gesture (gesture_tracker)                           │
│   Particle Avatar WebGL UI (port 8007)                              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                   EDGE NODE  (Mac — port 8003)                      │
│                   swayambhu_body.py  v14.0                          │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Sovereign    │  │ Affective    │  │ Proactive Agency         │  │
│  │ Spine        │  │ Engine       │  │ (ambient task initiation)│  │
│  │ (router)     │  │ (emotion ctx)│  │                          │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────────┘  │
│         └─────────────────┤                                         │
│                    ┌──────▼────────┐                                │
│                    │ Neural        │                                 │
│                    │ Pipeline      │                                 │
│                    └──────┬────────┘                                │
│                           │                                         │
│         ┌─────────────────┴──────────────────┐                     │
│         │                                    │                      │
│  ┌──────▼──────┐                    ┌────────▼───────┐             │
│  │ Local Dual  │                    │ AirGap Survival│             │
│  │ LLM Engine  │                    │ Mode + Queue   │             │
│  │ Coder+Tester│                    └────────────────┘             │
│  └──────┬──────┘                                                    │
│         │                                                           │
│  ┌──────▼──────────────────────────────────────────────────────┐   │
│  │     SecureShield  +  Agent Shield Bridge                     │   │
│  │  (Regex gate · AST parse · Namespace isolation)              │   │
│  └──────┬──────────────────────────────────────────────────────┘   │
│         │                                                           │
│  ┌──────▼──────────────────────────────────────────────────────┐   │
│  │  Universal Action Space · Blueprint Engine · Open Claw      │   │
│  │  (AppleScript / Accessibility API / Zsh subprocessing)      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              MEMORY & LEARNING TIER                         │   │
│  │  Hippocampus · TacticalEdgeRAG · Memory Evolution           │   │
│  │  Distillation Factory · MLX DPO Trainer · S-LoRA Router     │   │
│  └─────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  HTTPS / ngrok encrypted tunnel
┌──────────────────────────▼──────────────────────────────────────────┐
│                  CLOUD BRAIN  (Kaggle — port 8000)                  │
│                                                                     │
│  Groq llama-3.3-70b-versatile · ElevenLabs TTS streaming           │
│  WebSocket token streaming · Firebase Firestore · Ngrok gateway     │
│  Web3 x-402 Invoice Settlement (Polygon USDC)                       │
└─────────────────────────────────────────────────────────────────────┘
                           │
                    Firebase Firestore
              (Brain URL · Blueprint sync · State)
```

**Command lifecycle (text prompt — current working path):**
```
Text Input
    → Edge FastAPI /command
    → sovereign_spine (local vs cloud decision)
          ├── LOCAL:  tactical_edge_rag → blueprint_engine → SecureShield → macOS
          └── CLOUD:  ngrok tunnel → 70B LLM → tool-call plan → SecureShield → macOS
    → hippocampus (memory write)
    → speak response (pyttsx3 / ElevenLabs)
    → particle_avatar state push (WebSocket)
```

---

## ✨ Core Features

### 1. Hybrid LLM Routing & Parallel Inference

The **Dual-Model Edge Engine** loads two local models simultaneously in thread-safe `llama-cpp-python` slots:

| Slot | Default Model | Role |
|---|---|---|
| `coder` | DeepSeek-Coder-V2-Lite-Instruct Q4_K_M (8.9 GB) | Code generation, blueprint creation |
| `tester` | Qwen2.5-1.5B-Instruct Q4_K_M (0.9 GB) | QA, test generation, output validation |
| `draft` | Llama-3.2-1B-Instruct Q4_K_M (0.6 GB) | Speculative decoding accelerator |

The **Maker/Checker** pattern fires both LLMs in concurrent threads. Total latency is `max(coder_time, tester_time)` — not the sum.

### 2. Deterministic Security (SecureShield + Agent Shield Bridge)

LLM output **never reaches the shell raw**. Every blueprint passes through:

```
LLM Output
    → security_shield.py  (regex pattern scan)
    → agent_shield_bridge.py  (cross-agent policy enforcement)
    → AST compile()
    → Namespace-isolated exec()
    → Result returned
```

Blocked patterns: `rm -rf`, `sudo rm`, `mkfs`, `dd if=`, `os.system()`, `eval()`, `exec()`, `__import__()`, `shell=True`.

### 3. Memory Architecture (Three Tiers)

| Tier | Module | Technology | Scope |
|---|---|---|---|
| Working | `neural_pipeline.py` | In-process dict | Current session |
| Episodic | `hippocampus.py` | SQLite3 + NumPy vectors | Cross-session recall |
| Semantic | `tactical_edge_rag.py` | FAISS / TF-IDF | Blueprint knowledge base |
| Evolutionary | `memory_evolution.py` | Scored preference pairs | DPO fine-tune feed |

### 4. On-Device Learning Flywheel *(In Development)*

```
execution_log → distillation_factory.py → 70B judge score
                                        → DPO preference pairs
                                        → mix_dpo_trainer.py (MLX, Apple Silicon)
                                        → s_lora_router.py (hot-swap adapter)
                                        → improved local model — no cloud required
```

### 5. Blueprint System (30+ Built-in Skills)

| Category | Blueprints |
|---|---|
| Apps | `open_safari`, `open_chrome`, `open_terminal`, `open_vscode`, `open_finder`, `open_calendar`, `quit_app` |
| System | `get_battery`, `get_disk_space`, `get_ip`, `lock_screen`, `sleep_display`, `take_screenshot` |
| Audio | `adjust_volume`, `mute_volume`, `play_music`, `pause_music`, `next_track` |
| Clipboard | `get_clipboard`, `set_clipboard` |
| Input | `type_text`, `press_key` |
| Web | `open_url`, `web_search` |
| Files | `list_workspace`, `create_folder` |
| Productivity | `set_reminder`, `empty_trash` |

### 6. 4K Particle Avatar (WebGL + FFT Audio-Reactive)

A browser-based 3D particle sphere with 4,000 particles. Audio-reactive: Web Audio API samples microphone FFT at 60fps and injects bass amplitude into the particle physics engine in real-time.

| State | Color | Visual |
|---|---|---|
| `idle` | Blue | Gentle slow pulse |
| `listening` | Green | Energised, responsive |
| `processing` | Purple | Rapid shimmer |
| `speaking` | Cyan | FFT audio burst |
| `error` | Red-orange | Aggressive pulse |
| `defcon` | Deep red | Maximum shatter |

---

## 📁 Complete File Reference

> **Legend:**
> - ✅ Active and running in the current v14.0 build
> - 🚧 Implemented but not yet fully integrated into the main boot sequence
> - 🔬 Experimental / under active development

### Core System

| File | Status | Description |
|---|---|---|
| `Autonomous-AI-System.ipynb` | ✅ | **Cloud Brain** — Kaggle notebook hosting the 70B Groq LLM, ElevenLabs TTS streaming, Firebase sync, and ngrok gateway |
| `swayambhu_body.py` | ✅ | **Central Nervous System** — `EdgeNodeOrchestrator` v14.0, all organ mounts, boot sequence, FastAPI edge server (port 8003), 15-test self-test suite |
| `particle_avatar.py` | 🚧| **4K WebGL Particle Avatar** — `ParticleAvatarServer` (port 8007), FFT audio-reactive physics, WebSocket state bridge |
| `audit.py` | ✅ | **Deep Architecture Auditor** — AST import analysis, endpoint cross-reference, dead-call detection, missing dependency scanner |
| `swayambhu_utils.py` | ✅ | Shared constants, `PROJECT_ROOT` resolution, path helpers used across all modules |
| `python_env_fix.py` | 🚧 | Environment diagnostic and repair script — fixes common `llama-cpp-python` / Metal / MPS build issues on Apple Silicon |

### Intelligence & Reasoning

| File | Status | Description |
|---|---|---|
| `sovereign_spine.py` | ✅ | **Confidence Router** — decides local vs cloud inference per command based on complexity scoring, local model readiness, and DEFCON level |
| `neural_pipeline.py` | ✅ | **Intent Extraction Pipeline** — NLP preprocessing, entity recognition, command normalisation before routing |
| `reasoning_engine.py` | 🔬 | Chain-of-thought reasoning scaffolding for multi-step task decomposition |
| `dual_model_engine.py` | ✅ | **Parallel Inference Engine** — thread-safe Maker/Checker LLM slots with `wait_loaded()` barrier and capability flag reporting |
| `speculative_engine.py` | 🚧 | **Speculative Decoding** — wires a 0.6B draft model into the primary slot via `LlamaDraftModel` for 30–50% latency reduction |
| `s_lora_router.py` | 🔬 | **S-LoRA Adapter Router** — hot-swaps fine-tuned LoRA adapters per task domain without reloading the base model |
| `meta_agent_factory.py` | 🔬 | Dynamic agent spawning — creates specialised sub-agents for complex multi-domain tasks at runtime |

### Memory & Learning

| File | Status | Description |
|---|---|---|
| `hippocampus.py` | ✅ | **Long-term Memory** — SQLite3 + NumPy vector store for cross-session episodic memory. Zero external vector DB dependencies |
| `tactical_edge_rag.py` | ✅ | **Tactical RAG** — FAISS-accelerated blueprint knowledge retrieval with cloud shadow-sync |
| `memory_evolution.py` | 🔬 | Tracks memory quality scores over time, prunes low-value entries, promotes high-value ones to persistent storage |
| `distillation_factory.py` | 🔬 | **Nocturnal Distillation Flywheel** — harvests failed execution logs, scores them with a 70B judge, generates DPO preference pairs |
| `mix_dpo_trainer.py` | 🔬 | **MLX DPO Trainer** — on-device Direct Preference Optimisation fine-tuning using Apple Silicon MLX framework |
| `test_mlx_dpo_trainer.py` | 🔬 | Test suite and benchmarks for the MLX DPO training pipeline |

### Security

| File | Status | Description |
|---|---|---|
| `security_shield.py` | 🚧 | **Primary Execution Gate** — regex + AST shield blocking destructive patterns before any code touches the OS |
| `agent_shield_bridge.py` | 🚧 | **Cross-Agent Policy Enforcement** — validates tool calls between agents, prevents privilege escalation across agent boundaries |
| `lizard_brain.py` | ✅ | **Threat Detection + Self-Patching** — monitors system behaviour for anomalies, can trigger `DeadMansSwitch` and apply hot patches |
| `tap_adversarial.py` | 🔬 | **Red-Team / Adversarial Testing** — automated prompt injection and jailbreak attempt suite for hardening the execution pipeline |

### Execution & OS Integration

| File | Status | Description |
|---|---|---|
| `blueprint_engine.py` | ✅ | **Blueprint Runtime** — loading, versioning, delta-sync, and validated execution of all OS skill blueprints |
| `universal_action_space.py` | ✅ | **macOS Action Bridge** — translates AI intent into Accessibility API calls, AppleScript, and Zsh scripts |
| `openclaw.py` | ✅ | **General-Purpose Claw** — `OpenClawGeneral` for unstructured OS interactions not covered by named blueprints |
| `software_firm.py` | 🚧 | **Multi-Agent Coding Pipeline** — Manager/Coder/Tester agent triad for end-to-end code generation, review, and test execution |
| `kinematic_fsm.py` | 🔬 | **Kinematic State Machine** — deterministic FSM for complex multi-step physical actions (drag, scroll, form fill sequences) |
| `proactive_agency.py` | ✅ | **Ambient Task Initiation** — monitors system context (calendar, clipboard, active app) and proactively suggests or executes actions |

### Perception & Input

| File | Status | Description |
|---|---|---|
| `wake_detector.py` | 🚧 | **Hotword Detection** — always-on lightweight model listening for the "Swayambhu" trigger word |
| `acoustic_gate.py` | 🚧 | **Acoustic Preprocessor** — noise suppression, voice activity detection (VAD), and audio normalisation before STT |
| `gesture_tracker.py` | 🔬 | **Webcam Gesture Input** — MediaPipe-based hand gesture recognition mapped to the command vocabulary |

### Affective & Emotional Intelligence

| File | Status | Description |
|---|---|---|
| `affective_engine.py` | ✅ | **Emotion State Model** — maintains a valence/arousal state that modulates response style, verbosity, and tool selection |
| `empathy_wire.py` | ✅ | **Biometric Monitor** — reads BPM from connected heart rate sensor; injects stress-override prefix into cloud prompts when BPM > 115 |

---

## 🚀 Quick Start

### Prerequisites

- macOS 13+ (Ventura or later recommended)
- Python 3.10+
- 8 GB minimum
- Kaggle account (for Cloud Brain)
- Firebase project with Firestore enabled
- Ngrok account (free tier works)
- Groq API key (free tier: 30 req/min on 70B)

### 1. Clone & Install

```bash
git clone https://github.com/your-username/swayambhu.git
cd swayambhu

pip install fastapi uvicorn groq firebase-admin pyngrok \
            nest_asyncio httpx requests numpy pyttsx3 \
            SpeechRecognition llama-cpp-python
```

Optional accelerators:
```bash
pip install faiss-cpu       # RAG acceleration
pip install setproctitle    # process stealth
pip install mlx mlx-lm      # on-device DPO training (Apple Silicon only)
pip install mediapipe       # gesture tracking
```

### 2. Download Local Models

```bash
mkdir -p ~/Swayambhu/models
python -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF',
                'qwen2.5-1.5b-instruct-q4_k_m.gguf',
                local_dir=os.path.expanduser('~/Swayambhu/models'))
"
```

Alternatively, `MultiModelManifest` in `swayambhu_body.py` will auto-download missing models on first boot (requires ~10 GB free disk).

### 3. Configure Secrets

```bash
export GROQ_API_KEY="gsk_..."
export NGROK_TOKEN="your_ngrok_authtoken"
export ELEVENLABS_API_KEY="your_key"   # optional, enables cloud TTS
# Place firebase_key.json in the project root
```

### 4. Start the Cloud Brain (Kaggle)

Open `Autonomous-AI-System.ipynb` in Kaggle and add secrets via the Kaggle Secrets UI:

| Secret | Value |
|---|---|
| `GROQ_API_KEY` | Your Groq key |
| `NGROK_TOKEN` | Your ngrok token |
| `ELEVENLABS_API_KEY` | Your ElevenLabs key |
| `FIREBASE_B64` | Output of `base64 -i firebase_key.json` |

Run all cells. You should see:
```
🚀 SWAYAMBHU CLOUD ONLINE AT: https://xxxx.ngrok-free.dev
🔥 Firebase updated with new Brain URL.
```

### 5. Start the Edge Node

```bash
python swayambhu_body.py
```

Expected output:
```
🌌 ═══ SWAYAMBHU EDGE NODE v14.0 BOOTING ═══
✅ [EdgeServer] Listening on port 8003
✨ [ParticleAvatar] Serving on http://localhost:8007
🌌 ═══ BOOT COMPLETE ═══
    ✅ firebase
    ✅ tactical_rag
    ✅ local_coder
    ✅ local_tester
```

The avatar opens in your browser automatically. **Type commands in the text box** — this is the current working input method. Voice input is in development.

### 6. (Optional) Install as macOS Daemon

```python
from swayambhu_body import LaunchAgentDaemon
LaunchAgentDaemon().daemonize()   # auto-starts on login
LaunchAgentDaemon().uninstall()   # remove
```

---

## 🔌 API Reference

### Edge Node — `http://localhost:8003`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/command` | Route a text command through the full orchestration pipeline |
| `GET` | `/health` | Full status: version, capabilities, DEFCON level, RAM, queue depth |
| `GET` | `/blueprints` | List all loaded blueprint IDs and count |
| `POST` | `/blueprint/execute` | Directly execute a named blueprint by ID |

**POST /command example:**
```json
// Request
{ "command": "take a screenshot and open it in Preview" }

// Response
{
  "message": "Screenshot saved to ~/Desktop/shot_1234567890.png",
  "plan": [{"action": "actuate", "params": {"script": "..."}}],
  "state": "idle"
}
```

### Cloud Brain — `https://xxxx.ngrok-free.dev`

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/command` | 70B reasoning + tool-call plan generation |
| `GET` | `/health` | Gateway liveness check |
| `POST` | `/sandbox/evaluate` | B2B adversarial red-team + x-402 USDC invoice |
| `WS` | `/ws_stream` | Token streaming + real-time TTS audio chunks |

### Particle Avatar — `http://localhost:8007`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve WebGL avatar HTML (cache-busted on every request) |
| `GET` | `/health` | Avatar server status + active WebSocket connection count |
| `WS` | `/ws/avatar` | Push state / emotion / text changes to the particle sphere |

---

## 🔒 Security Model

Five execution layers ensure LLM output never directly invokes system calls:

| Layer | Module | Mechanism |
|---|---|---|
| 1 | `security_shield.py` | Regex scan blocks 10 destructive patterns |
| 2 | `agent_shield_bridge.py` | Cross-agent policy enforcement, privilege scope |
| 3 | AST compilation | `compile()` validates syntax before any execution |
| 4 | Namespace isolation | `exec()` runs in `{"__builtins__": ...}` only |
| 5 | `DeadMansSwitch` | Physical Wi-Fi severance (`networksetup -setairportpower en0 off`) on critical breach |

DEFCON levels (1–5) gate cloud access. At DEFCON 1 all external communication is suspended and the system runs entirely offline.

---

## 🧪 Running Tests

```bash
# Full self-test suite (15 tests, no extra dependencies needed)
python swayambhu_body.py --test

# Architecture audit (outputs deep_audit_report.json)
python audit.py

# MLX DPO trainer tests
python test_mlx_dpo_trainer.py
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SWAYAMBHU_MODEL_DIR` | `~/Swayambhu/models` | Path to GGUF model directory |
| `SWAYAMBHU_EDGE_PORT` | `8003` | Edge FastAPI server port |
| `AVATAR_PORT` | `8007` | Particle avatar server port |
| `SWAYAMBHU_HEARTBEAT` | `30` | Edge→Cloud heartbeat interval (seconds) |
| `SWAYAMBHU_FIREBASE_DB` | _(project id)_ | Firebase database ID override |
| `BRAIN_URL` | _(auto-discovered)_ | Manual override for ngrok Brain URL |
| `GROQ_API_KEY` | — | Required for cloud inference |
| `NGROK_TOKEN` | — | Required for cloud tunnel |
| `ELEVENLABS_API_KEY` | — | Optional: cloud TTS voice synthesis |
| `ELEVENLABS_VOICE_ID` | Rachel (default) | ElevenLabs voice ID |

---

## 🗺️ Roadmap

### v14.x — Current Sprint

- [ ] Fully integrate `wake_detector.py` + `acoustic_gate.py` into the live voice pipeline
- [ ] Connect `affective_engine.py` to the command router for tone modulation
- [ ] Complete `blueprint_engine.py` versioning and cloud delta-sync
- [ ] Activate `kinematic_fsm.py` for multi-step UI automation sequences
- [ ] Stabilise `sovereign_spine.py` confidence scoring algorithm

### v15.0 — Learning & Adaptation

- [ ] End-to-end nocturnal distillation flywheel (`distillation_factory` → `mix_dpo_trainer`)
- [ ] S-LoRA adapter hot-swap for domain-specialised inference without base model reload
- [ ] `memory_evolution.py` quality scoring and automatic low-value entry pruning

### v16.0 — Perception & Proactivity

- [ ] Gesture command input via `gesture_tracker.py` (MediaPipe)
- [ ] `proactive_agency.py` ambient context monitoring (calendar, clipboard, active app)
- [ ] `meta_agent_factory.py` dynamic specialised agent spawning at runtime
- [ ] Prompt injection hardening via `tap_adversarial.py` automated red-team suite

### v17.0 — Multi-Node & Mobile

- [ ] Multi-Mac federation (multiple edge nodes sharing one Cloud Brain)
- [ ] Android/iOS companion app via the Edge REST API
- [ ] Live Polygon USDC settlement for B2B API consumption

---

## 🤝 Contributing

Pull requests are welcome. For major changes please open an issue first to coordinate.

**Good first issues** (no deep ML knowledge required):
- Add new blueprints to `BUILTIN_BLUEPRINTS` in `swayambhu_body.py`
- Write tests for any of the 🚧 modules
- Improve logging and error messages across any module

**Looking for collaborators on:**
- `distillation_factory.py` + `mix_dpo_trainer.py` — on-device DPO pipeline
- `acoustic_gate.py` — noise suppression and VAD
- `kinematic_fsm.py` — multi-step UI automation state machine
- `proactive_agency.py` — ambient context monitoring

When adding a new blueprint follow this pattern:
```python
"my_blueprint": (
    'import subprocess\ndef run(**kw):\n'
    '    # your code — no shell=True, no eval, no exec\n'
    '    return {"status": "OK"}\n'
),
```

All blueprints must: define `run(**kwargs)`, avoid `shell=True`, and return a dict.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by a Sovereign Architect.**  
*"The machine should serve the mind, not the other way around."*

⭐ Star this repo if you find it interesting — it helps others discover the project.

</div>
