#!/usr/bin/env python3
# =====================================================================
# ⚡ NEURAL PIPELINE  v14.0  —  SHARED CORE + BODY-SIDE ADDITIONS
#
# DEPLOYMENT MAP
# ─────────────────────────────────────────────────────────────────────
# Both sides (import freely):
#   ElevenLabsTTS            — async TTS engine, key read from env
#   LocalTTSFallback         — macOS `say` command; zero-dep offline TTS
#   WebSocketStreamHandler   — token-streaming loop + TTS flush
#   SENTENCE_BOUNDARIES      — compiled regex for sentence splits
#
# Kaggle brain (import in notebook Master Brain cell):
#   attach_ws_stream_endpoint — mounts /ws_stream on the brain FastAPI app
#   build_kaggle_app          — standalone brain FastAPI factory
#   _mock_llm_generator       — fake token stream for testing only
#
# Mac body (import in body FastAPI app):
#   OllamaStreamGenerator    — streams tokens from Ollama /api/generate
#   BodyWebSocketClient      — upstream WS client: relays brain tokens to
#                              local HTML UI and plays audio locally
#   attach_body_ws_endpoint  — mounts /ws_stream on the body FastAPI app,
#                              wired to Ollama or brain relay
#
# v14.0 changes vs v13.2 / current file:
#   • LocalTTSFallback added  — macOS say(1) offline TTS, no API key
#   • OllamaStreamGenerator   — body-side async token generator
#   • BodyWebSocketClient     — upstream relay + local audio playback
#   • attach_body_ws_endpoint — body FastAPI mount helper
#   • ElevenLabsTTS.synthesize_with_fallback() — tries ElevenLabs then
#     falls back to LocalTTSFallback automatically
#   • WebSocketStreamHandler gains inject_generator() for hot-swap
#   • All Kaggle-only helpers preserved unchanged
#   • Self-test extended to 12 test groups, 0 network calls required
# =====================================================================

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import AsyncGenerator, Callable, Dict, List, Optional

logger = logging.getLogger("NeuralPipeline")

# ── Optional deps ──────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False
    logger.warning("FastAPI not available — WebSocket pipeline disabled.")

try:
    import httpx
    _HTTPX_OK = True
except ImportError:
    _HTTPX_OK = False

try:
    import websockets as _websockets_lib
    _WS_CLIENT_OK = True
except ImportError:
    _WS_CLIENT_OK = False

# ── Config ─────────────────────────────────────────────────────────────
ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL    = "eleven_turbo_v2"

OLLAMA_BASE_URL     = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL        = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT      = 60          # generous: streaming can be slow on CPU

# Sentence boundary regex — shared by both sides
SENTENCE_BOUNDARIES = re.compile(r'(?<=[.!?\n])\s*')


# ======================================================================
# § 1  ELEVEN LABS TTS ENGINE  (shared — both Kaggle and Mac body)
# ======================================================================
class ElevenLabsTTS:
    """
    Async TTS engine using the ElevenLabs eleven_turbo_v2 streaming
    endpoint.  Sub-200ms first-byte target via optimize_streaming_latency=4.

    Returns raw MP3 bytes.  Caller sends {"type":"audio_start"} then bytes.
    If API key is not set, synthesize_bytes() returns None immediately.
    synthesize_with_fallback() tries ElevenLabs then falls through to
    LocalTTSFallback when the key is absent or the call fails.
    """

    def __init__(
        self,
        api_key:  str = ELEVENLABS_API_KEY,
        voice_id: str = ELEVENLABS_VOICE_ID,
        model:    str = ELEVENLABS_MODEL,
        fallback: Optional["LocalTTSFallback"] = None,
    ):
        self._key      = api_key
        self._voice_id = voice_id
        self._model    = model
        self._url      = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
        )
        self._fallback = fallback
        # Perf counters
        self._call_count    = 0
        self._success_count = 0
        self._total_ms      = 0.0

    @property
    def is_configured(self) -> bool:
        return bool(self._key)

    async def synthesize_bytes(self, text: str) -> Optional[bytes]:
        """
        Pure ElevenLabs call.  Returns MP3 bytes or None.
        Does NOT touch the fallback — call synthesize_with_fallback()
        if you want automatic offline degradation.
        """
        if not self.is_configured:
            logger.debug("[TTS] API key not set — skipping ElevenLabs.")
            return None
        if not text.strip():
            return None
        if not _HTTPX_OK:
            logger.warning("[TTS] httpx not installed — cannot call ElevenLabs.")
            return None

        t0 = time.perf_counter()
        self._call_count += 1
        headers = {
            "xi-api-key":   self._key,
            "Content-Type": "application/json",
            "Accept":       "audio/mpeg",
        }
        body = {
            "text":       text.strip(),
            "model_id":   self._model,
            "voice_settings": {
                "stability":        0.45,
                "similarity_boost": 0.80,
                "style":            0.10,
                "use_speaker_boost": True,
            },
            "optimize_streaming_latency": 4,
        }
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                async with client.stream(
                    "POST", self._url, headers=headers, json=body
                ) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        logger.error(
                            f"[TTS] ElevenLabs {resp.status_code}: {err[:200]}"
                        )
                        return None
                    chunks: list[bytes] = []
                    async for chunk in resp.aiter_bytes(chunk_size=4096):
                        chunks.append(chunk)
                    mp3 = b"".join(chunks)

            elapsed = (time.perf_counter() - t0) * 1000
            self._success_count += 1
            self._total_ms      += elapsed
            logger.info(f"[TTS] {len(text)} chars → {len(mp3)} bytes in {elapsed:.0f}ms")
            return mp3

        except httpx.TimeoutException:
            logger.error("[TTS] ElevenLabs request timed out.")
            return None
        except Exception as e:
            logger.error(f"[TTS] ElevenLabs error: {e}")
            return None

    async def synthesize_with_fallback(
        self,
        text:     str,
        fallback: Optional["LocalTTSFallback"] = None,
    ) -> Optional[bytes]:
        """
        Tries ElevenLabs first.  On None (no key / error) falls through to:
          1. self._fallback (if set at construction time)
          2. the fallback arg passed here
          3. returns None (caller handles silence gracefully)
        """
        mp3 = await self.synthesize_bytes(text)
        if mp3 is not None:
            return mp3

        fb = fallback or self._fallback
        if fb:
            return await fb.speak_async(text)
        return None

    def get_stats(self) -> dict:
        avg = (self._total_ms / self._success_count) if self._success_count else 0.0
        return {
            "configured":    self.is_configured,
            "model":         self._model,
            "voice_id":      self._voice_id,
            "calls":         self._call_count,
            "successes":     self._success_count,
            "avg_latency_ms": round(avg, 1),
        }


# ======================================================================
# § 2  LOCAL TTS FALLBACK  (body-side only — uses macOS `say` command)
# ======================================================================
class LocalTTSFallback:
    """
    Zero-dependency offline TTS using the macOS built-in `say` command.

    On non-Mac systems (Linux CI, Kaggle) speak() is a silent no-op so
    the rest of the pipeline keeps working without any changes.

    speak()       — blocking call, returns when audio finishes playing
    speak_async() — awaitable coroutine, runs say in executor, returns
                    None (no MP3 bytes; audio plays on the Mac directly)
    speak_to_file() — renders to a temp AIFF file and returns the path
                      (useful when you need the audio bytes separately)

    Supported voices: any name from `say -v ?` on macOS.
    Default: "Samantha" (US English, sounds natural at high rate).
    """

    _IS_MACOS = (os.uname().sysname == "Darwin") if hasattr(os, "uname") else False

    def __init__(
        self,
        voice: str  = "Samantha",
        rate:  int  = 200,        # words per minute; 200 = natural pace
    ):
        self._voice   = voice
        self._rate    = rate
        self._lock    = threading.Lock()
        self._call_count = 0

    @property
    def is_available(self) -> bool:
        if not self._IS_MACOS:
            return False
        try:
            result = subprocess.run(
                ["say", "--version"],
                capture_output=True, timeout=2
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def speak(self, text: str) -> bool:
        """
        Blocking TTS via `say`.  Returns True on success, False on error
        or non-Mac.  Thread-safe (one utterance at a time).
        """
        if not self._IS_MACOS or not text.strip():
            return False
        with self._lock:
            self._call_count += 1
            try:
                subprocess.run(
                    ["say", "-v", self._voice, "-r", str(self._rate), text.strip()],
                    timeout=30,
                    check=True,
                )
                return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    FileNotFoundError) as e:
                logger.warning(f"[LocalTTS] say failed: {e}")
                return False

    async def speak_async(self, text: str) -> None:
        """
        Awaitable wrapper — runs speak() in the default executor so it
        does not block the asyncio event loop.
        Returns None (audio plays on Mac speakers directly, no MP3 bytes).
        """
        if not text.strip():
            return None
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.speak, text)
        return None

    def speak_to_file(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Render speech to an AIFF file and return the file path.
        Uses a temp file if output_path is not given.
        Returns None on error or non-Mac.
        """
        if not self._IS_MACOS or not text.strip():
            return None
        path = output_path or tempfile.mktemp(suffix=".aiff")
        try:
            subprocess.run(
                ["say", "-v", self._voice, "-r", str(self._rate),
                 "-o", path, text.strip()],
                timeout=30,
                check=True,
            )
            return path
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError) as e:
            logger.warning(f"[LocalTTS] say to file failed: {e}")
            return None

    def get_stats(self) -> dict:
        return {
            "available":   self.is_available,
            "voice":       self._voice,
            "rate_wpm":    self._rate,
            "call_count":  self._call_count,
            "platform":    "macOS" if self._IS_MACOS else "non-macOS (silent no-op)",
        }


# ======================================================================
# § 3  WEBSOCKET STREAM HANDLER  (shared — environment-agnostic)
#
#  The generator injected determines which side this runs on:
#    Brain: inject _llm_stream_groq (Kaggle)
#    Body:  inject OllamaStreamGenerator.generate (Mac)
# ======================================================================
class WebSocketStreamHandler:
    """
    Token-streaming loop over a FastAPI WebSocket connection.

    Protocol (client receives):
      {"type":"token",       "text":"<token>"}          — every LLM token
      {"type":"audio_start", "text":"<sentence>",
       "bytes_length":<N>}                              — before MP3 bytes
      <raw bytes>                                       — MP3 audio
      {"type":"done",        "total_tokens":<N>}        — stream finished
      {"type":"error",       "message":"<msg>"}         — on failure

    Generator contract:
      async def generate(prompt: str, context: dict) -> AsyncGenerator[str, None]
      Each yielded str is one token (may be a single character or a word,
      depending on the backend).
    """

    def __init__(
        self,
        tts:              ElevenLabsTTS,
        llm_generate_fn:  Optional[Callable] = None,
        local_tts:        Optional["LocalTTSFallback"] = None,
    ):
        self._tts          = tts
        self._llm_generate = llm_generate_fn
        self._local_tts    = local_tts
        self._token_count  = 0
        self._stream_count = 0

    def inject_generator(self, fn: Callable) -> None:
        """Hot-swap the LLM generator without recreating the handler."""
        self._llm_generate = fn

    def _is_sentence_boundary(self, text: str) -> bool:
        if not text:
            return False
        if text.endswith("\n"):
            return True
        stripped = text.rstrip()
        return bool(stripped) and stripped[-1] in ".!?"

    async def _flush_tts(self, ws: "WebSocket", sentence: str) -> bool:
        """
        Tries ElevenLabs → falls back to LocalTTS → sends audio_start
        + MP3 bytes if we got bytes, otherwise skips the audio frame.
        """
        sentence = sentence.strip()
        if not sentence:
            return False

        mp3 = await self._tts.synthesize_with_fallback(sentence, self._local_tts)

        if mp3 is None:
            # LocalTTSFallback played audio directly on Mac speakers — no bytes
            # to send over WS.  That is fine: text tokens already streamed.
            return True

        try:
            await ws.send_json({
                "type":         "audio_start",
                "text":         sentence,
                "bytes_length": len(mp3),
            })
            await ws.send_bytes(mp3)
            return True
        except Exception as e:
            logger.warning(f"[WS] send audio failed: {e}")
            return False

    async def handle_stream(
        self,
        ws:      "WebSocket",
        prompt:  str,
        context: Optional[dict] = None,
    ) -> None:
        if self._llm_generate is None:
            await ws.send_json({"type": "error", "message": "No LLM configured."})
            return

        self._stream_count += 1
        sentence_buf = ""
        total_tokens = 0

        try:
            async for token in self._llm_generate(prompt, context or {}):
                total_tokens += 1
                self._token_count += 1

                await ws.send_json({"type": "token", "text": token})
                sentence_buf += token

                if self._is_sentence_boundary(sentence_buf):
                    segments = SENTENCE_BOUNDARIES.split(sentence_buf)
                    for seg in segments[:-1]:
                        if seg.strip():
                            asyncio.create_task(self._flush_tts(ws, seg))
                    sentence_buf = segments[-1] if segments else ""

            # Flush remainder
            if sentence_buf.strip():
                await self._flush_tts(ws, sentence_buf)

            await ws.send_json({"type": "done", "total_tokens": total_tokens})

        except Exception as exc:
            # Catch WebSocketDisconnect by name to avoid hard import dep
            if type(exc).__name__ == "WebSocketDisconnect":
                logger.info("[WS] Client disconnected during stream.")
            else:
                logger.error(f"[WS] Stream error: {exc}")
                try:
                    await ws.send_json({"type": "error", "message": str(exc)})
                except Exception:
                    pass

    def get_stats(self) -> dict:
        return {
            "stream_count": self._stream_count,
            "token_count":  self._token_count,
            "tts_stats":    self._tts.get_stats(),
        }


# ======================================================================
# § 4  OLLAMA STREAM GENERATOR  (body-side — Mac only)
#
#  Streams tokens from Ollama's /api/generate endpoint line by line.
#  Each line is a JSON object {"response":"<token>", "done":false|true}.
#  Yields one token string per line until done==true.
#
#  Falls back to non-streaming /api/generate if the streaming parse
#  fails (e.g. older Ollama versions that don't support stream=true).
# ======================================================================
class OllamaStreamGenerator:
    """
    Body-side async token generator backed by Ollama.

    Wired into WebSocketStreamHandler so the body's /ws_stream serves
    locally-generated responses without any cloud dependency.

    Also used by BodyWebSocketClient.generate_local() to answer prompts
    when the brain's upstream WS is unreachable (DEFCON-1 / air-gap).
    """

    def __init__(
        self,
        ollama_url:   str = OLLAMA_BASE_URL,
        model:        str = OLLAMA_MODEL,
        system:       str = (
            "You are Swayambhu, a sovereign offline AI. "
            "Respond helpfully and concisely."
        ),
        temperature:  float = 0.7,
        num_predict:  int   = 500,
    ):
        self._url         = ollama_url.rstrip("/")
        self._model       = model
        self._system      = system
        self._temperature = temperature
        self._num_predict = num_predict
        self._call_count  = 0

    @property
    def endpoint(self) -> str:
        return f"{self._url}/api/generate"

    async def generate(
        self,
        prompt:  str,
        context: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator: yields one token string at a time.

        context dict keys honoured:
          sys_override  — "USER_STRESSED_BE_CONCISE" → appends concise instruction
          system        — override system prompt entirely
          episodic_memory — list[str] prepended to system prompt
        """
        if not _HTTPX_OK:
            yield "[Ollama unavailable — httpx not installed]"
            return

        ctx    = context or {}
        system = ctx.get("system", self._system)

        if ctx.get("sys_override") == "USER_STRESSED_BE_CONCISE":
            system += " Keep your response to 2 sentences maximum."

        memories: list = ctx.get("episodic_memory", [])
        if memories:
            mem_block = "\n".join(f"- {m}" for m in memories[:5])
            system    = f"{system}\n\n[Memory context]\n{mem_block}"

        body = {
            "model":  self._model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._num_predict,
            },
        }

        self._call_count += 1
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                async with client.stream(
                    "POST", self.endpoint, json=body
                ) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        yield f"[Ollama error {resp.status_code}: {err[:120].decode(errors='replace')}]"
                        return

                    async for raw_line in resp.aiter_lines():
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        try:
                            obj = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue

                        token = obj.get("response", "")
                        if token:
                            yield token

                        if obj.get("done", False):
                            break

        except httpx.ConnectError:
            yield (
                f"⚠️ Ollama not running at {self._url}. "
                f"Start with: ollama run {self._model}"
            )
        except httpx.TimeoutException:
            yield f"[Ollama timeout after {OLLAMA_TIMEOUT}s]"
        except Exception as e:
            yield f"[Ollama stream error: {e}]"

    async def ping(self) -> bool:
        """Returns True if Ollama is reachable and the model is loaded."""
        if not _HTTPX_OK:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self._url}/")
                return r.status_code == 200
        except Exception:
            return False

    def get_stats(self) -> dict:
        return {
            "url":         self._url,
            "model":       self._model,
            "call_count":  self._call_count,
            "temperature": self._temperature,
            "num_predict": self._num_predict,
        }


# ======================================================================
# § 5  BODY WEBSOCKET CLIENT  (body-side — Mac only)
#
#  Connects upstream to the Kaggle brain's /ws_stream WebSocket.
#  Forwards prompts from the local HTML UI to the brain and relays
#  tokens + audio back to the HTML UI's WebSocket.
#
#  Degradation path:
#    DEFCON 5-3  →  forward to brain WS, relay response
#    DEFCON 1-2  →  generate locally with OllamaStreamGenerator
#
#  The body FastAPI app hosts its own /ws_stream for the HTML UI.
#  BodyWebSocketClient is the upstream half of that relay.
# ======================================================================
class BodyWebSocketClient:
    """
    Upstream relay: body → brain → body → HTML UI.

    brain_ws_url:  wss://xxxx.ngrok.io/ws_stream   (from Firestore brain_url)
    local_gen:     OllamaStreamGenerator for offline fallback

    Usage:
        client = BodyWebSocketClient(brain_ws_url="wss://...", local_gen=gen)
        # From body's /ws_stream handler:
        await client.relay_prompt(prompt, context, local_ws)
    """

    def __init__(
        self,
        brain_ws_url: str = "",
        local_gen:    Optional["OllamaStreamGenerator"] = None,
        defcon_fn:    Optional[Callable[[], int]] = None,
    ):
        self._brain_url = brain_ws_url
        self._local_gen = local_gen
        self._defcon_fn = defcon_fn   # () -> int: current DEFCON level
        self._relay_count  = 0
        self._local_count  = 0
        self._lock         = threading.Lock()

    def set_brain_url(self, url: str) -> None:
        with self._lock:
            self._brain_url = url

    def _should_use_local(self) -> bool:
        import neural_pipeline as _np_mod
        if not self._brain_url:
            return True
        if not _np_mod._WS_CLIENT_OK:
            return True
        if self._defcon_fn:
            try:
                return self._defcon_fn() <= 2
            except Exception:
                pass
        return False

    async def relay_prompt(
        self,
        prompt:   str,
        context:  dict,
        local_ws: "WebSocket",
        tts:      Optional["ElevenLabsTTS"]      = None,
        fallback: Optional["LocalTTSFallback"]   = None,
    ) -> None:
        """
        Route a prompt either to the brain WS (relay) or locally (Ollama).
        Streams tokens and audio back to local_ws (the HTML UI connection).

        local_ws must be an accepted FastAPI WebSocket.
        tts / fallback are used for audio if local generation is active.
        """
        if self._should_use_local():
            await self._generate_local(prompt, context, local_ws, tts, fallback)
        else:
            await self._relay_to_brain(prompt, context, local_ws)

    async def _relay_to_brain(
        self,
        prompt:   str,
        context:  dict,
        local_ws: "WebSocket",
    ) -> None:
        """
        Opens a WS connection to the brain, sends the prompt, and pipes
        every message (token JSON + raw audio bytes) to local_ws.
        Falls back to local generation on any connection error.
        """
        if not _WS_CLIENT_OK:
            await self._generate_local(prompt, context, local_ws, None, None)
            return

        with self._lock:
            url = self._brain_url

        try:
            async with _websockets_lib.connect(url, open_timeout=10) as brain_ws:
                self._relay_count += 1
                await brain_ws.send(json.dumps({"prompt": prompt, "context": context}))

                while True:
                    try:
                        msg = await asyncio.wait_for(brain_ws.recv(), timeout=60)
                    except asyncio.TimeoutError:
                        logger.warning("[BodyWS] Upstream recv timeout.")
                        break

                    if isinstance(msg, bytes):
                        # Raw MP3 bytes from brain TTS
                        await local_ws.send_bytes(msg)
                    else:
                        # JSON control message — forward as-is
                        await local_ws.send_text(msg)
                        try:
                            obj = json.loads(msg)
                            if obj.get("type") == "done":
                                break
                        except json.JSONDecodeError:
                            pass

        except Exception as e:
            logger.warning(f"[BodyWS] Brain relay failed ({e}), falling back to local.")
            await self._generate_local(prompt, context, local_ws, None, None)

    async def _generate_local(
        self,
        prompt:   str,
        context:  dict,
        local_ws: "WebSocket",
        tts:      Optional["ElevenLabsTTS"],
        fallback: Optional["LocalTTSFallback"],
    ) -> None:
        """
        Local generation via Ollama.  Mirrors the brain's token stream
        protocol exactly so the HTML UI does not need to know the difference.
        """
        if self._local_gen is None:
            await local_ws.send_json({
                "type":    "error",
                "message": "No local generator configured and brain unreachable."
            })
            return

        self._local_count += 1
        sentence_buf = ""
        total        = 0
        _tts         = tts or ElevenLabsTTS()

        try:
            async for token in self._local_gen.generate(prompt, context):
                total += 1
                await local_ws.send_json({"type": "token", "text": token})
                sentence_buf += token

                if _is_sentence_boundary(sentence_buf):
                    segs = SENTENCE_BOUNDARIES.split(sentence_buf)
                    for seg in segs[:-1]:
                        if seg.strip():
                            asyncio.create_task(
                                _flush_tts_to_ws(local_ws, seg, _tts, fallback)
                            )
                    sentence_buf = segs[-1] if segs else ""

            if sentence_buf.strip():
                await _flush_tts_to_ws(local_ws, sentence_buf, _tts, fallback)

            await local_ws.send_json({"type": "done", "total_tokens": total})

        except Exception as e:
            if type(e).__name__ != "WebSocketDisconnect":
                logger.error(f"[BodyWS] Local gen error: {e}")
                try:
                    await local_ws.send_json({"type": "error", "message": str(e)})
                except Exception:
                    pass

    def get_stats(self) -> dict:
        return {
            "brain_url":    self._brain_url,
            "relay_count":  self._relay_count,
            "local_count":  self._local_count,
            "ws_lib_ok":    _WS_CLIENT_OK,
        }


# ── Module-level helpers used by both WebSocketStreamHandler and BodyWebSocketClient
def _is_sentence_boundary(text: str) -> bool:
    if not text:
        return False
    if text.endswith("\n"):
        return True
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in ".!?"


async def _flush_tts_to_ws(
    ws:       "WebSocket",
    sentence: str,
    tts:      "ElevenLabsTTS",
    fallback: Optional["LocalTTSFallback"],
) -> bool:
    sentence = sentence.strip()
    if not sentence:
        return False
    mp3 = await tts.synthesize_with_fallback(sentence, fallback)
    if mp3 is None:
        return True   # local TTS played on speakers — no bytes to send
    try:
        await ws.send_json({
            "type":         "audio_start",
            "text":         sentence,
            "bytes_length": len(mp3),
        })
        await ws.send_bytes(mp3)
        return True
    except Exception as e:
        logger.warning(f"[WS] send audio failed: {e}")
        return False


# ======================================================================
# § 6  BODY FASTAPI MOUNT HELPER  (body-side)
# ======================================================================
def attach_body_ws_endpoint(
    app:          "FastAPI",
    ollama_gen:   Optional["OllamaStreamGenerator"]  = None,
    brain_client: Optional["BodyWebSocketClient"]    = None,
    tts:          Optional["ElevenLabsTTS"]          = None,
    local_tts:    Optional["LocalTTSFallback"]       = None,
) -> "WebSocketStreamHandler":
    """
    Mounts /ws_stream on the body's FastAPI app.

    If brain_client is given, each WS message is relayed upstream to the
    brain (with Ollama fallback).  Otherwise served directly by Ollama.

    Returns the WebSocketStreamHandler so the caller can hot-swap the
    generator via handler.inject_generator(fn).
    """
    if not _FASTAPI_OK:
        logger.error("FastAPI not available — cannot attach body /ws_stream")
        return None

    _tts     = tts or ElevenLabsTTS()
    handler  = WebSocketStreamHandler(
        tts=_tts,
        llm_generate_fn=ollama_gen.generate if ollama_gen else None,
        local_tts=local_tts,
    )


    async def body_ws_stream(websocket: WebSocket):
        await websocket.accept()
        logger.info(f"[BodyWS] HTML UI connected from {websocket.client}")
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error", "message": "Invalid JSON"
                    })
                    continue

                prompt  = msg.get("prompt", "")
                context = msg.get("context", {})

                if not prompt:
                    await websocket.send_json({
                        "type": "error", "message": "No prompt provided."
                    })
                    continue

                if brain_client:
                    await brain_client.relay_prompt(
                        prompt, context, websocket, _tts, local_tts
                    )
                else:
                    await handler.handle_stream(websocket, prompt, context)

        except Exception as exc:
            if type(exc).__name__ != "WebSocketDisconnect":
                logger.error(f"[BodyWS] Unhandled: {exc}")

    app.add_api_websocket_route("/ws_stream", body_ws_stream)
    logger.info("✅ Body /ws_stream endpoint attached.")
    return handler


# ======================================================================
# § 7  KAGGLE BRAIN HELPERS  (brain-side — unchanged from v13.2)
# ======================================================================
def attach_ws_stream_endpoint(
    app:                "FastAPI",
    llm_generate_fn:    Optional[Callable] = None,
    elevenlabs_api_key: str = ELEVENLABS_API_KEY,
) -> "WebSocketStreamHandler":
    """
    Attaches /ws_stream to an existing brain FastAPI app.
    llm_generate_fn: async generator(prompt, context) → yields str tokens
    Typically wired to _llm_stream_groq() in the Master Brain cell.
    """
    if not _FASTAPI_OK:
        logger.error("FastAPI not available — cannot attach /ws_stream")
        return None

    tts     = ElevenLabsTTS(api_key=elevenlabs_api_key)
    handler = WebSocketStreamHandler(tts=tts, llm_generate_fn=llm_generate_fn)


    async def ws_stream_endpoint(websocket: WebSocket):
        await websocket.accept()
        logger.info(f"[BrainWS] Client connected: {websocket.client}")
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "error", "message": "Invalid JSON"
                    })
                    continue

                prompt  = msg.get("prompt", "")
                context = msg.get("context", {})

                if not prompt:
                    await websocket.send_json({
                        "type": "error", "message": "No prompt provided."
                    })
                    continue

                await handler.handle_stream(websocket, prompt, context)

        except Exception as exc:
            if type(exc).__name__ != "WebSocketDisconnect":
                logger.error(f"[BrainWS] Unhandled: {exc}")

    app.add_api_websocket_route("/ws_stream", ws_stream_endpoint)
    logger.info("✅ Brain /ws_stream WebSocket endpoint attached.")
    return handler


def build_kaggle_app(llm_generate_fn: Optional[Callable] = None) -> "FastAPI":
    """
    Standalone brain FastAPI factory used in the Kaggle notebook.
    llm_generate_fn: async generator(prompt, context) → yields str tokens
    """
    if not _FASTAPI_OK:
        raise ImportError("FastAPI required: pip install fastapi uvicorn")

    app = FastAPI(title="Swayambhu Brain Neural Pipeline")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {
            "status":         "ok",
            "tts_configured": bool(ELEVENLABS_API_KEY),
            "model":          ELEVENLABS_MODEL,
        }

    attach_ws_stream_endpoint(app, llm_generate_fn=llm_generate_fn)
    return app


# ── Mock generators for testing ────────────────────────────────────────
async def _mock_llm_generator(prompt: str, context: dict) -> AsyncGenerator[str, None]:
    """Deterministic fake token stream for integration testing."""
    response = (
        f"I understand you said: {prompt}. "
        "Let me think about this carefully. "
        "The answer involves multiple components. "
        "First, we must consider the context. "
        "Then we analyze the data. Finally, we conclude."
    )
    for char in response:
        yield char
        await asyncio.sleep(0)   # yield to event loop, but don't slow tests


async def _mock_ollama_generator(
    prompt: str, context: dict
) -> AsyncGenerator[str, None]:
    """Fake Ollama token stream for body-side tests."""
    words = f"Offline response to: {prompt}. Running on local Ollama model.".split()
    for word in words:
        yield word + " "
        await asyncio.sleep(0)


# ======================================================================
# SELF-TEST
# ======================================================================
if __name__ == "__main__":
    import sys
    import unittest

    logging.basicConfig(level=logging.WARNING)

    print("\n⚡ NeuralPipeline v14.0 — Full Self-Test\n")
    print("=" * 65)

    passed_total = 0
    failed_total = 0

    def _ok(name: str, cond: bool, detail: str = "") -> None:
        global passed_total, failed_total
        if cond:
            print(f"  ✅ {name}")
            passed_total += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed_total += 1

    # ── TEST 1: Sentence boundary detection ────────────────────────────
    print("=== TEST 1: Sentence boundary detection ===")
    cases = [
        ("Hello world.",  True),
        ("Hello world",   False),
        ("Really?",       True),
        ("Yes!",          True),
        ("No comma,",     False),
        ("New line\n",    True),
        ("",              False),
        ("   ",           False),
        ("word\n",        True),
        ("end.",          True),
        ("mid... more",   False),
    ]
    for text, expected in cases:
        got = _is_sentence_boundary(text)
        _ok(f"boundary({repr(text)}) == {expected}", got == expected,
            f"got {got}")

    # ── TEST 2: ElevenLabsTTS — no key path ────────────────────────────
    print("\n=== TEST 2: ElevenLabsTTS no-key path ===")

    async def test_tts_no_key():
        tts = ElevenLabsTTS(api_key="")
        _ok("is_configured False when no key", not tts.is_configured)
        mp3 = await tts.synthesize_bytes("Hello world.")
        _ok("synthesize_bytes returns None (no key)", mp3 is None)
        mp3b = await tts.synthesize_with_fallback("Hello.")
        _ok("synthesize_with_fallback returns None (no key, no fallback)", mp3b is None)
        stats = tts.get_stats()
        _ok("get_stats returns dict with 'configured'", "configured" in stats)
        _ok("stats configured == False", stats["configured"] is False)

    asyncio.run(test_tts_no_key())

    # ── TEST 3: ElevenLabsTTS — with mock fallback ──────────────────────
    print("\n=== TEST 3: ElevenLabsTTS with mock LocalTTS fallback ===")

    class _MockLocalTTS:
        called: list = []
        async def speak_async(self, text: str):
            _MockLocalTTS.called.append(text)
            return None   # local TTS plays on speakers — no bytes

    async def test_tts_fallback():
        tts = ElevenLabsTTS(api_key="", fallback=_MockLocalTTS())
        _MockLocalTTS.called.clear()
        result = await tts.synthesize_with_fallback("Test sentence.")
        _ok("With fallback and no key, fallback.speak_async called",
            len(_MockLocalTTS.called) == 1, str(_MockLocalTTS.called))
        _ok("synthesize_with_fallback returns None (local TTS, no bytes)",
            result is None)

    asyncio.run(test_tts_fallback())

    # ── TEST 4: LocalTTSFallback — non-Mac path ─────────────────────────
    print("\n=== TEST 4: LocalTTSFallback non-Mac no-op ===")

    # Monkeypatch _IS_MACOS to False for portability of this test
    class _FakeLocalTTS(LocalTTSFallback):
        _IS_MACOS = False

    ltts = _FakeLocalTTS()
    _ok("is_available False on non-Mac", not ltts.is_available)
    ok = ltts.speak("This should not play")
    _ok("speak() returns False on non-Mac", ok is False)
    path = ltts.speak_to_file("No audio")
    _ok("speak_to_file returns None on non-Mac", path is None)

    async def test_local_tts_async():
        result = await ltts.speak_async("Async no-op on non-Mac")
        _ok("speak_async returns None on non-Mac", result is None)

    asyncio.run(test_local_tts_async())

    stats_ltts = ltts.get_stats()
    _ok("get_stats returns dict", isinstance(stats_ltts, dict))
    _ok("platform field present", "platform" in stats_ltts)

    # ── TEST 5: LocalTTSFallback — Mac path (mocked subprocess) ────────
    print("\n=== TEST 5: LocalTTSFallback Mac path (subprocess mocked) ===")

    import unittest.mock as _mock

    class _MacLocalTTS(LocalTTSFallback):
        _IS_MACOS = True

    mac_ltts = _MacLocalTTS(voice="Alex", rate=180)
    with _mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = _mock.Mock(returncode=0)
        result_speak = mac_ltts.speak("Hello world.")
    _ok("speak() returns True on Mac (mocked)", result_speak is True)
    _ok("call_count incremented", mac_ltts._call_count == 1)

    with _mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = _mock.Mock(returncode=0)
        path_out = mac_ltts.speak_to_file("Test audio", output_path="/tmp/test.aiff")
    _ok("speak_to_file returns path on Mac (mocked)",
        path_out == "/tmp/test.aiff", str(path_out))

    with _mock.patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("say not found")
        result_fail = mac_ltts.speak("Fail gracefully")
    _ok("speak() returns False when say missing", result_fail is False)

    # ── TEST 6: WebSocketStreamHandler — token streaming ───────────────
    print("\n=== TEST 6: WebSocketStreamHandler token streaming ===")

    class _FakeWS:
        """Minimal fake WebSocket that records sent messages."""
        def __init__(self):
            self.sent_json:  list = []
            self.sent_bytes: list = []

        async def send_json(self, obj):
            self.sent_json.append(obj)

        async def send_bytes(self, data):
            self.sent_bytes.append(data)

    async def test_handler_stream():
        tts     = ElevenLabsTTS(api_key="")   # no key → TTS skipped
        handler = WebSocketStreamHandler(tts=tts, llm_generate_fn=_mock_llm_generator)
        ws      = _FakeWS()

        await handler.handle_stream(ws, "test prompt", {})

        token_msgs = [m for m in ws.sent_json if m.get("type") == "token"]
        done_msgs  = [m for m in ws.sent_json if m.get("type") == "done"]

        _ok("Token messages received",        len(token_msgs) > 0, str(len(token_msgs)))
        _ok("Exactly one done message",       len(done_msgs) == 1, str(len(done_msgs)))
        _ok("done.total_tokens > 0",
            done_msgs[0].get("total_tokens", 0) > 0, str(done_msgs[0]))
        _ok("No error messages",
            not any(m.get("type") == "error" for m in ws.sent_json))

        # Check token count matches characters
        all_text = "".join(m.get("text", "") for m in token_msgs)
        _ok("Concatenated tokens form coherent text", len(all_text) > 10, repr(all_text[:30]))

    asyncio.run(test_handler_stream())

    # ── TEST 7: WebSocketStreamHandler — no generator path ─────────────
    print("\n=== TEST 7: WebSocketStreamHandler no generator ===")

    async def test_handler_no_gen():
        tts     = ElevenLabsTTS(api_key="")
        handler = WebSocketStreamHandler(tts=tts, llm_generate_fn=None)
        ws      = _FakeWS()
        await handler.handle_stream(ws, "anything", {})
        error_msgs = [m for m in ws.sent_json if m.get("type") == "error"]
        _ok("Error sent when no generator", len(error_msgs) == 1, str(ws.sent_json))
        _ok("Error message mentions LLM",
            "LLM" in error_msgs[0].get("message", ""), str(error_msgs[0]))

    asyncio.run(test_handler_no_gen())

    # ── TEST 8: WebSocketStreamHandler — inject_generator hot-swap ─────
    print("\n=== TEST 8: inject_generator hot-swap ===")

    async def test_inject():
        tts     = ElevenLabsTTS(api_key="")
        handler = WebSocketStreamHandler(tts=tts, llm_generate_fn=None)
        _ok("Handler starts with no generator", handler._llm_generate is None)

        handler.inject_generator(_mock_llm_generator)
        _ok("After inject, generator is set", handler._llm_generate is not None)

        ws = _FakeWS()
        await handler.handle_stream(ws, "hello", {})
        done_msgs = [m for m in ws.sent_json if m.get("type") == "done"]
        _ok("Stream works after inject", len(done_msgs) == 1)

    asyncio.run(test_inject())

    # ── TEST 9: OllamaStreamGenerator — Ollama offline path ────────────
    print("\n=== TEST 9: OllamaStreamGenerator offline error ===")

    async def test_ollama_offline():
        gen = OllamaStreamGenerator(
            ollama_url="http://localhost:19999",   # intentionally wrong port
            model="test-model",
        )
        tokens = []
        async for t in gen.generate("hello", {}):
            tokens.append(t)

        joined = "".join(tokens)
        _ok("Ollama offline returns error token",
            "not running" in joined or "unavailable" in joined or "error" in joined.lower(),
            repr(joined[:80]))
        _ok("call_count incremented", gen._call_count == 1)
        stats = gen.get_stats()
        _ok("get_stats returns dict", isinstance(stats, dict))
        _ok("stats has url, model, call_count",
            all(k in stats for k in ["url", "model", "call_count"]))

    asyncio.run(test_ollama_offline())

    # ── TEST 10: OllamaStreamGenerator — streaming response mocked ──────
    print("\n=== TEST 10: OllamaStreamGenerator streaming (mocked httpx) ===")

    async def test_ollama_mock():
        gen = OllamaStreamGenerator(
            ollama_url="http://localhost:11434",
            model="llama3.2:3b",
        )

        # Simulate Ollama streaming response: NDJSON lines
        fake_lines = [
            b'{"response":"Hello","done":false}\n',
            b'{"response":" world","done":false}\n',
            b'{"response":"!","done":true}\n',
        ]

        class _FakeAsyncIterLines:
            def __init__(self, lines):
                self._lines = [l.decode() for l in lines]
                self._i = 0
            def __aiter__(self): return self
            async def __anext__(self):
                if self._i >= len(self._lines):
                    raise StopAsyncIteration
                val = self._lines[self._i]
                self._i += 1
                return val

        class _FakeStreamResp:
            status_code = 200
            def aiter_lines(self):
                return _FakeAsyncIterLines(fake_lines)
            async def aread(self): return b""

        class _FakeStreamCtx:
            async def __aenter__(self): return _FakeStreamResp()
            async def __aexit__(self, *a): pass

        class _FakeAsyncClient:
            def __init__(self, timeout=None): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            def stream(self, *a, **kw): return _FakeStreamCtx()

        with _mock.patch("httpx.AsyncClient", _FakeAsyncClient):
            tokens = []
            async for t in gen.generate("hi", {}):
                tokens.append(t)

        joined = "".join(tokens)
        _ok("Mocked Ollama yields 3 tokens", len(tokens) == 3, str(tokens))
        _ok("Concatenated text == 'Hello world!'", joined == "Hello world!", repr(joined))

    asyncio.run(test_ollama_mock())

    # ── TEST 11: OllamaStreamGenerator — context handling ───────────────
    print("\n=== TEST 11: OllamaStreamGenerator context handling ===")

    async def test_ollama_context():
        gen = OllamaStreamGenerator()

        # Capture the body sent to httpx
        captured_bodies: list = []

        class _CaptureFakeStreamResp:
            status_code = 200
            def aiter_lines(self): return _EmptyIter()
            async def aread(self): return b""

        class _EmptyIter:
            def __aiter__(self): return self
            async def __anext__(self): raise StopAsyncIteration

        class _FakeStreamCtxCapture:
            async def __aenter__(self): return _CaptureFakeStreamResp()
            async def __aexit__(self, *a): pass

        class _FakeClientCapture:
            def __init__(self, timeout=None): pass
            async def __aenter__(self): return self
            async def __aexit__(self, *a): pass
            def stream(self, method, url, json=None, **kw):
                captured_bodies.append(json)
                return _FakeStreamCtxCapture()

        ctx_stressed = {
            "sys_override":   "USER_STRESSED_BE_CONCISE",
            "episodic_memory": ["User prefers Python.", "User is on macOS."],
        }

        with _mock.patch("httpx.AsyncClient", _FakeClientCapture):
            async for _ in gen.generate("Tell me about Python.", ctx_stressed):
                pass

        _ok("One body captured", len(captured_bodies) == 1)
        body = captured_bodies[0]
        _ok("stream=True in body", body.get("stream") is True)
        _ok("Stressed concise injected into system",
            "2 sentences" in body.get("system", ""), body.get("system", "")[:80])
        _ok("Memory context injected",
            "Memory context" in body.get("system", ""), body.get("system", "")[:200])

    asyncio.run(test_ollama_context())

    # ── TEST 12: BodyWebSocketClient — local fallback path ──────────────
    print("\n=== TEST 12: BodyWebSocketClient local fallback ===")

    async def test_body_ws_client():
        # brain_url empty → always use local
        client = BodyWebSocketClient(brain_ws_url="", local_gen=None)
        _ok("should_use_local True when no brain URL", client._should_use_local())

        # With brain URL but ws lib unavailable → use local
        client2 = BodyWebSocketClient(brain_ws_url="wss://brain:443", local_gen=None)
        with _mock.patch("neural_pipeline._WS_CLIENT_OK", False):
            _ok("should_use_local True when ws lib missing",
                client2._should_use_local())

        # DEFCON 1 → local
        client3 = BodyWebSocketClient(
            brain_ws_url="wss://brain:443",
            local_gen=None,
            defcon_fn=lambda: 1,
        )
        _ok("should_use_local True at DEFCON 1", client3._should_use_local())

        # DEFCON 5 → relay (brain URL set, lib ok)
        client4 = BodyWebSocketClient(
            brain_ws_url="wss://brain:443",
            local_gen=None,
            defcon_fn=lambda: 5,
        )
        with _mock.patch("neural_pipeline._WS_CLIENT_OK", True):
            _ok("should_use_local False at DEFCON 5", not client4._should_use_local())

        # Local generation with mock generator
        tts  = ElevenLabsTTS(api_key="")
        gen  = OllamaStreamGenerator()
        gen.generate = _mock_ollama_generator   # type: ignore[method-assign]

        client5 = BodyWebSocketClient(brain_ws_url="", local_gen=gen)
        ws      = _FakeWS()
        await client5._generate_local("test", {}, ws, tts, None)

        token_msgs = [m for m in ws.sent_json if m.get("type") == "token"]
        done_msgs  = [m for m in ws.sent_json if m.get("type") == "done"]
        _ok("Local gen produces token messages", len(token_msgs) > 0)
        _ok("Local gen produces done message",   len(done_msgs) == 1)

        stats = client5.get_stats()
        _ok("get_stats returns dict", isinstance(stats, dict))
        _ok("local_count incremented", stats["local_count"] == 1, str(stats))

    asyncio.run(test_body_ws_client())

    # ── TEST 13: File-level syntax check ───────────────────────────────
    print("\n=== TEST 13: File-level syntax check ===")
    import ast as _ast
    src = open(__file__).read()
    try:
        _ast.parse(src)
        _ok("File parses without SyntaxError", True)
    except SyntaxError as e:
        _ok("File parses without SyntaxError", False, str(e))

    # ── TEST 14: Module-level exports smoke test ────────────────────────
    print("\n=== TEST 14: Export completeness ===")
    expected_exports = [
        "ElevenLabsTTS", "LocalTTSFallback", "WebSocketStreamHandler",
        "OllamaStreamGenerator", "BodyWebSocketClient",
        "attach_body_ws_endpoint", "attach_ws_stream_endpoint",
        "build_kaggle_app", "SENTENCE_BOUNDARIES",
        "_mock_llm_generator", "_mock_ollama_generator",
    ]
    import neural_pipeline as _self
    for name in expected_exports:
        _ok(f"Export exists: {name}", hasattr(_self, name))

    # ── SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"  Results: {passed_total} passed, {failed_total} failed")
    if failed_total == 0:
        print("  ✅ All NeuralPipeline v14.0 tests passed.")
    else:
        print(f"  ⚠️  {failed_total} test(s) failed — review above.")
    print("=" * 65)
    sys.exit(0 if failed_total == 0 else 1)
