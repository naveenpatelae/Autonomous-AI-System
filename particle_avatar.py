#!/usr/bin/env python3
# =====================================================================
# ✨ PARTICLE AVATAR — 4K 3D Particle Ball HTML Generator
#
# UPGRADE: FFT Audio-Reactive Physics
# Incorporates HTML5 Web Audio API and AnalyserNode.
# The 3D sphere now physically vibrates based on real-time microphone
# frequency data (bass/treble bins) rather than pre-baked animations.
# =====================================================================

from __future__ import annotations
from pathlib import Path

PARTICLE_AVATAR_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Swayambhu — Sovereign AI</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #000008;
    overflow: hidden;
    font-family: 'Inter', 'SF Pro Display', system-ui, sans-serif;
    color: #fff;
    width: 100vw; height: 100vh;
  }
  #canvas-container {
    position: fixed; inset: 0;
    display: flex; align-items: center; justify-content: center;
  }
  canvas { display: block; }

  /* HUD */
  #hud {
    position: fixed; bottom: 0; left: 0; right: 0;
    padding: 20px 30px;
    background: linear-gradient(transparent, rgba(0,0,20,0.95));
    display: flex; flex-direction: column; align-items: center; gap: 12px;
    z-index: 10;
  }
  #greeting {
    font-size: 22px; font-weight: 300; letter-spacing: 2px;
    color: rgba(180,200,255,0.9);
    text-align: center;
    text-shadow: 0 0 20px rgba(100,150,255,0.5);
    transition: opacity 0.5s;
  }
  #response-text {
    font-size: 16px; font-weight: 300;
    color: rgba(150,200,255,0.8);
    max-width: 700px; text-align: center; line-height: 1.6;
    min-height: 40px;
    transition: opacity 0.3s;
  }
  #state-indicator {
    font-size: 11px; letter-spacing: 3px; text-transform: uppercase;
    color: rgba(100,200,150,0.6);
  }
  #controls {
    display: flex; gap: 12px; align-items: center;
  }
  .ctrl-btn {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.15);
    color: rgba(200,220,255,0.9);
    padding: 8px 20px; border-radius: 20px;
    cursor: pointer; font-size: 13px; letter-spacing: 1px;
    transition: all 0.2s;
  }
  .ctrl-btn:hover {
    background: rgba(100,150,255,0.2);
    border-color: rgba(100,150,255,0.5);
    box-shadow: 0 0 12px rgba(100,150,255,0.3);
  }
  .ctrl-btn.active {
    background: rgba(100,255,150,0.2);
    border-color: rgba(100,255,150,0.5);
    box-shadow: 0 0 12px rgba(100,255,150,0.3);
  }
  #input-area {
    display: flex; gap: 10px; width: 100%; max-width: 600px;
  }
  #cmd-input {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.15);
    color: #fff; padding: 10px 16px; border-radius: 20px;
    font-size: 14px; outline: none;
    transition: border-color 0.2s;
  }
  #cmd-input:focus {
    border-color: rgba(100,150,255,0.6);
    box-shadow: 0 0 12px rgba(100,150,255,0.2);
  }
  #send-btn {
    background: rgba(80,120,255,0.3);
    border: 1px solid rgba(80,120,255,0.5);
    color: #fff; padding: 10px 20px; border-radius: 20px;
    cursor: pointer; font-size: 14px;
    transition: all 0.2s;
  }
  #send-btn:hover { background: rgba(80,120,255,0.5); }

  /* Emotion badge */
  #emotion-badge {
    position: fixed; top: 20px; left: 20px;
    background: rgba(0,5,20,0.7);
    border: 1px solid rgba(100,150,200,0.3);
    border-radius: 20px; padding: 6px 14px;
    font-size: 11px; letter-spacing: 2px;
    color: rgba(150,200,255,0.7);
    z-index: 20;
  }
</style>
</head>
<body>

<div id="canvas-container"><canvas id="particle-canvas"></canvas></div>
<div id="emotion-badge">😐 NEUTRAL</div>

<div id="hud">
  <div id="greeting">Initializing...</div>
  <div id="response-text"></div>
  <div id="state-indicator">AWAKENING</div>
  <div id="input-area">
    <input id="cmd-input" type="text" placeholder="Speak or type a command..." autocomplete="off"/>
    <button id="send-btn" onclick="sendCommand()">↑ Send</button>
  </div>
  <div id="controls">
    <button id="voice-btn" class="ctrl-btn" onclick="toggleVoice()">🎤 Voice</button>
    <button class="ctrl-btn" onclick="sleepMode()">😴 Sleep</button>
  </div>
</div>

<script>
const API = 'http://localhost:8003';
let voiceActive = false, recognition = null;
// ═══════════════════════════════════════════════════════════════════
// 1. FFT AUDIO-REACTIVE PHYSICS (Web Audio API)
// ═══════════════════════════════════════════════════════════════════
let audioCtx;
let analyser;
let dataArray;
let isAudioReactive = false;

async function initAudioContext() {
    if (audioCtx) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();

        // Fast Fourier Transform Size (determines frequency bins)
        analyser.fftSize = 256; 
        const source = audioCtx.createMediaStreamSource(stream);
        source.connect(analyser);

        const bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(bufferLength);
        isAudioReactive = true;
        console.log("FFT Audio-Reactive Physics Engaged.");
    } catch (err) {
        console.warn("Microphone access denied for FFT physics:", err);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 2. 3D PARTICLE SPHERE (WebGL)
// ═══════════════════════════════════════════════════════════════════
const canvas = document.getElementById('particle-canvas');
const ctx2d = canvas.getContext('2d');

let W, H, animFrame;
let particles = [], NUM_PARTICLES = 4000;
let stateColor = [0.3, 0.5, 1.0];   
let stateFreq  = 1.0;               
let stateMode  = 'idle';
let globalTime = 0;
let isAsleep   = false;

function resize() {
  W = canvas.width  = window.innerWidth;
  H = canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();

function initParticles() {
  particles = [];
  for (let i = 0; i < NUM_PARTICLES; i++) {
    const theta = Math.acos(2 * Math.random() - 1);
    const phi   = Math.random() * Math.PI * 2;
    const r     = 180 + (Math.random() - 0.5) * 30;
    particles.push({
      theta, phi, r,
      baseR: r,
      speed:  0.0003 + Math.random() * 0.0005,
      phase:  Math.random() * Math.PI * 2,
      size:   1 + Math.random() * 1.5,
      brightness: 0.5 + Math.random() * 0.5,
    });
  }
}
initParticles();

const STATE_PARAMS = {
  idle:       { color: [0.2, 0.4, 1.0],   freq: 0.8,  amp: 8,   pulse: 0.3 },
  listening:  { color: [0.0, 1.0, 0.5],   freq: 2.5,  amp: 25,  pulse: 0.8 },
  processing: { color: [0.8, 0.3, 1.0],   freq: 4.0,  amp: 35,  pulse: 1.2 },
  speaking:   { color: [0.1, 0.8, 1.0],   freq: 3.0,  amp: 20,  pulse: 0.9 },
  sleeping:   { color: [0.1, 0.1, 0.3],   freq: 0.2,  amp: 3,   pulse: 0.1 },
  error:      { color: [1.0, 0.2, 0.1],   freq: 5.0,  amp: 40,  pulse: 1.5 },
  defcon:     { color: [1.0, 0.1, 0.0],   freq: 8.0,  amp: 50,  pulse: 2.0 }
};

let targetParams = {...STATE_PARAMS['idle']};
let currentColor = [0.2, 0.4, 1.0];
let currentFreq  = 0.8;

function setState(stateName) {
  const params = STATE_PARAMS[stateName] || STATE_PARAMS['idle'];
  targetParams = params;
  stateMode    = stateName;
  document.getElementById('state-indicator').textContent = stateName.toUpperCase().replace('_',' ');
}

function lerp(a, b, t) { return a + (b - a) * t; }

function render() {
  animFrame = requestAnimationFrame(render);
  globalTime += 0.016;

  // ── FFT Audio Physics Processing ──
  let fftAmpMod = 0;
  let fftJitter = 0;

  if (isAudioReactive && analyser) {
      analyser.getByteFrequencyData(dataArray);

      // Calculate overall volume (average across bins)
      let sum = 0;
      for(let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i];
      }
      let avgVolume = sum / dataArray.length;

      // Calculate bass (lower bins) for core pulsing
      let bassSum = 0;
      for(let i = 0; i < 10; i++) { bassSum += dataArray[i]; }
      let bassAvg = bassSum / 10;

      // Inject audio modifiers into the physics variables
      fftAmpMod = (avgVolume / 255) * 60; // Up to 60px extra amplitude based on volume
      fftJitter = (bassAvg / 255) * 2.5;  // Bass introduces phase jitter
  }

  // Lerp toward target params
  const lp = 0.04;
  for (let i = 0; i < 3; i++) {
    currentColor[i] = lerp(currentColor[i], targetParams.color[i], lp);
  }
  currentFreq = lerp(currentFreq, targetParams.freq, lp);

  ctx2d.clearRect(0, 0, W, H);
  ctx2d.fillStyle = 'rgba(0,0,8,0.15)';
  ctx2d.fillRect(0, 0, W, H);

  const cx = W * 0.5, cy = H * 0.5 - 60;

  // Combine base amplitude with FFT modifiers
  const finalAmp = targetParams.amp + fftAmpMod;
  const finalPulse = targetParams.pulse + (fftJitter * 0.5);

  // Glow orb
  const glowR = 200 + Math.sin(globalTime * 0.5) * 20 + fftAmpMod;
  const grad = ctx2d.createRadialGradient(cx, cy, 0, cx, cy, glowR);
  const [r,g,b] = currentColor;
  grad.addColorStop(0, `rgba(${Math.floor(r*80)},${Math.floor(g*80)},${Math.floor(b*180)},0.15)`);
  grad.addColorStop(1, 'transparent');
  ctx2d.fillStyle = grad;
  ctx2d.beginPath();
  ctx2d.arc(cx, cy, glowR, 0, Math.PI*2);
  ctx2d.fill();

  const slerpT = globalTime;
  for (const p of particles) {
    // Inject audio jitter into the rotational speed
    p.phi += (p.speed * currentFreq) + (Math.random() * fftJitter * 0.01);

    const wave = Math.sin(slerpT * currentFreq + p.phase) * finalAmp * finalPulse;
    const r3d = p.baseR + wave + (Math.random() * fftJitter * 5); // Audio shattering effect

    const x3d = r3d * Math.sin(p.theta) * Math.cos(p.phi);
    const y3d = r3d * Math.cos(p.theta);
    const z3d = r3d * Math.sin(p.theta) * Math.sin(p.phi);

    const fov = 600;
    const pz  = z3d + 500;
    const px  = cx + (x3d * fov) / pz;
    const py  = cy + (y3d * fov) / pz;
    const depth = (pz - 320) / 350;  

    if (depth < 0) continue;

    const alpha = p.brightness * (0.3 + 0.7 * depth);
    const sz    = Math.max(0.3, p.size * (0.5 + 0.8 * depth));
    const cr    = Math.floor(r * 220 * (0.5 + depth));
    const cg    = Math.floor(g * 220 * (0.5 + depth));
    const cb    = Math.floor(b * 255 * (0.6 + depth * 0.4));

    ctx2d.beginPath();
    ctx2d.arc(px, py, sz, 0, Math.PI * 2);
    ctx2d.fillStyle = `rgba(${cr},${cg},${cb},${alpha.toFixed(2)})`;
    ctx2d.fill();
  }
}
render();

// ═══════════════════════════════════════════════════════════════════
// 3. VOICE & SYSTEM CONTROLS
// ═══════════════════════════════════════════════════════════════════


// ── Networking State ──
let stateWs = null;   // Port 8007: Avatar UI states (colors/emotions)
let streamWs = null;  // Port 8003: Neural Pipeline (tokens/audio)
let audioQueue = [];
let isPlayingAudio = false;

async function apiPost(endpoint, body) {
  try {
    const res = await fetch(API + endpoint, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    return await res.json();
  } catch(e) { return {error: e.message}; }
}

function setResponse(text) {
  const el = document.getElementById('response-text');
  el.style.opacity = 0;
  setTimeout(() => { el.textContent = text; el.style.opacity = 1; }, 200);
}

// ── WebSocket Bridges ──
function connectWebSockets() {
  // 1. Avatar State WebSocket (Local UI Updates)
  try {
    stateWs = new WebSocket('ws://localhost:8007/ws/avatar');
    stateWs.onmessage = e => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.state)   setState(msg.state);
        if (msg.emotion) document.getElementById('emotion-badge').textContent = msg.emotion.toUpperCase();
        if (msg.text)    setResponse(msg.text);
      } catch(ex) {}
    };
    stateWs.onclose = () => setTimeout(connectWebSockets, 3000);
  } catch(e) {}

  // 2. Neural Pipeline WebSocket (Token & Audio Streaming)
  try {
    streamWs = new WebSocket('ws://localhost:8003/ws_stream');
    streamWs.binaryType = 'arraybuffer'; // Crucial for receiving raw MP3 bytes
    
    streamWs.onmessage = async (e) => {
      if (typeof e.data === 'string') {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'token') {
            const el = document.getElementById('response-text');
            el.textContent += msg.text;
            el.style.opacity = 1;
          } else if (msg.type === 'done') {
            // Keep 'speaking' state if audio is still playing, otherwise idle
            if (!isPlayingAudio && audioQueue.length === 0) {
              setTimeout(() => setState('idle'), 1500);
            }
          } else if (msg.type === 'error') {
            setResponse("Error: " + msg.message);
            setState('error');
          }
        } catch(ex) {}
      } else {
        // Raw MP3 bytes received! Queue and play.
        audioQueue.push(e.data);
        playNextAudio();
      }
    };
    streamWs.onclose = () => setTimeout(connectWebSockets, 3000);
  } catch(e) {}
}

// ── Audio Playback Pipeline ──
async function playNextAudio() {
  if (isPlayingAudio || audioQueue.length === 0) return;
  isPlayingAudio = true;
  setState('speaking'); // Trigger the UI to pulse to the audio!
  
  await initAudioContext(); // Ensure context is unlocked
  
  const audioData = audioQueue.shift();
  try {
    const audioBuffer = await audioCtx.decodeAudioData(audioData);
    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    
    // Route audio through the FFT Analyser to make the sphere vibrate!
    source.connect(analyser); 
    analyser.connect(audioCtx.destination); 
    
    source.onended = () => {
      isPlayingAudio = false;
      if (audioQueue.length === 0) {
        setState('idle'); // Audio finished, return to idle
      } else {
        playNextAudio();  // Play next chunk
      }
    };
    source.start(0);
  } catch(e) {
    console.error("Audio decode error:", e);
    isPlayingAudio = false;
    playNextAudio();
  }
}

// ── Command Execution ──
async function sendCommand(cmd) {
  const input = document.getElementById('cmd-input');
  const command = cmd || input.value.trim();
  if (!command) return;
  input.value = '';

  setState('processing');
  
  // Clear old text for the new stream
  const el = document.getElementById('response-text');
  el.textContent = '';
  el.style.opacity = 1;
  
  await initAudioContext();

  // Route command through the ultra-low-latency WebSocket
  if (streamWs && streamWs.readyState === WebSocket.OPEN) {
    streamWs.send(JSON.stringify({ prompt: command, context: {} }));
  } else {
    setResponse('Neural pipeline disconnected. Attempting reconnect...');
    setState('error');
    connectWebSockets();
  }
}

document.getElementById('cmd-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') sendCommand();
});

function toggleVoice() {
  initAudioContext(); // Ensure Audio Context is active for FFT Physics

  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    alert('Speech recognition not supported in this browser.');
    return;
  }
  if (voiceActive) {
    recognition && recognition.stop();
    voiceActive = false;
    document.getElementById('voice-btn').classList.remove('active');
    setState('idle');
    return;
  }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SR();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    voiceActive = true;
    document.getElementById('voice-btn').classList.add('active');
    setState('listening');
    document.getElementById('greeting').textContent = 'Listening...';
  };
  recognition.onresult = e => {
    const transcript = e.results[e.results.length-1][0].transcript;
    document.getElementById('cmd-input').value = transcript;
    if (e.results[e.results.length-1].isFinal) {
      voiceActive = false;
      document.getElementById('voice-btn').classList.remove('active');
      sendCommand(transcript);
    }
  };
  recognition.onerror = () => { voiceActive = false; document.getElementById('voice-btn').classList.remove('active'); setState('error'); };
  recognition.onend   = () => { voiceActive = false; document.getElementById('voice-btn').classList.remove('active'); };
  recognition.start();
}

async function sleepMode() {
  setState('sleeping');
  setResponse('Going to sleep. Wake me when needed.');
  await apiPost('/avatar/state', {state: 'sleeping'});
}

// Mount the bridges immediately
connectWebSockets();

// ── Core Readiness Loop ──
async function waitForBrain() {
  const input = document.getElementById('cmd-input');
  const btn = document.getElementById('send-btn');
  const greeting = document.getElementById('greeting');
  
  // 1. Lock the interface
  input.disabled = true;
  btn.disabled = true;
  greeting.textContent = "Booting Neural Core...";
  setState('processing');

  // 2. Poll the Edge Server until it wakes up
  while (true) {
    try {
      const r = await fetch(API + '/health');
      if (r.ok) {
        // 3. Unlock interface when brain is online
        greeting.textContent = "Sovereign OS Online.";
        input.disabled = false;
        btn.disabled = false;
        setState('idle');
        break;
      }
    } catch(e) {
      // Server not up yet, ignore the error and loop
    }
    await new Promise(resolve => setTimeout(resolve, 1500)); // wait 1.5s
  }
}

// Ignite the loop
waitForBrain();
</script>
</body>
</html>"""


def generate_particle_avatar(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(PARTICLE_AVATAR_HTML, encoding="utf-8")
    return output_path


# ─────────────────────────────────────────────────────────────────────
# PARTICLE AVATAR SERVER
# Hosts the WebGL HTML avatar on a local HTTP + WebSocket server so
# swayambhu_v13.py can call set_state() to drive the particle sphere.
# ─────────────────────────────────────────────────────────────────────
class ParticleAvatarServer:
    """
    Serves the 4K WebGL particle avatar HTML on `port` (default 8007)
    and exposes a WebSocket endpoint at ws://localhost:{port}/ws/avatar
    so the orchestrator can push state/emotion changes in real-time.

    Usage:
        server = ParticleAvatarServer(port=8007)
        server.start()
        server.set_state("listening")   # drives the particle sphere colour
        server.set_emotion("happy")
        server.stop()
    """

    VALID_STATES = {
        "idle", "listening", "processing", "speaking",
        "sleeping", "wake", "sleep", "error", "defcon",
    }

    def __init__(self, port: int = 8007):
        self._port = port
        self._app = None
        self._connections: list = []   # active WS connections
        self._conn_lock = None
        self._thread = None
        self._running = False
        self._current_state = "idle"
        self._html_path: Optional[Path] = None

    def start(self):
        """Start FastAPI server in a background daemon thread."""
        try:
            from fastapi import FastAPI, WebSocket, WebSocketDisconnect
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import HTMLResponse
            import uvicorn, asyncio

            self._conn_lock = asyncio.Lock() if False else None  # use list + thread safety
            import threading as _thr
            self._conn_lock = _thr.Lock()

            # Write the HTML to a temp file
            self._html_path = Path("/tmp/swayambhu_avatar.html")
            generate_particle_avatar(self._html_path)

            app = FastAPI(title="Swayambhu Particle Avatar")
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
            )
            self._app = app
            server_ref = self   # capture for closures

            @app.get("/", response_class=HTMLResponse)
            async def serve_avatar():
                try:
                    # Always regenerate from current source — never serve stale cache
                    generate_particle_avatar(server_ref._html_path)
                    content = server_ref._html_path.read_text(encoding="utf-8")
                    return HTMLResponse(
                        content=content,
                        headers={
                            "Cache-Control": "no-store, no-cache, must-revalidate",
                            "Pragma":        "no-cache",
                            "Expires":       "0",
                        },
                    )
                except Exception as e:
                    return HTMLResponse(content=f"<h1>Avatar error: {e}</h1>")

            @app.get("/health")
            async def avatar_health():
                return {
                    "status": "online",
                    "state": server_ref._current_state,
                    "connections": len(server_ref._connections),
                    "port": server_ref._port,
                }

            @app.websocket("/ws/avatar")
            async def ws_avatar(websocket: WebSocket):
                await websocket.accept()
                with server_ref._conn_lock:
                    server_ref._connections.append(websocket)
                try:
                    # Send current state immediately on connect
                    await websocket.send_json({
                        "state": server_ref._current_state,
                        "text": "Avatar connected.",
                    })
                    # Keep alive — receive and ignore any client pings
                    while True:
                        try:
                            await websocket.receive_text()
                        except Exception:
                            break
                except WebSocketDisconnect:
                    pass
                finally:
                    with server_ref._conn_lock:
                        if websocket in server_ref._connections:
                            server_ref._connections.remove(websocket)

            def _run():
                import asyncio as _asyncio
                import uvicorn
                loop = _asyncio.new_event_loop()
                _asyncio.set_event_loop(loop)
                server_ref._server_loop = loop   # expose for _broadcast
                config = uvicorn.Config(
                    app, host="0.0.0.0", port=self._port,
                    log_level="warning", loop="asyncio")
                server = uvicorn.Server(config)
                loop.run_until_complete(server.serve())

            self._thread = _thr.Thread(target=_run, daemon=True,
                                       name="ParticleAvatarServer")
            self._thread.start()
            self._running = True
            print(f"✨ [ParticleAvatar] Serving on http://localhost:{self._port}")
            print(f"   Open in browser to see the 4K particle sphere.")

        except ImportError as e:
            print(f"⚠️  [ParticleAvatar] FastAPI/uvicorn not available: {e}")
            print("   pip install fastapi uvicorn")
        except Exception as e:
            print(f"⚠️  [ParticleAvatar] Start error: {e}")

    def set_state(self, state: str):
        """
        Push a state change to all connected WebSocket clients.
        The particle sphere changes colour/behaviour in response.
        """
        if state not in self.VALID_STATES:
            return
        self._current_state = state
        self._broadcast({"state": state})

    def set_emotion(self, emotion: str):
        """Push an emotion label to connected clients."""
        self._broadcast({"emotion": emotion})

    def set_defcon(self, level: int):
        """Push a DEFCON level change to connected clients."""
        # Broadcast the DEFCON level to the HTML frontend
        self._broadcast({"defcon": level})

        # If we drop to DEFCON 1 (Air-Gapped), force the sphere into its red defcon state
        if level <= 1:
            self.set_state("defcon")

    def send_text(self, text: str):
        """Push a spoken text snippet to the avatar HUD."""
        self._broadcast({"text": text})

    def _broadcast(self, payload: dict):
        """Send JSON to all live WebSocket connections."""
        if not self._connections:
            return
        import asyncio, json
        msg = json.dumps(payload)
        dead = []
        with self._conn_lock:
            conns = list(self._connections)
        # The uvicorn server runs its own event loop in the daemon thread.
        # asyncio.get_event_loop() from a non-async context won't give us that
        # loop — we store it during _run() startup and use run_coroutine_threadsafe.
        loop = getattr(self, "_server_loop", None)
        for ws in conns:
            try:
                if loop and loop.is_running():
                    asyncio.run_coroutine_threadsafe(ws.send_text(msg), loop)
            except Exception:
                dead.append(ws)
        if dead:
            with self._conn_lock:
                for ws in dead:
                    if ws in self._connections:
                        self._connections.remove(ws)

    def stop(self):
        self._running = False
        print("✨ [ParticleAvatar] Server stopped.")

    def get_status(self) -> dict:
        return {
            "running":     self._running,
            "port":        self._port,
            "state":       self._current_state,
            "connections": len(self._connections),
            "url":         f"http://localhost:{self._port}",
        }


# ── Module-level singleton ────────────────────────────────────────────
_particle_server: Optional["ParticleAvatarServer"] = None


def get_particle_server(port: int = 8007) -> "ParticleAvatarServer":
    global _particle_server
    if _particle_server is None:
        _particle_server = ParticleAvatarServer(port=port)
    return _particle_server


if __name__ == "__main__":
    out = Path("./swayambhu_avatar.html")
    generate_particle_avatar(out)
    print(f"✅ Audio-Reactive FFT Particle avatar written to {out}")

    # Self-test: start the server
    import time
    server = ParticleAvatarServer(port=8007)
    server.start()
    time.sleep(1)
    print(f"Status: {server.get_status()}")
    server.set_state("listening")
    print("✅ ParticleAvatarServer test complete. Open http://localhost:8007")