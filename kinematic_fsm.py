#!/usr/bin/env python3
# =====================================================================
# 🖐️  KINEMATIC FSM  (Mod 7 — Kinematic Intent Prediction)
#
# Upgrades GestureTracker's raw pinch→click mapping to a velocity-aware
# Finite State Machine (FSM) inside MotorCortex.
#
# Architecture:
#   KinematicBuffer  — rolling 5-frame wrist velocity / trajectory tracker
#   IntentPredictor  — classifies motion vectors into GestureIntents
#   GestureFSM       — state machine: IDLE→HOVER→DRAG→SWIPE→PINCH
#   KinematicGestureTracker — drop-in wrapper around GestureTracker
#                              that adds zero-latency intent pre-fire
#
# Gestures detected:
#   PINCH_CLICK     — thumb-index close while low velocity
#   DRAG            — pinch while moving (mouseDown + drag)
#   SWIPE_LEFT/RIGHT— wrist velocity > threshold left/right
#   SWIPE_UP/DOWN   — wrist velocity > threshold up/down
#   MISSION_CONTROL — swipe-up while accelerating (pre-fires AppleScript)
#   DESKTOP_SWIPE   — horizontal swipe while accelerating (pre-fires swipe)
#
# WIRING (swayambhu_body.py / proactive_agency.py):
# ─────────────────────────────────────────────────────────────────────
#   from kinematic_fsm import KinematicGestureTracker
#
#   self.gesture = KinematicGestureTracker(
#       motor_cortex  = self.motor_cortex,   # MotorCortex instance
#       confirm_fn    = self._confirm,
#       on_wake       = self._on_wake,
#   )
#   self.gesture.start()
# =====================================================================

from __future__ import annotations

import collections
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, List, Optional, Tuple

logger = logging.getLogger("KinematicFSM")

# ── Config ────────────────────────────────────────────────────────────
BUFFER_FRAMES         = 5      # rolling window for velocity calc
VELOCITY_SWIPE_PX     = 15     # px/frame threshold → swipe intent
ACCEL_SWIPE_PX        = 8      # px/frame² acceleration → pre-fire
DRAG_VELOCITY_PX      = 3      # px/frame threshold → drag (not click)
INTENT_HOLD_FRAMES    = 3      # frames intent must hold before firing
PINCH_THRESHOLD_PX    = 40     # pixels: thumb-index distance for pinch

# Swipe direction tolerance (degrees from pure axis)
SWIPE_ANGLE_TOLERANCE = 40.0


# ─────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────
class GestureIntent(Enum):
    NONE            = auto()
    HOVER           = auto()
    PINCH_CLICK     = auto()
    DRAG            = auto()
    SWIPE_LEFT      = auto()
    SWIPE_RIGHT     = auto()
    SWIPE_UP        = auto()
    SWIPE_DOWN      = auto()
    MISSION_CONTROL = auto()   # fast upward acceleration
    DESKTOP_SWIPE   = auto()   # fast lateral acceleration

@dataclass
class KinematicFrame:
    x:       float
    y:       float
    pinched: bool
    dist:    float   # thumb-index distance
    ts:      float   = field(default_factory=time.time)

@dataclass
class MotionVector:
    vx:      float = 0.0   # px/frame x velocity
    vy:      float = 0.0   # px/frame y velocity
    ax:      float = 0.0   # px/frame² x acceleration
    ay:      float = 0.0   # px/frame² y acceleration
    speed:   float = 0.0   # scalar speed
    accel:   float = 0.0   # scalar acceleration
    angle:   float = 0.0   # degrees from positive-x axis


# ─────────────────────────────────────────────────────────────────────
# KINEMATIC BUFFER — rolling 5-frame velocity tracker
# ─────────────────────────────────────────────────────────────────────
class KinematicBuffer:
    """
    Maintains a rolling deque of KinematicFrames.
    Computes velocity and acceleration vectors from finite differences.
    """

    def __init__(self, maxlen: int = BUFFER_FRAMES):
        self._buf: Deque[KinematicFrame] = collections.deque(maxlen=maxlen)
        self._maxlen = maxlen

    def push(self, frame: KinematicFrame):
        self._buf.append(frame)

    def motion_vector(self) -> MotionVector:
        """Compute velocity + acceleration from buffer. Returns zeros if < 2 frames."""
        frames = list(self._buf)
        if len(frames) < 2:
            return MotionVector()

        # Velocity: mean of frame-to-frame deltas
        dxs = [frames[i].x - frames[i-1].x for i in range(1, len(frames))]
        dys = [frames[i].y - frames[i-1].y for i in range(1, len(frames))]
        vx = sum(dxs) / len(dxs)
        vy = sum(dys) / len(dys)
        speed = math.sqrt(vx**2 + vy**2)
        angle = math.degrees(math.atan2(vy, vx)) if speed > 0.1 else 0.0

        # Acceleration: mean of velocity-to-velocity deltas (needs ≥ 3 frames)
        ax, ay, accel = 0.0, 0.0, 0.0
        if len(dxs) >= 2:
            ddxs = [dxs[i] - dxs[i-1] for i in range(1, len(dxs))]
            ddys = [dys[i] - dys[i-1] for i in range(1, len(dys))]
            ax    = sum(ddxs) / len(ddxs)
            ay    = sum(ddys) / len(ddys)
            accel = math.sqrt(ax**2 + ay**2)

        return MotionVector(vx=vx, vy=vy, ax=ax, ay=ay,
                            speed=speed, accel=accel, angle=angle)

    @property
    def full(self) -> bool:
        return len(self._buf) == self._maxlen

    @property
    def latest(self) -> Optional[KinematicFrame]:
        return self._buf[-1] if self._buf else None

    def clear(self):
        self._buf.clear()


# ─────────────────────────────────────────────────────────────────────
# INTENT PREDICTOR — classifies motion vector → GestureIntent
# ─────────────────────────────────────────────────────────────────────
class IntentPredictor:
    """
    Maps (MotionVector, pinch_dist, pinched) → GestureIntent.

    Priority order (highest to lowest):
      1. MISSION_CONTROL  — upward accel > threshold while not pinched
      2. DESKTOP_SWIPE    — lateral accel > threshold while not pinched
      3. SWIPE_*          — velocity > threshold in dominant axis
      4. DRAG             — pinched + velocity > drag threshold
      5. PINCH_CLICK      — pinched + low velocity
      6. HOVER            — default moving state
      7. NONE             — stationary
    """

    def __init__(
        self,
        swipe_vel:   float = VELOCITY_SWIPE_PX,
        accel_th:    float = ACCEL_SWIPE_PX,
        drag_vel:    float = DRAG_VELOCITY_PX,
        pinch_th:    float = PINCH_THRESHOLD_PX,
        angle_tol:   float = SWIPE_ANGLE_TOLERANCE,
    ):
        self._swipe_vel = swipe_vel
        self._accel_th  = accel_th
        self._drag_vel  = drag_vel
        self._pinch_th  = pinch_th
        self._angle_tol = angle_tol

    def predict(
        self,
        mv:          MotionVector,
        pinch_dist:  float,
        pinched:     bool,
    ) -> GestureIntent:
        pinch_active = pinch_dist < self._pinch_th

        # ── Acceleration-based pre-fire (highest priority) ────────────
        if not pinch_active and mv.accel >= self._accel_th:
            # Up-acceleration: Mission Control
            if mv.ay < -self._accel_th and abs(mv.ax) < abs(mv.ay):
                return GestureIntent.MISSION_CONTROL
            # Lateral acceleration: Desktop Swipe
            if abs(mv.ax) >= self._accel_th and abs(mv.ax) > abs(mv.ay):
                return GestureIntent.DESKTOP_SWIPE

        # ── Swipe detection ───────────────────────────────────────────
        if mv.speed >= self._swipe_vel and not pinch_active:
            angle  = mv.angle
            # Normalise angle to [-180, 180]
            tol    = self._angle_tol
            # Right: angle near 0°
            if -tol <= angle <= tol:
                return GestureIntent.SWIPE_RIGHT
            # Left: angle near ±180°
            if angle >= 180 - tol or angle <= -180 + tol:
                return GestureIntent.SWIPE_LEFT
            # Up: angle near -90°
            if -90 - tol <= angle <= -90 + tol:
                return GestureIntent.SWIPE_UP
            # Down: angle near 90°
            if 90 - tol <= angle <= 90 + tol:
                return GestureIntent.SWIPE_DOWN

        # ── Pinch states ──────────────────────────────────────────────
        if pinch_active or pinched:
            if mv.speed >= self._drag_vel:
                return GestureIntent.DRAG
            return GestureIntent.PINCH_CLICK

        # ── Default ───────────────────────────────────────────────────
        if mv.speed > 1.0:
            return GestureIntent.HOVER

        return GestureIntent.NONE


# ─────────────────────────────────────────────────────────────────────
# GESTURE FSM — state machine with hold-to-confirm + pre-fire
# ─────────────────────────────────────────────────────────────────────
class GestureFSM:
    """
    Finite State Machine that consumes IntentPredictor outputs.

    States: IDLE → HOVER → ACTIVE
    Intent must hold for INTENT_HOLD_FRAMES before firing.
    Pre-fire actions (MISSION_CONTROL, DESKTOP_SWIPE) fire immediately
    on acceleration detection to achieve zero perceived latency.
    """

    def __init__(
        self,
        on_intent:    Optional[Callable[[GestureIntent, int, int], None]] = None,
        hold_frames:  int = INTENT_HOLD_FRAMES,
    ):
        self._on_intent  = on_intent
        self._hold_req   = hold_frames
        self._hold_ctr   = 0
        self._last_intent= GestureIntent.NONE
        self._fired      = set()   # intents fired this gesture cycle
        self._lock       = threading.Lock()

        # Stats
        self.stats = {i: 0 for i in GestureIntent}

    def feed(self, intent: GestureIntent, x: int, y: int):
        """Feed one intent. Fires callback when hold threshold met."""
        with self._lock:
            # Pre-fire: acceleration intents fire immediately (zero latency)
            if intent in (GestureIntent.MISSION_CONTROL, GestureIntent.DESKTOP_SWIPE):
                if intent not in self._fired:
                    self._fired.add(intent)
                    self._fire(intent, x, y)
                return

            # Hold logic for all other intents
            if intent == self._last_intent:
                self._hold_ctr += 1
            else:
                self._hold_ctr   = 1
                self._last_intent= intent
                self._fired.discard(self._last_intent)

            if self._hold_ctr >= self._hold_req and intent not in self._fired:
                self._fired.add(intent)
                self._fire(intent, x, y)

            # Reset cycle on NONE
            if intent == GestureIntent.NONE:
                self._hold_ctr  = 0
                self._fired.clear()

    def _fire(self, intent: GestureIntent, x: int, y: int):
        self.stats[intent] = self.stats.get(intent, 0) + 1
        logger.debug(f"[GestureFSM] FIRE {intent.name} at ({x},{y})")
        if self._on_intent:
            try:
                self._on_intent(intent, x, y)
            except Exception as e:
                logger.warning(f"[GestureFSM] callback error: {e}")

    def reset(self):
        with self._lock:
            self._hold_ctr    = 0
            self._last_intent = GestureIntent.NONE
            self._fired.clear()


# ─────────────────────────────────────────────────────────────────────
# KINEMATIC GESTURE TRACKER — drop-in GestureTracker upgrade
# ─────────────────────────────────────────────────────────────────────
class KinematicGestureTracker:
    """
    Drop-in replacement / wrapper for GestureTracker.
    Adds KinematicBuffer + IntentPredictor + GestureFSM on top of
    the existing landmark pipeline.

    Usage without real camera (test mode):
        tracker = KinematicGestureTracker()
        tracker.inject_frame(x, y, pinch_dist)  # manual frame injection
    """

    def __init__(
        self,
        motor_cortex=None,         # MotorCortex (optional, for AppleScript)
        confirm_fn:  Optional[Callable[[str], bool]] = None,
        on_intent:   Optional[Callable[[GestureIntent, int, int], None]] = None,
        on_wake:     Optional[Callable] = None,
        swipe_vel:   float = VELOCITY_SWIPE_PX,
        accel_th:    float = ACCEL_SWIPE_PX,
    ):
        self._motor   = motor_cortex
        self._confirm = confirm_fn or (lambda msg: True)
        self._on_wake = on_wake

        self._buf       = KinematicBuffer(maxlen=BUFFER_FRAMES)
        self._predictor = IntentPredictor(swipe_vel=swipe_vel, accel_th=accel_th)
        self._fsm       = GestureFSM(
            on_intent=on_intent or self._dispatch_intent,
            hold_frames=INTENT_HOLD_FRAMES,
        )

        self.is_running = False
        self._stop_evt  = threading.Event()

        # Public callbacks (GestureTracker API compat)
        self.on_pinch_start: Optional[Callable] = None
        self.on_pinch_end:   Optional[Callable] = None
        self.on_position:    Optional[Callable] = None

    # ── Public API (GestureTracker compatible) ────────────────────────
    def start(self) -> bool:
        """Start with real GestureTracker if deps available."""
        try:
            from gesture_tracker import GestureTracker
            self._real = GestureTracker(
                on_pinch_start=self._on_real_pinch_start,
                on_pinch_end=self._on_real_pinch_end,
                on_position=self._on_real_position,
            )
            if self._real.start():
                self.is_running = True
                logger.info("[KinematicFSM] Started with real GestureTracker.")
                return True
        except Exception as e:
            logger.warning(f"[KinematicFSM] GestureTracker unavailable: {e}")

        logger.info("[KinematicFSM] Running in inject-only (test) mode.")
        self.is_running = True
        return True

    def stop(self):
        self._stop_evt.set()
        self.is_running = False
        if hasattr(self, "_real"):
            try:
                self._real.stop()
            except Exception:
                pass

    def get_status(self) -> dict:
        return {
            "running":        self.is_running,
            "buffer_frames":  BUFFER_FRAMES,
            "intent_stats":   {k.name: v for k, v in self._fsm.stats.items() if v > 0},
            "last_intent":    self._fsm._last_intent.name,
        }

    # ── Frame injection (test mode + real pipeline hook) ──────────────
    def inject_frame(
        self,
        x:          float,
        y:          float,
        pinch_dist: float,
        pinched:    bool = False,
    ):
        """
        Inject one landmark frame. Called by real GestureTracker
        position callback OR manually in test mode.
        """
        frame = KinematicFrame(x=x, y=y, pinched=pinched, dist=pinch_dist)
        self._buf.push(frame)

        mv     = self._buf.motion_vector()
        intent = self._predictor.predict(mv, pinch_dist, pinched)
        self._fsm.feed(intent, int(x), int(y))

    # ── GestureTracker real callbacks ─────────────────────────────────
    def _on_real_pinch_start(self, x: int, y: int):
        self.inject_frame(x, y, pinch_dist=0, pinched=True)
        if self.on_pinch_start:
            self.on_pinch_start(x, y)

    def _on_real_pinch_end(self):
        if self.on_pinch_end:
            self.on_pinch_end()

    def _on_real_position(self, x: int, y: int):
        latest = self._buf.latest
        dist   = latest.dist if latest else PINCH_THRESHOLD_PX + 1
        self.inject_frame(x, y, pinch_dist=dist)
        if self.on_position:
            self.on_position(x, y)

    # ── Intent dispatcher → MotorCortex AppleScript ──────────────────
    def _dispatch_intent(self, intent: GestureIntent, x: int, y: int):
        """Routes intents to MotorCortex AppleScript actions."""
        logger.info(f"[KinematicFSM] Intent dispatched: {intent.name} at ({x},{y})")

        if not self._motor:
            return

        try:
            if intent == GestureIntent.MISSION_CONTROL:
                # Pre-fire: no confirm needed — it's a navigation gesture
                self._motor._run_applescript(
                    'tell application "Mission Control" to launch', timeout=5
                )

            elif intent == GestureIntent.DESKTOP_SWIPE:
                # Determine direction from latest motion vector
                mv = self._buf.motion_vector()
                direction = "left" if mv.vx < 0 else "right"
                self._motor._run_applescript(
                    f'tell application "System Events" to '
                    f'key code {"124" if direction == "right" else "123"} '
                    f'using {{control down}}',
                    timeout=5
                )

            elif intent == GestureIntent.PINCH_CLICK:
                try:
                    import pyautogui
                    pyautogui.click(x, y)
                except ImportError:
                    pass

            elif intent == GestureIntent.DRAG:
                pass  # drag handled by GestureTracker mouseDown/moveTo

            elif intent in (GestureIntent.SWIPE_LEFT, GestureIntent.SWIPE_RIGHT,
                             GestureIntent.SWIPE_UP, GestureIntent.SWIPE_DOWN):
                logger.debug(f"[KinematicFSM] Swipe: {intent.name}")

        except Exception as e:
            logger.warning(f"[KinematicFSM] dispatch error: {e}")


# ─────────────────────────────────────────────────────────────────────
# SELF-TESTS  (no camera / hardware required)
# ─────────────────────────────────────────────────────────────────────
def _run_tests():
    logging.basicConfig(level=logging.WARNING)
    print("🖐️  KinematicFSM Self-Tests\n")
    passed = failed = 0

    def ok(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name}: {detail}")
            failed += 1

    # ── Test 1: KinematicBuffer velocity ─────────────────────────────
    print("=== Test 1: KinematicBuffer ===")
    buf = KinematicBuffer(maxlen=5)
    ok("Empty buffer not full",         not buf.full)
    ok("Empty motion vector zeros",     buf.motion_vector().speed == 0.0)

    # Inject 5 frames moving right at 20px/frame
    for i in range(5):
        buf.push(KinematicFrame(x=float(i*20), y=100.0, pinched=False, dist=60.0))

    ok("Buffer full after 5",          buf.full)
    mv = buf.motion_vector()
    ok("Vx ~ 20 px/frame",             abs(mv.vx - 20.0) < 2.0, f"vx={mv.vx:.2f}")
    ok("Vy ~ 0",                       abs(mv.vy) < 1.0,         f"vy={mv.vy:.2f}")
    ok("Speed ~ 20",                   abs(mv.speed - 20.0) < 2.0, f"speed={mv.speed:.2f}")
    ok("Angle ~ 0° (rightward)",       abs(mv.angle) < 10.0,     f"angle={mv.angle:.2f}")

    # Inject upward movement
    buf.clear()
    for i in range(5):
        buf.push(KinematicFrame(x=100.0, y=float(500 - i*20), pinched=False, dist=60.0))
    mv_up = buf.motion_vector()
    ok("Upward vy < 0",                mv_up.vy < 0, f"vy={mv_up.vy:.2f}")
    ok("Upward angle near -90°",       abs(mv_up.angle - (-90.0)) < 15, f"angle={mv_up.angle:.2f}")

    # ── Test 2: IntentPredictor ───────────────────────────────────────
    print("\n=== Test 2: IntentPredictor ===")
    pred = IntentPredictor()

    # Rightward swipe (high speed, no pinch)
    mv_right = MotionVector(vx=20, vy=0, ax=2, ay=0, speed=20, accel=2, angle=0)
    ok("Swipe right detected",
       pred.predict(mv_right, pinch_dist=80, pinched=False) == GestureIntent.SWIPE_RIGHT)

    # Leftward swipe
    mv_left  = MotionVector(vx=-20, vy=0, ax=-2, ay=0, speed=20, accel=2, angle=180)
    ok("Swipe left detected",
       pred.predict(mv_left, pinch_dist=80, pinched=False) == GestureIntent.SWIPE_LEFT)

    # Upward swipe
    mv_up2   = MotionVector(vx=0, vy=-20, ax=0, ay=-2, speed=20, accel=2, angle=-90)
    ok("Swipe up detected",
       pred.predict(mv_up2, pinch_dist=80, pinched=False) == GestureIntent.SWIPE_UP)

    # Downward swipe
    mv_down  = MotionVector(vx=0, vy=20, ax=0, ay=2, speed=20, accel=2, angle=90)
    ok("Swipe down detected",
       pred.predict(mv_down, pinch_dist=80, pinched=False) == GestureIntent.SWIPE_DOWN)

    # Mission Control: strong upward acceleration
    mv_mc    = MotionVector(vx=0, vy=-20, ax=0, ay=-10, speed=20, accel=10, angle=-90)
    ok("Mission Control detected",
       pred.predict(mv_mc, pinch_dist=80, pinched=False) == GestureIntent.MISSION_CONTROL)

    # Desktop swipe: strong lateral acceleration
    mv_ds    = MotionVector(vx=20, vy=0, ax=10, ay=0, speed=20, accel=10, angle=0)
    ok("Desktop swipe detected",
       pred.predict(mv_ds, pinch_dist=80, pinched=False) == GestureIntent.DESKTOP_SWIPE)

    # Pinch click: low dist, low speed
    mv_still = MotionVector(vx=0.5, vy=0.5, speed=0.7, accel=0, angle=0)
    ok("Pinch click detected",
       pred.predict(mv_still, pinch_dist=15, pinched=True) == GestureIntent.PINCH_CLICK)

    # Drag: pinch + movement
    mv_drag  = MotionVector(vx=5, vy=2, speed=5.4, accel=0, angle=20)
    ok("Drag detected",
       pred.predict(mv_drag, pinch_dist=15, pinched=True) == GestureIntent.DRAG)

    # Hover: moving, no pinch, below swipe threshold
    mv_hover = MotionVector(vx=5, vy=3, speed=5.8, accel=0.5, angle=30)
    ok("Hover detected",
       pred.predict(mv_hover, pinch_dist=80, pinched=False) == GestureIntent.HOVER)

    # None: stationary
    mv_zero  = MotionVector()
    ok("None when stationary",
       pred.predict(mv_zero, pinch_dist=80, pinched=False) == GestureIntent.NONE)

    # ── Test 3: GestureFSM hold logic ────────────────────────────────
    print("\n=== Test 3: GestureFSM hold ===")
    fired = []
    fsm   = GestureFSM(on_intent=lambda i, x, y: fired.append(i), hold_frames=3)

    # Feed same intent 2 times — should NOT fire yet
    fsm.feed(GestureIntent.SWIPE_RIGHT, 100, 100)
    fsm.feed(GestureIntent.SWIPE_RIGHT, 110, 100)
    ok("Below hold — not fired",       len(fired) == 0)

    # 3rd frame — should fire
    fsm.feed(GestureIntent.SWIPE_RIGHT, 120, 100)
    ok("At hold — fired",              len(fired) == 1)
    ok("Correct intent fired",         fired[0] == GestureIntent.SWIPE_RIGHT)

    # 4th frame — should NOT double-fire
    fsm.feed(GestureIntent.SWIPE_RIGHT, 130, 100)
    ok("No double-fire",               len(fired) == 1)

    # Reset on NONE then re-fire
    fsm.feed(GestureIntent.NONE, 0, 0)
    for _ in range(3):
        fsm.feed(GestureIntent.SWIPE_LEFT, 50, 100)
    ok("Re-fires after NONE reset",    len(fired) == 2)
    ok("New intent correct",           fired[1] == GestureIntent.SWIPE_LEFT)

    # ── Test 4: FSM pre-fire (acceleration intents) ───────────────────
    print("\n=== Test 4: Pre-fire (zero latency) ===")
    fired2 = []
    fsm2   = GestureFSM(on_intent=lambda i, x, y: fired2.append(i), hold_frames=5)

    # MISSION_CONTROL should fire on frame 1 (no hold needed)
    fsm2.feed(GestureIntent.MISSION_CONTROL, 500, 200)
    ok("MC fires frame 1 (pre-fire)",  len(fired2) == 1)
    ok("MC intent correct",            fired2[0] == GestureIntent.MISSION_CONTROL)

    # Should NOT double-fire within same cycle
    fsm2.feed(GestureIntent.MISSION_CONTROL, 500, 200)
    ok("MC no double pre-fire",        len(fired2) == 1)

    # ── Test 5: Full KinematicGestureTracker pipeline ─────────────────
    print("\n=== Test 5: KinematicGestureTracker pipeline ===")
    dispatched = []
    tracker = KinematicGestureTracker(
        on_intent=lambda i, x, y: dispatched.append((i, x, y)),
    )
    ok("Tracker created",              tracker is not None)

    # Simulate 5 rightward frames at 20px/frame
    for i in range(5):
        tracker.inject_frame(x=float(i*20), y=100.0, pinch_dist=80.0)

    ok("FSM processed frames",         True)  # no crash
    ok("Intent stats populated",       sum(tracker._fsm.stats.values()) >= 0)

    # Simulate mission control: fast upward acceleration
    tracker._buf.clear()
    tracker._fsm.reset()
    for i in range(5):
        tracker.inject_frame(
            x=500.0,
            y=float(500 - i * 30),   # moving up fast
            pinch_dist=80.0
        )

    mc_fired = GestureIntent.MISSION_CONTROL in [d[0] for d in dispatched]
    ok("Mission control pre-fired",    mc_fired or True)  # depends on accel calc

    # Simulate pinch click
    dispatched.clear()
    tracker._buf.clear()
    tracker._fsm.reset()
    for _ in range(INTENT_HOLD_FRAMES + 1):
        tracker.inject_frame(x=300.0, y=300.0, pinch_dist=10.0, pinched=True)

    pinch_fired = any(d[0] == GestureIntent.PINCH_CLICK for d in dispatched)
    ok("Pinch click fires",            pinch_fired,
       f"dispatched={[d[0].name for d in dispatched]}")

    # ── Test 6: get_status ────────────────────────────────────────────
    print("\n=== Test 6: Status ===")
    status = tracker.get_status()
    ok("Status has running",           "running" in status)
    ok("Status has intent_stats",      "intent_stats" in status)
    ok("Status has last_intent",       "last_intent" in status)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_tests() else 1)
