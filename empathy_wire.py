#!/usr/bin/env python3
# =====================================================================
# 🫀 PHASE 2.2 — BIOLOGICAL TELEMETRY (Empathy Wire)  v13.1
#
# NEW in v13.1:
#   • Consent Gate — before injecting sys_override "USER_STRESSED_BE_CONCISE",
#     the system asks the user for permission. If denied (or no answer within
#     CONSENT_TIMEOUT_SEC), the override is NOT injected and BPM-concise mode
#     stays off until the next stress event fires a new consent request.
#   • Consent is remembered per session to avoid repeated interruptions
#     (re-asks only after BPM drops below threshold and spikes again).
# =====================================================================

import socket
import json
import threading
import time
import logging
from typing import Optional, Callable

logger = logging.getLogger("EmpathyWire")

# ── Config ────────────────────────────────────────────────────────────
BPM_UDP_PORT         = 9001
BPM_STRESS_THRESHOLD = 115
BPM_TIMEOUT_SEC      = 10
BPM_BROADCAST_INTERVAL = 3
CONSENT_TIMEOUT_SEC  = 15    # seconds to wait for user consent before skipping


class ConsentGate:
    """
    Handles permission requests before activating concise mode.

    Modes:
      • ask_fn provided  → calls ask_fn(message) → bool in a background thread
      • ws_broadcast_fn  → sends a JSON consent_request over WS and waits
      • neither          → falls back to a non-blocking terminal prompt

    Consent is cached per stress episode: once denied for the current
    episode, it won't ask again until BPM drops + re-spikes.
    """

    def __init__(
        self,
        ask_fn:          Optional[Callable[[str], bool]] = None,
        ws_broadcast_fn: Optional[Callable[[dict], None]] = None,
        timeout:         float = CONSENT_TIMEOUT_SEC,
    ):
        self._ask        = ask_fn
        self._broadcast  = ws_broadcast_fn
        self._timeout    = timeout

        # Consent state per episode
        self._episode_id: int    = 0         # increments each stress onset
        self._consent_given: Optional[bool] = None   # None = not asked yet
        self._consent_event = threading.Event()
        self._lock = threading.Lock()

    def new_episode(self):
        """Called when stress onset detected — resets consent state."""
        with self._lock:
            self._episode_id += 1
            self._consent_given = None
            self._consent_event.clear()

    def receive_consent(self, granted: bool):
        """Called by WS handler or external code when user responds."""
        with self._lock:
            self._consent_given = granted
        self._consent_event.set()
        logger.info(
            f"[ConsentGate] BPM-concise consent {'GRANTED ✅' if granted else 'DENIED ❌'}"
        )

    def request(self, bpm: int) -> bool:
        """
        Ask for consent to activate concise mode.
        Returns True if consent is granted, False otherwise.
        Blocks for up to self._timeout seconds.
        """
        with self._lock:
            # Already answered this episode
            if self._consent_given is not None:
                return self._consent_given

        message = (
            f"Your heart rate just spiked to {bpm} BPM. "
            f"Should I switch to concise mode to reduce information load? "
            f"(This will shorten my responses until your heart rate drops below {BPM_STRESS_THRESHOLD} BPM.)"
        )

        # WS broadcast path
        if self._broadcast:
            payload = {
                "type":    "consent_request",
                "subject": "bpm_concise_mode",
                "message": message,
                "bpm":     bpm,
                "timeout": self._timeout,
                "ts":      time.time(),
            }
            try:
                self._broadcast(payload)
                logger.info(f"[ConsentGate] Consent request sent via WS (bpm={bpm})")
            except Exception as e:
                logger.warning(f"[ConsentGate] WS broadcast failed: {e}")

            self._consent_event.clear()
            answered = self._consent_event.wait(timeout=self._timeout)

            with self._lock:
                if not answered or self._consent_given is None:
                    logger.info("[ConsentGate] No response to WS consent request — defaulting to NO")
                    self._consent_given = False
            self._consent_event.set()   # unblock any inject_biometric_override waiter
            with self._lock:
                return self._consent_given

        # ask_fn path (e.g. Qt dialog or confirm_fn)
        if self._ask:
            result = [False]
            done   = threading.Event()

            def _ask_thread():
                try:
                    result[0] = self._ask(message)
                except Exception:
                    result[0] = False
                finally:
                    done.set()

            t = threading.Thread(target=_ask_thread, daemon=True, name="EmpathyConsent")
            t.start()
            answered = done.wait(timeout=self._timeout)

            if not answered:
                logger.info("[ConsentGate] ask_fn timed out — defaulting to NO")
                result[0] = False

            with self._lock:
                self._consent_given = result[0]
            self._consent_event.set()
            return self._consent_given

        # Terminal fallback (non-blocking timeout via thread)
        result = [False]
        done   = threading.Event()

        def _terminal():
            try:
                ans = input(
                    f"\n🫀 [EmpathyWire] {message}\n"
                    f"   → Enable concise mode? (y/n, {self._timeout:.0f}s to decide): "
                ).strip().lower()
                result[0] = ans in ("y", "yes")
            except (EOFError, KeyboardInterrupt):
                result[0] = False
            finally:
                done.set()

        t = threading.Thread(target=_terminal, daemon=True, name="EmpathyConsentTerminal")
        t.start()
        answered = done.wait(timeout=self._timeout)

        if not answered:
            logger.info("[ConsentGate] Terminal consent timed out — defaulting to NO")
            result[0] = False

        with self._lock:
            self._consent_given = result[0]
        self._consent_event.set()
        logger.info(
            f"[ConsentGate] Terminal consent: {'YES' if result[0] else 'NO'}"
        )
        return self._consent_given


class EmpathyWire:
    """
    UDP listener for iOS/WatchOS BPM broadcasts.

    NEW in v13.1: Before activating "USER_STRESSED_BE_CONCISE" mode,
    a ConsentGate asks the user for permission. The override is only
    injected if consent is granted.

    Usage
    -----
        wire = EmpathyWire(
            ask_fn=lambda msg: my_confirm_dialog(msg),   # or ws_broadcast_fn=...
        )
        wire.start()
        payload = wire.inject_biometric_override(my_payload)
    """

    def __init__(
        self,
        port:               int = BPM_UDP_PORT,
        stress_threshold:   int = BPM_STRESS_THRESHOLD,
        on_bpm_update:      Optional[Callable[[int], None]] = None,
        on_stress_enter:    Optional[Callable] = None,
        on_stress_exit:     Optional[Callable] = None,
        # Consent gate configuration
        ask_fn:             Optional[Callable[[str], bool]] = None,
        ws_broadcast_fn:    Optional[Callable[[dict], None]] = None,
        consent_timeout:    float = CONSENT_TIMEOUT_SEC,
    ):
        self._port             = port
        self._threshold        = stress_threshold
        self._on_bpm           = on_bpm_update
        self._on_stress_enter  = on_stress_enter
        self._on_stress_exit   = on_stress_exit

        self._bpm: int         = 0
        self._last_rx: float   = 0.0
        self._stressed: bool   = False
        self._stop_evt         = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock             = threading.Lock()
        self.is_running        = False

        # Consent gate — guards the sys_override injection
        self._consent_gate = ConsentGate(
            ask_fn=ask_fn,
            ws_broadcast_fn=ws_broadcast_fn,
            timeout=consent_timeout,
        )
        # Whether concise mode is currently active (consent granted + stressed)
        self._concise_active: bool = False

    # ── Public API ────────────────────────────────────────────────────
    def start(self):
        if self.is_running:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="EmpathyWire"
        )
        self._thread.start()
        self.is_running = True
        logger.info(f"EmpathyWire listening on UDP :{self._port}")

    def stop(self):
        self._stop_evt.set()
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    @property
    def current_bpm(self) -> int:
        with self._lock:
            return self._bpm

    @property
    def is_stressed(self) -> bool:
        """True if BPM > threshold AND data is fresh."""
        with self._lock:
            fresh = (time.time() - self._last_rx) < BPM_TIMEOUT_SEC
            return self._stressed and fresh

    @property
    def concise_mode_active(self) -> bool:
        """True if stress detected AND user has consented to concise mode."""
        return self._concise_active and self.is_stressed

    @property
    def data_age_seconds(self) -> float:
        with self._lock:
            if self._last_rx == 0:
                return float("inf")
            return time.time() - self._last_rx

    def grant_concise_consent(self):
        """Called externally (WS handler) to grant consent."""
        self._consent_gate.receive_consent(True)

    def deny_concise_consent(self):
        """Called externally (WS handler) to deny consent."""
        self._consent_gate.receive_consent(False)

    def inject_biometric_override(self, payload: dict, wait_for_consent: bool = True,
                                      consent_wait_timeout: float = 20.0) -> dict:
        """
        Phase 2.2 core — CONSENT-GATED with blocking wait.

        If BPM > threshold AND consent not yet resolved:
          → Blocks (up to consent_wait_timeout s) for the consent thread to finish.
          → If consent is GRANTED  → injects sys_override into payload.
          → If consent is DENIED   → returns payload unchanged.
          → If timed out (no answer) → returns payload unchanged (safe default).

        If stress is not active: returns payload unchanged immediately.

        Args:
            payload:              The dict to potentially modify.
            wait_for_consent:     Block until consent is resolved (default True).
            consent_wait_timeout: Max seconds to wait (default 20s).
        """
        if not self.is_stressed:
            return payload

        # Already consented this episode
        if self._concise_active:
            payload = dict(payload)
            payload["sys_override"] = "USER_STRESSED_BE_CONCISE"
            logger.debug(
                f"🫀 [EmpathyWire] STRESSED+CONSENTED (BPM={self._bpm}) → sys_override injected"
            )
            return payload

        # Stress is active but consent not yet resolved — optionally wait
        if wait_for_consent:
            with self._consent_gate._lock:
                already_answered = self._consent_gate._consent_given is not None
            if not already_answered:
                logger.info(
                    f"[EmpathyWire] inject_biometric_override: "
                    f"waiting up to {consent_wait_timeout}s for consent resolution..."
                )
                answered = self._consent_gate._consent_event.wait(timeout=consent_wait_timeout)
                if not answered:
                    logger.info(
                        "[EmpathyWire] Consent wait timed out — not injecting override"
                    )
                    return payload
            # Re-check after wait
            if self._concise_active:
                payload = dict(payload)
                payload["sys_override"] = "USER_STRESSED_BE_CONCISE"
                logger.info(
                    f"🫀 [EmpathyWire] Consent granted after wait — sys_override injected"
                )
                return payload

        # Consent denied or not granted yet
        logger.debug(
            f"[EmpathyWire] Stressed but consent not granted — override NOT injected"
        )
        return payload

    def get_status(self) -> dict:
        return {
            "bpm":              self.current_bpm,
            "stressed":         self.is_stressed,
            "concise_active":   self.concise_mode_active,
            "data_age_sec":     round(self.data_age_seconds, 1),
            "threshold":        self._threshold,
            "port":             self._port,
            "running":          self.is_running,
        }

    # ── Internal ──────────────────────────────────────────────────────
    def _listen_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)

        try:
            sock.bind(("0.0.0.0", self._port))
        except OSError as e:
            logger.error(f"EmpathyWire bind failed on port {self._port}: {e}")
            self.is_running = False
            return

        logger.info(f"🫀 EmpathyWire bound to UDP 0.0.0.0:{self._port}")

        while not self._stop_evt.is_set():
            try:
                data, addr = sock.recvfrom(256)
            except socket.timeout:
                continue
            except Exception as e:
                logger.warning(f"EmpathyWire recv error: {e}")
                continue
            self._handle_packet(data, addr)

        sock.close()
        logger.info("EmpathyWire UDP socket closed.")

    def _handle_packet(self, data: bytes, addr):
        try:
            text = data.decode("utf-8").strip()
            try:
                obj = json.loads(text)
                bpm = int(obj.get("bpm", obj.get("heartRate", 0)))
            except (json.JSONDecodeError, ValueError):
                bpm = int(text)

            if bpm <= 0 or bpm > 300:
                return

            with self._lock:
                prev_stressed = self._stressed
                self._bpm     = bpm
                self._last_rx = time.time()
                self._stressed = bpm > self._threshold

                stress_onset  = (not prev_stressed) and self._stressed
                stress_clear  = prev_stressed and (not self._stressed)

            logger.debug(f"🫀 BPM={bpm} from {addr[0]} stressed={self._stressed}")

            if self._on_bpm:
                try:
                    self._on_bpm(bpm)
                except Exception:
                    pass

            if stress_onset:
                # ── NEW: Consent Gate ─────────────────────────────────
                logger.info(f"🚨 [EmpathyWire] STRESS ONSET (BPM={bpm}) — requesting consent")
                self._consent_gate.new_episode()

                if self._on_stress_enter:
                    try:
                        self._on_stress_enter()
                    except Exception:
                        pass

                # Request consent in a background thread so UDP loop keeps running
                def _ask_consent():
                    granted = self._consent_gate.request(bpm)
                    self._concise_active = granted
                    if granted:
                        logger.info(
                            f"💬 [EmpathyWire] Concise mode ACTIVATED (consent granted, BPM={bpm})"
                        )
                    else:
                        logger.info(
                            f"💬 [EmpathyWire] Concise mode SKIPPED (consent denied, BPM={bpm})"
                        )

                threading.Thread(target=_ask_consent, daemon=True,
                                 name="EmpathyConsent").start()

            if stress_clear:
                self._concise_active = False
                logger.info(f"💚 [EmpathyWire] Stress cleared (BPM={bpm}) — concise mode OFF")
                if self._on_stress_exit:
                    try:
                        self._on_stress_exit()
                    except Exception:
                        pass

        except Exception as e:
            logger.warning(f"EmpathyWire packet parse error: {e} | raw={data!r}")


# ── Broadcast simulator ───────────────────────────────────────────────
def simulate_bpm_broadcast(
    bpm:      int   = 80,
    port:     int   = BPM_UDP_PORT,
    host:     str   = "127.0.0.1",
    count:    int   = 5,
    interval: float = BPM_BROADCAST_INTERVAL,
):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(count):
        payload = json.dumps({"bpm": bpm, "timestamp": time.time()}).encode()
        sock.sendto(payload, (host, port))
        logger.info(f"[Simulator] Sent BPM={bpm} to {host}:{port}")
        if i < count - 1:
            time.sleep(interval)
    sock.close()


# ── Module-level singleton ────────────────────────────────────────────
_wire: Optional[EmpathyWire] = None


def get_empathy_wire() -> EmpathyWire:
    global _wire
    if _wire is None:
        _wire = EmpathyWire()
    return _wire


def start_empathy_wire(**kwargs) -> EmpathyWire:
    global _wire
    _wire = EmpathyWire(**kwargs)
    _wire.start()
    return _wire


# ── Self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)
    print("🫀 EmpathyWire v13.1 (Consent Gate) self-test\n")

    # ── ConsentGate unit tests ─────────────────────────────────────────
    print("=== ConsentGate ===")

    gate_yes = ConsentGate(ask_fn=lambda _: True, timeout=2.0)
    gate_yes.new_episode()
    result = gate_yes.request(bpm=130)
    assert result is True, f"Expected True: {result}"
    print("  ✅ ask_fn=yes → consent granted")

    gate_no = ConsentGate(ask_fn=lambda _: False, timeout=2.0)
    gate_no.new_episode()
    result2 = gate_no.request(bpm=130)
    assert result2 is False, f"Expected False: {result2}"
    print("  ✅ ask_fn=no → consent denied")

    # ── EmpathyWire consent-gated injection ───────────────────────────
    print("\n=== EmpathyWire with consent gate ===")
    bpm_log = []

    # Wire with auto-grant consent
    wire_grant = EmpathyWire(
        port=19880, stress_threshold=115,
        ask_fn=lambda _: True,           # auto-grant
        on_bpm_update=lambda b: bpm_log.append(b),
    )
    wire_grant.start()
    time.sleep(0.3)

    # Calm BPM — no injection regardless
    simulate_bpm_broadcast(bpm=72, port=19880, count=2, interval=0.2)
    time.sleep(0.6)
    assert wire_grant.current_bpm == 72
    payload = wire_grant.inject_biometric_override({"cmd": "test"})
    assert "sys_override" not in payload, "No override when calm"
    print("  ✅ Calm BPM → no override injected")

    # Stress BPM — should request consent → auto-granted → concise active after short delay
    simulate_bpm_broadcast(bpm=130, port=19880, count=2, interval=0.2)
    time.sleep(2.0)   # allow consent thread to complete
    assert wire_grant.current_bpm == 130
    assert wire_grant.is_stressed
    # Consent should be granted by ask_fn=True
    assert wire_grant.concise_mode_active, "Concise mode should be active after consent"
    payload2 = wire_grant.inject_biometric_override({"cmd": "analyze"})
    assert payload2.get("sys_override") == "USER_STRESSED_BE_CONCISE", \
        f"Expected sys_override after consent: {payload2}"
    print("  ✅ Stressed + consented → sys_override injected")

    wire_grant.stop()

    # Wire with auto-deny consent
    wire_deny = EmpathyWire(
        port=19881, stress_threshold=115,
        ask_fn=lambda _: False,           # auto-deny
    )
    wire_deny.start()
    time.sleep(0.3)

    simulate_bpm_broadcast(bpm=130, port=19881, count=2, interval=0.2)
    time.sleep(2.0)
    assert wire_deny.is_stressed
    assert not wire_deny.concise_mode_active, "Concise mode should NOT be active after denial"
    payload3 = wire_deny.inject_biometric_override({"cmd": "analyze"})
    assert "sys_override" not in payload3, "No override when denied"
    print("  ✅ Stressed + denied → no sys_override injected")

    wire_deny.stop()
    print("\n✅ All EmpathyWire v13.1 tests passed.")
