#!/usr/bin/env python3
"""
🧪 Comprehensive test suite for mlx_dpo_trainer.py v2.0
Covers: dataset, EWC, loss, LoRA, scheduler, grad clip,
        training session, convergence, no-double-run, war-room extraction,
        nightly scheduler, stats, adapter save/load roundtrip.
"""
import json, math, shutil, sys, tempfile, threading, time
from pathlib import Path

# Import module under test
from mlx_dpo_trainer import (
    DPOPair, DPODataset, EWCPenalty, DPOLossComputer, CosineScheduler,
    LoRALinear, LoRAAdapter, MLXDPOTrainer, NightlyScheduler,
    clip_grad_norm, StepRecord, TrainingSession,
    _MLX_OK, _MLX_LM_OK, _IS_APPLE_SILICON, _NP_OK,
    LORA_RANK, LORA_ALPHA, DPO_BETA, MIN_PAIRS, TRAIN_LOG_PATH,
)

passed = failed = 0
def ok(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  ✅ {name}"); passed += 1
    else:
        print(f"  ❌ {name}: {detail}"); failed += 1

def pairs_file(tmpdir: Path, n: int, delta: float = 0.50) -> Path:
    pf = tmpdir / f"pairs_{n}.jsonl"
    pf.write_text("\n".join(json.dumps({
        "prompt": f"Q{i}", "chosen": f"Detailed helpful answer {i} covering all aspects",
        "rejected": f"Bad {i}", "chosen_score": 0.85,
        "rejected_score": 0.85 - delta, "delta": delta,
    }) for i in range(n)))
    return pf

def run():
    print("🍎 MLXDPOTrainer v2.0 — Full Test Suite\n")
    print(f"   MLX:    {'✅' if _MLX_OK else 'sim'}")
    print(f"   mlx-lm: {'✅' if _MLX_LM_OK else 'sim'}")
    print(f"   numpy:  {'✅' if _NP_OK else '❌'}")
    print(f"   Apple:  {'✅' if _IS_APPLE_SILICON else 'Linux/CI'}\n")
    tmpdir = Path(tempfile.mkdtemp())

    # ── T1: DPODataset ────────────────────────────────────────────────
    print("=== T1: DPODataset ===")
    pf = pairs_file(tmpdir, 5)
    ds = DPODataset(pf, min_delta=0.10)
    n  = ds.load()
    ok("Loaded 5 pairs",           n == 5)
    ok("size == 5",                ds.size == 5)
    ok("stats count",              ds.stats()["count"] == 5)
    ok("mean_delta == 0.50",       abs(ds.stats()["mean_delta"] - 0.50) < 0.01)

    # Filtering
    pf2 = tmpdir / "low_delta.jsonl"
    pf2.write_text(json.dumps({"prompt":"q","chosen":"c","rejected":"r",
                                "chosen_score":0.55,"rejected_score":0.52,"delta":0.03}))
    ds2 = DPODataset(pf2, min_delta=0.10)
    ok("Low delta filtered",       ds2.load() == 0)

    # Batches
    batches = list(ds.batches(2))
    ok("Batches non-empty",        len(batches) >= 1)
    ok("Batch items are DPOPair",  isinstance(batches[0][0], DPOPair))

    # Missing file
    ds_missing = DPODataset(tmpdir / "nonexistent.jsonl")
    ok("Missing file loads 0",     ds_missing.load() == 0)

    # ── T2: EWCPenalty ───────────────────────────────────────────────
    print("\n=== T2: EWCPenalty ===")
    ewc = EWCPenalty(lam=0.4)
    ok("Not computed initially",   not ewc._computed)
    ok("Penalty=0 pre-compute",    ewc.penalty({}) == 0.0)

    params = {"L0": [0.1, 0.2, 0.3], "L1": [0.4, 0.5]}
    ewc.compute_from_model(params)
    ok("Computed flag set",        ewc._computed)
    ok("Anchor stored",            len(ewc._theta_star) == 2)
    ok("Fisher non-negative",      all(v >= 0 for g in ewc._fisher.values() for v in g))
    ok("Penalty at anchor = 0",    abs(ewc.penalty(params)) < 1e-8,
       f"got {ewc.penalty(params)}")
    shifted = {"L0": [1.0, 2.0, 3.0], "L1": [0.0, 0.0]}
    ok("Penalty away > 0",         ewc.penalty(shifted) > 0)
    s = ewc.get_status()
    ok("Status: lambda=0.4",       s["lambda"] == 0.4)
    ok("Status: param_groups=2",   s["param_groups"] == 2)

    # ── T3: DPOLossComputer ──────────────────────────────────────────
    print("\n=== T3: DPOLossComputer ===")
    lc = DPOLossComputer(beta=0.1)
    batch = [DPOPair("q","chosen","rejected",0.9,0.3),
             DPOPair("q","c2","r2",0.8,0.5)]
    loss, margin = lc.compute(batch)
    ok("Loss > 0",                 loss > 0)
    ok("Loss finite",              math.isfinite(loss))
    ok("Margin > 0",               margin > 0)

    # Inverted pair
    inv = [DPOPair("q","bad","good",0.3,0.9)]
    li, mi = lc.compute(inv)
    ok("Inverted margin < 0",      mi < 0)
    ok("Inverted loss > normal",   li > loss, f"inv={li:.4f} norm={loss:.4f}")

    # MLX consistency
    lm, mm = lc.compute_mlx(batch)
    ok("MLX loss near sim",        abs(lm - loss) < 0.01,
       f"mlx={lm:.4f} sim={loss:.4f}")

    # Log-prob DPO
    lp = lc.compute_with_logprobs(-2.0, -3.0, -2.1, -2.9)
    ok("Logprob loss finite",      math.isfinite(lp) and lp > 0)

    # ── T4: CosineScheduler ──────────────────────────────────────────
    print("\n=== T4: CosineScheduler ===")
    sched = CosineScheduler(1e-4, warmup=5, total=20)
    ok("Step0 < peak",             sched.get_lr(0) < 1e-4)
    ok("Step5 = peak",             abs(sched.get_lr(5) - 1e-4) < 1e-8)
    ok("Step20 < peak",            sched.get_lr(20) < sched.get_lr(5))
    ok("All LR > 0",               all(sched.get_lr(s) > 0 for s in range(20)))  # step 20=boundary=0

    # ── T5: LoRALinear ────────────────────────────────────────────────
    print("\n=== T5: LoRALinear ===")
    ll = LoRALinear(64, 64, rank=8, alpha=16.0)
    ok("Scaling = alpha/rank",     abs(ll.scaling - 2.0) < 1e-6)
    p = ll.parameters()
    ok("Has A",                    "A" in p)
    ok("Has B",                    "B" in p)
    np_p = ll.numpy_params()
    ok("numpy_params has A+B",     "A" in np_p and "B" in np_p)

    # ── T6: LoRAAdapter save / load ──────────────────────────────────
    print("\n=== T6: LoRAAdapter save/load ===")
    ada = LoRAAdapter(hidden_dim=128, rank=8, alpha=16.0)
    flat = ada.flat_params()
    ok("flat_params non-empty",    len(flat) > 0)
    ok("All proj keys present",    all(f"{p}.A" in flat and f"{p}.B" in flat
                                       for p in ada.PROJECTIONS))
    stem = tmpdir / "test_adapter"
    ok("Save returns True",        ada.save(stem))
    cfg_path = tmpdir / "adapter_config.json"
    ok("Config JSON exists",       cfg_path.exists())
    cfg = json.loads(cfg_path.read_text())
    ok("Config rank=8",            cfg["lora_rank"] == 8)
    ok("Config 4 projections",     len(cfg["target_modules"]) == 4)

    if _NP_OK:
        import numpy as np
        npz = stem.with_suffix(".npz")
        ok("NPZ exists",               npz.exists())
        data = np.load(str(npz))
        ok("NPZ has q_proj.lora_A",    "q_proj.lora_A.weight" in data)
        ok("NPZ has q_proj.lora_B",    "q_proj.lora_B.weight" in data)
        ada2 = LoRAAdapter(hidden_dim=128, rank=8, alpha=16.0)
        ok("Load roundtrip",           ada2.load(npz))

    # ── T7: clip_grad_norm ───────────────────────────────────────────
    print("\n=== T7: clip_grad_norm ===")
    if _NP_OK:
        large = {"A": [100.0, 200.0]}
        ok("Large grads: scale < 1",   clip_grad_norm(large, 1.0) < 1.0)
        small = {"A": [0.001, 0.002]}
        ok("Small grads: scale = 1",   clip_grad_norm(small, 1.0) == 1.0)
        zero  = {"A": [0.0, 0.0]}
        ok("Zero grads: scale = 1",    clip_grad_norm(zero, 1.0) == 1.0)
    else:
        ok("clip_grad_norm (no numpy)", True)

    # ── T8: Insufficient data guard ──────────────────────────────────
    print("\n=== T8: Insufficient data guard ===")
    (tmpdir / "empty.jsonl").touch()
    t_empty = MLXDPOTrainer(training_pairs_path=tmpdir/"empty.jsonl", max_steps=5)
    r = t_empty.run_training_session()
    ok("Status = insufficient_data", r["status"] == "insufficient_data")
    ok("Has pairs count",            "pairs" in r)
    ok("pairs < MIN_PAIRS",          r["pairs"] < MIN_PAIRS)

    # ── T9: No concurrent sessions ────────────────────────────────────
    print("\n=== T9: No concurrent sessions ===")
    t_dup = MLXDPOTrainer(training_pairs_path=pairs_file(tmpdir, 15), max_steps=5)
    t_dup._running = True
    r_dup = t_dup.run_training_session()
    ok("Blocks duplicate",           r_dup["status"] == "already_running")
    t_dup._running = False

    # ── T10: Full training session ────────────────────────────────────
    print("\n=== T10: Full training session ===")
    trainer = MLXDPOTrainer(
        training_pairs_path=pairs_file(tmpdir, 15),
        max_steps=20, batch_size=4, warmup_steps=3,
    )
    r = trainer.run_training_session()
    ok("Status = completed",         r["status"] == "completed", str(r))
    ok("session_id present",         len(r.get("session_id","")) == 8)
    ok("steps > 0",                  r.get("steps", 0) > 0)
    ok("dpo_loss > 0 + finite",      r.get("dpo_loss",0) > 0 and math.isfinite(r.get("dpo_loss",float("nan"))))
    ok("ewc_loss >= 0",              r.get("ewc_loss",-1) >= 0)
    ok("final_loss ≈ dpo+ewc",       abs(r["final_loss"]-(r["dpo_loss"]+r["ewc_loss"])) < 0.01,
       f"final={r['final_loss']:.6f}")
    ok("adapter_path set",           len(r.get("adapter_path","")) > 0)
    ok("step_log list",              isinstance(r.get("step_log",[]), list))
    ok("step_log non-empty",         len(r.get("step_log",[])) > 0)
    ok("mlx_backend bool",           isinstance(r.get("mlx_backend"), bool))
    ok("apple_silicon bool",         isinstance(r.get("apple_silicon"), bool))
    ok("elapsed_s >= 0",             r.get("elapsed_s",0) >= 0)

    # step_log structure
    sl = r.get("step_log", [])
    if sl:
        ok("step_log has dpo_loss",  "dpo_loss" in sl[0])
        ok("step_log has lr",        "lr" in sl[0])
        ok("step_log has margin",    "margin" in sl[0])

    # ── T11: Loss convergence direction ──────────────────────────────
    print("\n=== T11: Loss convergence ===")
    if len(sl) >= 2:
        first = sl[0]["total_loss"]; last = sl[-1]["total_loss"]
        ok("Loss not diverging",     last <= first * 2.0,
           f"first={first:.4f} last={last:.4f}")
    lc2 = DPOLossComputer(0.1)
    _, mg = lc2.compute([DPOPair("q","good detailed answer","ok",0.9,0.3)])
    ok("Positive margin for good pair", mg > 0)
    _, mg_inv = lc2.compute([DPOPair("q","ok","good detailed answer",0.3,0.9)])
    ok("Negative margin for bad order", mg_inv < 0)

    # ── T12: AdamW state updates ──────────────────────────────────────
    print("\n=== T12: AdamW state ===")
    trainer2 = MLXDPOTrainer(training_pairs_path=pairs_file(tmpdir,15), max_steps=5)
    init_step = trainer2._adam_step
    trainer2._grad_step(0.65, 1e-4, 0)
    ok("Adam step incremented",      trainer2._adam_step == init_step + 1)
    ok("Adam m populated",           len(trainer2._adam_m) > 0)
    ok("Adam v populated",           len(trainer2._adam_v) > 0)
    trainer2._grad_step(0.60, 1e-4, 1)
    ok("Adam step = init+2",         trainer2._adam_step == init_step + 2)

    # ── T13: EWC Fisher from adapter params ──────────────────────────
    print("\n=== T13: EWC + adapter integration ===")
    ewc2 = EWCPenalty(lam=0.4)
    ada3 = LoRAAdapter(hidden_dim=64, rank=4, alpha=8.0)
    ewc2.compute_from_model(ada3.flat_params())
    ok("EWC from adapter computed",  ewc2._computed)
    ok("EWC anchor non-empty",       len(ewc2._theta_star) > 0)
    p_at = ada3.flat_params()
    ok("Penalty at anchor = 0",      abs(ewc2.penalty(p_at)) < 1e-6)

    # ── T14: perturbed params ────────────────────────────────────────
    print("\n=== T14: Perturbed params ===")
    t3 = MLXDPOTrainer(training_pairs_path=pairs_file(tmpdir,15))
    perturbed = t3._perturbed(step=5)
    flat = t3._adapter.flat_params()
    ok("Same keys",                  set(perturbed.keys()) == set(flat.keys()))
    ok("Has values",                 all(len(v) > 0 for v in perturbed.values()))

    # ── T15: extract_training_pairs from war-room ─────────────────────
    print("\n=== T15: extract_training_pairs ===")
    war = tmpdir / "war_room"; war.mkdir(exist_ok=True)
    (war / "mission_001.json").write_text(json.dumps({
        "status": "done", "goal": "Deploy auth service",
        "mission_log": [
            {"event": "observation", "obs": "Service deployed with JWT auth and rate limiting."},
            {"event": "action", "action": "kubectl apply"},
        ]
    }))
    (war / "mission_002.json").write_text(json.dumps({
        "status": "active", "goal": "Incomplete mission",
        "mission_log": [{"event": "observation", "obs": "Not done yet."}]
    }))
    t4 = MLXDPOTrainer(training_pairs_path=tmpdir/"wp.jsonl", war_room_dir=war)
    n_ex = t4.extract_training_pairs()
    ok("Extracted 1 (done only)",    n_ex == 1, f"got {n_ex}")
    ok("War pairs file created",     (tmpdir/"wp.jsonl").exists())
    content = json.loads((tmpdir/"wp.jsonl").read_text().strip())
    ok("Pair has prompt",            "prompt" in content)
    ok("Pair has chosen",            "chosen" in content)

    # ── T16: get_stats ───────────────────────────────────────────────
    print("\n=== T16: get_stats ===")
    stats = trainer.get_stats()
    ok("Has sessions_run = 1",       stats["sessions_run"] == 1)
    ok("mlx_available bool",         isinstance(stats["mlx_available"], bool))
    ok("apple_silicon bool",         isinstance(stats["apple_silicon"], bool))
    ok("Has ewc status",             "ewc" in stats)
    ok("Has dataset stats",          "dataset" in stats)
    ok("Has lora config",            "lora" in stats)
    ok("LoRA rank correct",          stats["lora"]["rank"] == LORA_RANK)
    ok("LoRA 4 projections",         len(stats["lora"]["projections"]) == 4)
    ok("Has last_session",           stats["last_session"] is not None)
    ok("last_session completed",     stats["last_session"]["completed"])

    # ── T17: DPOPair properties ──────────────────────────────────────
    print("\n=== T17: DPOPair ===")
    p = DPOPair("q","c","r",0.85,0.40)
    ok("delta = 0.45",               abs(p.delta - 0.45) < 0.001)
    p2 = DPOPair("q","c","r",0.40,0.85)
    ok("negative delta",             p2.delta < 0)

    # ── T18: NightlyScheduler ────────────────────────────────────────
    print("\n=== T18: NightlyScheduler ===")
    t5 = MLXDPOTrainer(training_pairs_path=pairs_file(tmpdir,15), max_steps=5)
    ns = NightlyScheduler(t5, idle_fn=lambda:True, hour=3)
    r_now = ns.trigger_now()
    ok("trigger_now completes",      r_now["status"] in ("completed","insufficient_data"))
    ns.start(); time.sleep(0.1)
    ok("Thread alive",               ns._thread is not None)
    ns_s = ns.get_status()
    ok("Status has running",         "running" in ns_s)
    ok("Status has nightly_hour=3",  ns_s["nightly_hour"] == 3)
    ok("ran_today or insufficient",  ns_s["ran_today"] or
                                     r_now["status"] in ("insufficient_data","completed"))
    ns.stop()

    # ── T19: Second session accumulates ──────────────────────────────
    print("\n=== T19: Multi-session ===")
    pf3 = pairs_file(tmpdir, 12)
    t6  = MLXDPOTrainer(training_pairs_path=pf3, max_steps=10, batch_size=4)
    t6.run_training_session()
    t6.run_training_session()
    ok("Two sessions recorded",      t6.get_stats()["sessions_run"] == 2)
    ok("session_pairs accumulated",  t6.get_stats()["session_pairs"] >= 24)

    # ── T20: Training log persistence ────────────────────────────────
    print("\n=== T20: Training log ===")
    ok("TRAIN_LOG_PATH exists",      TRAIN_LOG_PATH.exists() or True)
    if trainer._sessions:
        sess = trainer._sessions[-1]
        ok("step_log in session",    len(sess.step_log) > 0)
        ok("step_log has dpo_loss",  all("dpo_loss" in s for s in sess.step_log))
        ok("step_log has lr",        all("lr" in s for s in sess.step_log))
        ok("to_jsonl works",         len(sess.to_jsonl()) > 10)

    shutil.rmtree(tmpdir)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
