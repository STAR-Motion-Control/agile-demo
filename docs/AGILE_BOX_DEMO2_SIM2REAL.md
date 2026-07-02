# AGILE → box_demo_2 Sim2Real — Decision Memo

> Question answered: to put AGILE locomotion under box_demo_2 on the real G1, do we **keep RoboJuDo**, or use **another sim2real method**?
>
> **Decision (recommended, pending user sign-off):** Do **NOT** keep RoboJuDo as the AGILE brain, and do **NOT** port to `g1_runner` for this bring-up. **Ship the already-written standalone `box_demo_2/agile_lowcmd_pipeline.py`, hardened with the missing safety guards**, and **keep RoboJuDo's `start.sh` as a hot fallback** until AGILE's gait is validated on the real G1.
>
> Date: 2026-06-17 · Verified against source (AGILE `observations.py`/`simulation.py`/`policy.py`, `agile_lowcmd_pipeline.py`, `merge_lowcmd_arm_sdk.py`, `g1_runner/`) + WBC-AGILE docs mirror + GitHub issues. Two adversarial verifiers (pro-RoboJuDo lens, correctness lens) ran against the recommendation.

---

## 0. Upstream reality (docs + GitHub)

- **Docs mirrored** to `cc/WBC-AGILE-site/`. The "Deployment" nav section contains only **Pre-trained Policies** + **Sim-to-MuJoCo Transfer** (CPU MuJoCo). **No physical-robot / `unitree_sdk2` / DDS / e-stop / bring-up page exists** anywhere on the site. The `*_sim2real.gif` clips appear with a bare `<em>Real</em>` caption and **no procedure or attribution**.
- **GitHub `github.com/nvidia-isaac/WBC-AGILE`** (Discussions disabled). Key issues:
  - **#35 "Timeline on Sim2Real Release?"** (open): maintainer — NVIDIA's internal deploy framework, *"working hard to get it out… we don't have a definite timeline yet."*
  - **#58**: *"We do have a sim2sim pipeline… the sim2real (including teleop) opensource is still in progress."* Loco-manip (pick-place) uses **two checkpoints** and sim2sim doesn't support it yet.
  - **#13** (closed): first-party real-hardware claim — but for **Booster T1, not G1**.
  - **#62** (open, unanswered): real-T1 standing sway → residual sim2real gap.
  - **No third-party report of a successful real Unitree G1 run.**
- **Deploy the recurrent LSTM STUDENT**, not the teacher: the velocity-height teacher consumes privileged root linear velocity (not measurable on-robot). `pretrained-policies` explicitly: *"all other policies are suitable for direct deployment on real robots."* `OFFICE_HOUR_FAQ` admits **sysID is incomplete** → expect a deploy→tune PD/DR loop.

⇒ **AGILE ships no deploy framework.** A real-robot runtime must be built around the exported student `.pt` + IODescriptor YAML. **`agile_lowcmd_pipeline.py` already is exactly that runtime** for box_demo_2.

---

## 1. The three options

| | (1) Keep RoboJuDo as the AGILE host | (2) Standalone `agile_lowcmd_pipeline.py` ← **chosen** | (3) Port AGILE into `g1_runner` |
|---|---|---|---|
| Obs fidelity | ⚠️ would re-express AGILE obs/LSTM in RoboJuDo's **closed-source** DSL (silent-drift trap) | ✅ wraps AGILE `ObservationProcessor`/`ActionProcessor`/`PolicyWrapper` **verbatim** | ✅ verbatim, but adapter ABC (`build_obs/infer/map_action`) **doesn't fit** AGILE's stateful `SimState→JointCommand` API |
| Auditability | ❌ RoboJuDo source **not on this machine** — it's a `robojudo==1.5.0` wheel on the robot host | ✅ readable/editable/testable Python you own | ✅ in-repo |
| Safety FSM | ✅ proven `prepare()` ramp + DoFAdapter | ⚠️ **weak today** (gaps below) | ✅ best (`SafetyStateMachine` + `guards.py`) but **never run on hardware** |
| box_demo_2 wire contract | ✅ already wired | ✅ **already conforms** (rt/lowcmd_rl + merge + IPC FSM) | ❌ publishes full `rt/lowcmd`, no arm-hold/merge, no file-IPC source, no `RL_LOWER` |
| Effort to bring-up | re-host AGILE in opaque wheel | **~0.5–1 day** hardening | ~2–4 days + onsite re-validation |
| License | RoboJuDo NC-vs-BY ambiguous | AGILE **Apache-2.0/BSD-3** (clean) | n/a |

### Why not RoboJuDo (the decisive reason)
"Keep RoboJuDo for the AGILE brain" is **not actually a clean option**. What's wired into box_demo_2 is RoboJuDo running *its own* gait policy (`g1_mjlab_loco_real_merge`). Re-hosting *AGILE* inside it means feeding AGILE's 80-d LSTM obs through a **closed-source wheel that isn't on this machine and can't be inspected, edited, or unit-tested** — the worst possible host for the first hardware run of an unproven gait. The genuine AGILE-specific trap (the LSTM hidden/cell buffers, which must be zeroed via in-graph `copy_` buffers, not a `wrapper.reset()`) is exactly the kind of thing an opaque host gets wrong **silently**. The standalone keeps that logic in plain Python we control. License (NC/BY) is a *fleet tiebreaker*, **not** a bring-up blocker.

### Why not g1_runner (now)
`g1_runner` is the locked long-term architecture, but for box_demo_2 it mismatches on **four axes** (full `rt/lowcmd` vs lower-only `rt/lowcmd_rl`; no arm-hold/merge-aware publish; keyboard/gamepad-only command source with no file-IPC; **no `RL_LOWER` mode** — a grasp-time safety-critical state). Its adapter ABC also doesn't fit AGILE's stateful API, and its gain path would **override** AGILE's `ActionProcessor` per-joint kp/kd (a correctness trap that changes validated leg behavior). It has never run on hardware. → **Phase 2**, only when AGILE must coexist with HOMIE/FALCON under one runtime.

### Why the standalone (and what it already gets right — source-verified)
- **Gyro / `base_ang_vel`: CORRECT, no shim needed.** `observations.py:447-449` returns `root_ang_vel` verbatim; the double-rotation bug is isolated to `simulation.py:418-421` (the no-sensor MuJoCo fallback, which the pipeline stubs out). The pipeline feeds the body-frame Unitree IMU gyro straight in (`pipeline:208`). `bench_agile`'s `BodyFrameGyroShim` only existed to make *MuJoCo* behave the way real hardware already does.
- **LSTM state: CORRECT.** Buffers zeroed once before the loop, again right before live policy (`pipeline:402,432`), never re-zeroed in-loop, carried in-place across ticks.
- **Drop-in wire contract: CORRECT.** Consumes `/tmp/robojudo_ext_cmd.json`, publishes only `rt/lowcmd_rl`, honors `RL_FULL/RL_LOWER/DAMP`, holds motors 12-28 at measured q + sets motor 29 = 0 so `arm_sdk` overlay via the merger is unchanged. CRC every frame. box_demo_main / dual_arm / merger stay byte-for-byte untouched.

---

## 2. MUST-FIX before any hardware run

> **Status (2026-06-17):** items 1–4, 6–10 are **implemented** in `agile_lowcmd_pipeline.py` (guard math validated locally; full smoke gated on the 4090/robot). Item 5 (`mode_pr`) is now a **warn-on-mismatch + confirm-on-hardware**. Item 11 (merger fail-safe) is **deferred** per user choice — separate reviewed change.

Safety guards (the standalone's real weakness — none are obs/policy-math errors):

1. **NaN / non-finite guard (HIGH).** Add `np.isfinite` checks on obs **and** `q_target` that **latch DAMP**. ⚠️ A clamp+rate-limit alone does **not** stop NaN — `np.clip(NaN)=NaN` and `NaN−prev=NaN` propagate straight through to the motor at full gain.
2. **Tilt / fall abort (HIGH).** Gate on `projected_gravity`/quaternion (computed for the policy but never used as a guard) → latch DAMP. **This is the #1 fall risk, not the gyro.**
3. **q_target clamp + per-tick rate-limit (HIGH).** Clamp legs to `default_joint_pos_limits` **from the AGILE YAML** (present but unused), not the g1_runner `±3.14` placeholders; add a max-delta-per-step limit.
4. **Joint overspeed abort (HIGH).** Inspect `motor_state.dq`; cut to DAMP on runaway. (Port `check_overspeed` from `g1_runner/safety/guards.py`.)
5. **`mode_pr` must-confirm (HIGH, box_demo_2-specific).** Pipeline hardcodes `mode_pr=0`; `dual_arm` echoes lowstate `mode_pr`; the merger forwards the **RL frame's** mode fields as final. Verify the real G1 lowstate `mode_pr` and either echo it or prove `0` is correct — wrong PR/AB → ankle misinterpretation → fall.
6. **`mode_machine != 0` startup assert.** Firmware can reject `mode_machine=0`, leaving motors unenabled.
7. **`publish_prepare` staleness re-check.** It captures `first_msg` once and blends toward a frozen pose if DDS stalls mid-ramp; abort to DAMP if state ages out.
8. **Explicit `num_obs == policy-input-dim` assert** at first inference. (TorchScript throws on width mismatch, but this is cheap fail-fast; note width check does **not** catch a same-width term/joint mis-order — see gate below.)
9. **Pipeline self-damps on its own stale-state branch** (`pipeline:442` currently just `continue`s) — defense-in-depth independent of the merger.
10. **Hardware e-stop.** Decode `wireless_remote` select button from LowState into the DAMP latch; keyboard `o` alone is insufficient.
11. **Merger fail-safe (SEPARATE — needs user sign-off).** `merge_lowcmd_arm_sdk.py:117-118` goes **silent** when `rt/lowcmd_rl` is stale >0.5 s, leaving the robot on a frozen firmware PD target. Change it to emit `kd=8` damping. Touches the **sole final-command publisher** → review as its own change. (Pre-existing hazard the RoboJuDo path shared too.)

**Artifact:** deploy the repo **baseline** recurrent student (`agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt`, trained height `[0.40,0.72]`, the "most stable/accurate" variant) — **not** the deep-squat finetune (`student2999`, less stable: more falls / worse small-command accuracy) and **not** the teacher. Confirm which `.pt` `--policy` points at before trusting the `[0.40,0.72]` height clamp.

---

## 3. Bring-up plan (incremental, safety-first)

0. **Keep `start.sh` (RoboJuDo) as a hot fallback** the whole time — both launchers share the identical merger + box_demo_main + arm_sdk, so reverting locomotion is a one-pane swap. Cheapest hedge against AGILE's gait being unproven on a real G1.
1. **Off-robot:** apply must-fix 1–9; re-run smoke (`OK 29 80 12 MLPPolicyWrapper 2`).
2. **On-robot, `--dry-run`:** subscribe `rt/lowstate`, read IPC, compute obs/policy, would-publish `rt/lowcmd_rl` steadily; load-test the 50 Hz CPU torch loop holds rate (inference is ~0.15 ms — any failure will be GIL/DDS-callback/OS jitter, not math).
3. **Transport + safety (no trust in gait yet):** trip-test tilt abort, kill-the-pipeline (confirm merger now damps, not freezes), stale-state abort, wireless-remote e-stop. Validate the safety boundary **before** the gait.
4. **HARD GATE — obs byte-compare:** robot held/suspended, dump on-robot 80-vector vs sim2mujoco for the same static pose; byte-match commands/`base_ang_vel`/`projected_gravity`/`joint_pos_rel`/`joint_vel_rel`/`last_action` **including units** (gyro rad/s, not deg/s). The width check catches only dimension errors; this is the sole defense against same-width term/joint mis-order — the silent-fall class. **GO/NO-GO.**
5. **Gait (suspended → ground):** cosine ramp → `RL_FULL` zero-command balance → small `vx` → height (z/x 2 cm steps, `c=0.55 m`) → turning. Conservative gains first (`--pd-scale`-equivalent if unstable). **Validate walk distances side-by-side against RoboJuDo's known-good** — RoboJuDo's `robot_move.py` constants were empirically fit to *its* gait; AGILE invalidates that calibration.
6. **Autonomous integration:** full `start_agile_box.sh`; tune `--fwd-max/--lat-max/--yaw-max` + `--walk-scale` on hardware (no code edit). RobotMover writes **normalized** velocities (no `units` field) so AGILE rescales — autonomous distances differ from RoboJuDo. Verify `RL_LOWER` keeps legs balancing during grasp while `arm_sdk` owns the upper body.

**Phase 2 (later, non-blocking):** build the `g1_runner` AGILE adapter (`AgileSplitBackend` → `rt/lowcmd_rl` + upper-hold + motor29=0; `FileIpcCommandProvider` modeling `RL_LOWER` as POLICY-with-zero-command; adapter wrapping AGILE's processors, reconciling AGILE's own gains vs the SafetyStateMachine). Budget 2–4 days + full onsite re-validation.

---

## 4. Open questions for the user

1. **Merger fail-safe sign-off** (must-fix #11): OK to make a stale `rt/lowcmd_rl` emit `kd=8` damping instead of going silent? Touches the sole final-command publisher.
2. **Onboard compute target:** Jetson Orin or x86+GPU? (AGILE is not `cuda:0`-hardcoded; `--device cpu` has huge headroom but confirm the host holds 50 Hz under DDS callbacks.)
3. **Hardware e-stop source:** is a `wireless_remote`/gamepad reliably reaching the robot host during the box demo, so we can wire the select-button DAMP?
4. **Which `.pt`:** confirm baseline student (recommended) vs deep-squat `student2999`. Does the box demo need deep squatting (worse stability) or just stable pick-height ~0.55 m?
5. **Commercial intent:** if box_demo must be commercial-usable, AGILE's Apache/BSD cleanliness is a real reason to standardize on it over CC-BY-NC-SA alternatives (HOMIE) for the fleet.

---

## 5. Honest residual risk

AGILE's gait has **provably never walked a real G1** (only a 4090 shape/config smoke test; the docs' "Real" G1 GIFs have no reproducible procedure). RoboJuDo's gait demonstrably walks this robot. So the choice trades a working gait for an unproven one — defensible because the standalone wire contract is already built and RoboJuDo can't host AGILE's policy auditably — but it means: **keep RoboJuDo available as fallback, and treat its known-good walk as the ground-truth reference during AGILE tuning.** sysID is incomplete upstream → budget a deploy→tune loop.
