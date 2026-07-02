

# [lz] ===== OUR true-23-DoF velocity-height student (obs 68, no frozen hands) =====
# Distinct from [ih]'s g1_ih_velheight_23dof (which runs the 29-body velheight MODEL on the
# 23-DoF env). These use OUR 23-body policy (G1AgileVelHeight23DoFPolicyCfg) + the model
# velheight_23dof_recurrent.pt (our model_2999 distilled student). Mirrors the
# g1_agile_velocity_23dof real/keyboard pattern. wrist_load_n=0.0 — our student was NOT
# trained with a wrist load (unlike [ih]'s wrist20).
from .policy.g1_agile_velheight_23dof_cfg import G1AgileVelHeight23DoFPolicyCfg  # noqa: E402


@cfg_registry.register
class g1_lz_velheight_23dof(RlPipelineCfg):
    """[lz] OUR 23-DoF velocity-height recurrent student — RoboJuDo MuJoCo sim2sim.
    Run: SDL_AUDIODRIVER=dummy python scripts/run_pipeline.py -c g1_lz_velheight_23dof"""

    robot: str = "g1"
    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg(
        sim_dt=0.005,
        sim_decimation=4,
        wrist_load_n=0.0,  # our student is not wrist-load trained
    )
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(),
    ]
    policy: G1AgileVelHeight23DoFPolicyCfg = G1AgileVelHeight23DoFPolicyCfg()


@cfg_registry.register
class g1_lz_velheight_23dof_real(g1_lz_velheight_23dof):
    """[lz] OUR 23-DoF velocity-height student on native 23-DoF G1 hardware.
    Arms are held via the SAME rt/lowcmd as the legs (arm_sdk_motor_idx=None), per the [ih]
    23-DoF real findings. Starts with TRAINED gains (no real-only PD override yet) — if a
    fore-aft resonance appears, port the g1_agile_velocity_23dof_real PD tune."""

    prepare_ramp_seconds: float = 6.0
    prepare_progress_bar: bool = False
    env: G1_23RealEnvCfg = G1_23RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(
            net_if="enP8p1s0",
            arm_sdk_motor_idx=None,
        ),
        dof=G1_23RaisedArmsDoF(),
        forward_kinematic=None,
        update_with_fk=False,
    )
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(),
    ]
    do_safety_check: bool = True
    # [lz] real-robot speed cap (sim keeps the inherited 0.8). Linear capped for safety
    # (vx 0.4 / vy 0.3 m/s); wz kept at the FULL trained max 1.0 rad/s — the student
    # under-tracks yaw, so capping wz made turning look dead. _real_keyboard inherits this.
    policy: G1AgileVelHeight23DoFPolicyCfg = G1AgileVelHeight23DoFPolicyCfg(
        max_cmd=[0.4, 0.3, 1.0],
    )


@cfg_registry.register
class g1_lz_velheight_23dof_real_keyboard(g1_lz_velheight_23dof_real):
    """[lz] Keyboard teleop on the real 23-DoF robot. NOTE: 'r'/'f' are the velheight policy's
    height keys (taller / squat lower) — do NOT bind 'r' to MOTION_RESET (would clash).
    Shutdown on o / esc / ctrl_c."""

    ctrl: list[KeyboardCtrlCfg | UnitreeCtrlCfg] = [
        KeyboardCtrlCfg(
            # [lz] 0.5s (default 0.25) so OS key-repeat sustains a held key across the
            # ~250ms repeat-delay gap on the terminal backend (no release events) -> smooth
            # hold-to-move; release stops ~0.5s later. With pynput, release is instant.
            terminal_key_timeout=0.5,
            triggers={
                "o": "[SHUTDOWN]",
                "O": "[SHUTDOWN]",
                "Key.esc": "[SHUTDOWN]",
                "Key.ctrl_c": "[SHUTDOWN]",
            },
        ),
        UnitreeCtrlCfg(),
    ]
