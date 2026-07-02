# [lz] Config for OUR AGILE velocity-HEIGHT 23-DoF (true 23-body, no hands) recurrent student.
# Pairs with robojudo/policy/agile_velheight_23dof_policy.py. Sibling of [ih]'s
# g1_agile_velheight_cfg.py (29-body + frozen hands). Differences vs the [ih] velheight cfg:
#   - obs_dof = 23 body joints (OUR exported IO-descriptor order), NOT 29.
#   - policy_type = AgileVelHeight23DoFRecurrentPolicy (drops the 24-hand zero-pad -> obs 68).
#   - policy_name = velheight_23dof_recurrent (our model file).
#   - height range allows the deep-squat floor (~0.25 m, our validated depth) .
# The 12-leg action DoF (G1AgileVHLegsDoF: scale + AGILE gains 40.18/99.10/28.50) is IDENTICAL
# to the [ih] velheight cfg, so it is reused verbatim.
#
# obs_dof order + default_pos are taken VERBATIM from the exported 23-DoF IO descriptor
# (velocity_height_g1_23dof_distillation_recurrent_v0_IO_descriptors.yaml):
#   joint_pos_rel / joint_vel_rel joint_names (23) and joint_pos_offsets (23).

from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.config.g1.policy.g1_agile_velheight_cfg import (
    G1AgileVelHeightPolicyCfg,
    G1AgileVHLegsDoF,
)


class G1AgileVH23BodyDoF(DoFConfig):
    """The 23 body joints in OUR AGILE joint_pos_rel order (IsaacLab articulation order for
    the basic 23-DoF G1 = the 29-DoF interleaved order minus waist_roll/pitch & both wrist
    pitch/yaw). Used to slice/reorder the env dof_pos/vel for joint_pos_rel/joint_vel_rel."""

    joint_names: list[str] = [
        "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
        "left_hip_roll_joint", "right_hip_roll_joint",
        "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
        "left_hip_yaw_joint", "right_hip_yaw_joint",
        "left_shoulder_roll_joint", "right_shoulder_roll_joint",
        "left_knee_joint", "right_knee_joint",
        "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
        "left_ankle_pitch_joint", "right_ankle_pitch_joint",
        "left_elbow_joint", "right_elbow_joint",
        "left_ankle_roll_joint", "right_ankle_roll_joint",
        "left_wrist_roll_joint", "right_wrist_roll_joint",
    ]
    # AGILE default_joint_pos for these 23 (== obs joint_pos_offsets): legs only are non-zero.
    default_pos: list[float] | None = [
        -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.3, 0.3, 0.0, 0.0, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]


class G1AgileVelHeight23DoFPolicyCfg(G1AgileVelHeightPolicyCfg):
    """OUR true-23-DoF velheight recurrent student. Inherits action_scales / obs_scales /
    max_cmd / commands_map from the [ih] velheight cfg (identical); overrides obs_dof (23),
    policy class, model name, and the height range."""

    policy_type: str = "AgileVelHeight23DoFRecurrentPolicy"
    # -> assets/models/g1/agile/velheight_23dof_recurrent.pt  (our model_2999 student)
    policy_name: str = "velheight_23dof_recurrent"

    obs_dof: DoFConfig = G1AgileVH23BodyDoF()
    action_dof: DoFConfig = G1AgileVHLegsDoF()   # identical 12 legs + AGILE gains/offset

    # Our 23-DoF student is the deep-squat fine-tune (trained base_height down to ~0.20 m,
    # validated depth_floor ~0.25 m). Allow the squat floor; default stand 0.72 (training
    # DEFAULT_PELVIS_HEIGHT). r = taller, f = squat lower.
    height_default: float = 0.72
    height_min: float = 0.25
    height_max: float = 0.72
    height_step: float = 0.01
