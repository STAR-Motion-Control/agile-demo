# [lz] AGILE velocity-HEIGHT 23-DoF (TRUE 23-body, no hands) recurrent (LSTM) policy.
#
# Deploys OUR distilled 23-DoF basic-G1 student (Velocity-Height-G1-23dof-Distillation-
# Recurrent, model_2999). It is a sibling of [ih]'s AgileVelHeightRecurrentPolicy, which
# targets the 29-body + 24-frozen-hand variant (obs 128). The basic 23-DoF G1 has NO hands
# and only 23 body joints, so the exported obs has NO 24-hand zero-pad:
#
#   obs (68) = commands(4) + base_ang_vel(3) + projected_gravity(3)
#              + joint_pos_rel(23) + joint_vel_rel(23)*0.1 + last_action(12)
#
# Action (12 legs), LSTM hidden-state carry/reset, command remap, height (r/f) are inherited.
# Verified against the exported 23-DoF IO descriptor (joint_pos_rel/joint_vel_rel shape 23).

import numpy as np

from robojudo.policy import policy_registry
from robojudo.policy.agile_velheight_policy import AgileVelHeightRecurrentPolicy
from robojudo.utils.util_func import command_remap, get_gravity_orientation


@policy_registry.register
class AgileVelHeight23DoFRecurrentPolicy(AgileVelHeightRecurrentPolicy):
    """True 23-body velheight student: same as the parent but WITHOUT the 24 hand zeros,
    and with a HOLD-TO-MOVE keyboard fix (see _get_commands)."""

    def get_observation(self, env_data, ctrl_data):
        commands = self._get_commands(ctrl_data)                     # (4,) [vx,vy,wz,height]
        gravity = get_gravity_orientation(env_data.base_quat)        # (3,)
        # env_data.dof_pos / dof_vel are already sliced to the 23 body joints (obs_dof order)
        body_pos_rel = (env_data.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos  # (23,)
        body_vel_rel = env_data.dof_vel * self.obs_scales.dof_vel    # (23,)
        obs = np.concatenate([
            commands,                                                # 4
            env_data.base_ang_vel * self.obs_scales.ang_vel,         # 3
            gravity,                                                 # 3
            body_pos_rel,                                            # 23  (NO hand zero-pad)
            body_vel_rel,                                            # 23
            self.last_action,                                        # 12
        ]).astype(np.float32)                                        # total 68
        return obs, {"commands": commands}

    def _get_commands(self, ctrl_data) -> np.ndarray:
        """[lz] HOLD-TO-MOVE fix. The parent edge-tracks its own _held_keys from press/release
        events — but the terminal keyboard backend (used when pynput has no display, e.g. over
        SSH) sends ONLY press events, so _held_keys never clears -> 'tap w => walks forever'.
        Instead use the controller's authoritative `keys_pressed` set, which already removes a
        key on pynput release AND auto-expires terminal-source keys after terminal_key_timeout.
        => pynput: release stops instantly; terminal: OS key-repeat sustains the hold, release
        stops after the timeout. r/f height still step on press edges. Joystick path unchanged.
        Returns [vx, vy, wz, height]."""
        vel = np.zeros(3, dtype=np.float32)
        for key in ctrl_data.keys():
            if key in ["JoystickCtrl", "UnitreeCtrl"]:
                axes = ctrl_data[key]["axes"]
                vel[0] = command_remap(axes["LeftY"], self.commands_map[0])
                vel[1] = command_remap(axes["LeftX"], self.commands_map[1])
                vel[2] = command_remap(axes["RightX"], self.commands_map[2])
                break
            if key in ["KeyboardCtrl"]:
                kc = ctrl_data[key]
                # height: step on press edges (works for both pynput and terminal)
                for event in kc.get("keyboard_event", []):
                    if event.get("type") != "keyboard" or not event.get("pressed"):
                        continue
                    if event["name"] == "r":
                        self._height = min(self.height_max, self._height + self.height_step)
                    elif event["name"] == "f":
                        self._height = max(self.height_min, self._height - self.height_step)
                # velocity: rebuild from the controller's held-set (NOT edge-tracked)
                for name in kc.get("keys_pressed", []):
                    if name in self._MOVE_KEYS:
                        axis, sign = self._MOVE_KEYS[name]
                        vel[axis] = command_remap(sign, self.commands_map[axis])
                break
        vel = vel * self.max_cmd
        cmd = np.array([vel[0], vel[1], vel[2], self._height], dtype=np.float32)
        # [lz] debug: print the velocity command only when it CHANGES (key press/release),
        # so q/e turning can be confirmed (wz != 0) without console spam.
        last = getattr(self, "_last_cmd_dbg", None)
        if last is None or float(np.abs(cmd[:3] - last).max()) > 1e-3:
            print(f"[velheight23] cmd vx={cmd[0]:+.2f} vy={cmd[1]:+.2f} "
                  f"wz={cmd[2]:+.2f} h={cmd[3]:.2f}", flush=True)
            self._last_cmd_dbg = cmd[:3].copy()
        return cmd
