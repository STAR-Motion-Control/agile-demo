"""Pure-data contract for the onboard manipulation arm runtime."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
DEFAULT_ARM_RUNTIME_SOCKET = "/tmp/groot_arm_runtime.sock"
DEFAULT_ARM_RUNTIME_STATUS = "/tmp/groot_arm_runtime_status.json"

# Left arm, right arm, then waist. This is the established box_demo q ordering.
ARM_JOINT_INDICES = (
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    12,
    13,
    14,
)
GRIP_JOINT_INDICES = (16, 18, 23, 25)
GRIP_Q_POSITIONS = tuple(ARM_JOINT_INDICES.index(index) for index in GRIP_JOINT_INDICES)
ARM_JOINT_COUNT = len(ARM_JOINT_INDICES)

PROFILE_MANIP = "manip_v1"
PROFILE_WAIST_LEGACY = "waist_legacy_v1"
ALLOWED_PROFILES = frozenset((PROFILE_MANIP, PROFILE_WAIST_LEGACY))
ALLOWED_SCOPES = frozenset(("upper_body",))

SOURCE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
TOKEN_RE = re.compile(r"^[a-f0-9]{32}$")


class ArmProtocolError(RuntimeError):
    """Raised when an arm-runtime message violates the wire contract."""

    def __init__(self, message: str, code: str = "INVALID_REQUEST"):
        super().__init__(message)
        self.code = code


def finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ArmProtocolError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ArmProtocolError(f"{name} must be finite")
    return result


def finite_vector(
    value: Any,
    name: str,
    *,
    length: int = ARM_JOINT_COUNT,
    absolute_limit: float = 10.0,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise ArmProtocolError(f"{name} must be an array")
    result = tuple(finite_float(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(result) != length:
        raise ArmProtocolError(f"{name} must contain exactly {length} values")
    if any(abs(item) > absolute_limit for item in result):
        raise ArmProtocolError(f"{name} exceeds the sanity limit")
    return result


@dataclass(frozen=True)
class ArmFrame:
    q: tuple[float, ...]
    weight: float = 1.0
    profile: str = PROFILE_MANIP
    scope: str = "upper_body"

    @classmethod
    def from_wire(cls, raw: Any) -> "ArmFrame":
        if not isinstance(raw, Mapping):
            raise ArmProtocolError("frame must be an object")
        profile = str(raw.get("profile", PROFILE_MANIP))
        scope = str(raw.get("scope", "upper_body"))
        if profile not in ALLOWED_PROFILES:
            raise ArmProtocolError(f"unsupported profile: {profile!r}")
        if scope not in ALLOWED_SCOPES:
            raise ArmProtocolError(f"unsupported scope: {scope!r}")
        weight = finite_float(raw.get("weight", 1.0), "frame.weight")
        if not 0.0 <= weight <= 1.0:
            raise ArmProtocolError("frame.weight must be in [0, 1]")
        return cls(
            q=finite_vector(raw.get("q"), "frame.q"),
            weight=weight,
            profile=profile,
            scope=scope,
        )
    def to_wire(self) -> dict[str, Any]:
        return {
            "q": list(self.q),
            "weight": self.weight,
            "profile": self.profile,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class UpperBodyState:
    boot_id: str
    state_seq: int
    state_rx_mono_ns: int
    state_age_ms: float
    mode_machine: int
    mode_pr: int
    q: tuple[float, ...]
    tau_est: tuple[float, ...]
    policy_seq: int
    policy_rx_mono_ns: int
    policy_age_ms: float | None
    policy_q: tuple[float, ...] | None
    owner: str | None
    applied_seq: int
    runtime_state: str

    @classmethod
    def from_snapshot(cls, raw: Any) -> "UpperBodyState":
        if not isinstance(raw, Mapping):
            raise ArmProtocolError("snapshot response must be an object")
        state = raw.get("state")
        runtime = raw.get("runtime")
        policy = raw.get("policy")
        if not isinstance(state, Mapping) or not isinstance(runtime, Mapping):
            raise ArmProtocolError("snapshot is missing state/runtime")
        if policy is not None and not isinstance(policy, Mapping):
            raise ArmProtocolError("snapshot.policy must be an object or null")
        q = finite_vector(state.get("q"), "snapshot.state.q")
        tau_est = finite_vector(
            state.get("tau_est"),
            "snapshot.state.tau_est",
            absolute_limit=1000.0,
        )
        policy_q = None
        policy_age_ms = None
        policy_seq = 0
        policy_rx_mono_ns = 0
        if policy is not None:
            policy_q = finite_vector(policy.get("q"), "snapshot.policy.q")
            policy_age_ms = finite_float(
                policy.get("age_ms"), "snapshot.policy.age_ms"
            )
            policy_seq = int(policy.get("sequence", 0))
            policy_rx_mono_ns = int(policy.get("received_mono_ns", 0))
        return cls(
            boot_id=str(raw.get("boot_id", "")),
            state_seq=int(state.get("sequence", 0)),
            state_rx_mono_ns=int(state.get("received_mono_ns", 0)),
            state_age_ms=finite_float(state.get("age_ms"), "snapshot.state.age_ms"),
            mode_machine=int(state.get("mode_machine", 0)),
            mode_pr=int(state.get("mode_pr", 0)),
            q=q,
            tau_est=tau_est,
            policy_seq=policy_seq,
            policy_rx_mono_ns=policy_rx_mono_ns,
            policy_age_ms=policy_age_ms,
            policy_q=policy_q,
            owner=(
                None
                if runtime.get("owner") in (None, "")
                else str(runtime.get("owner"))
            ),
            applied_seq=int(runtime.get("applied_sequence", -1)),
            runtime_state=str(runtime.get("state", "unknown")),
        )
