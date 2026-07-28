# ONNX Runtime resource limits

`G1GearWbcPolicy` creates separate ONNX Runtime sessions for the standing and
walking policies. Each session now uses these defaults:

| Setting | Default |
| --- | --- |
| Intra-op threads | 1 |
| Inter-op threads | 1 |
| Execution mode | Sequential |
| Intra-op and inter-op spinning | Disabled |

The defaults avoid ONNX Runtime worker pools consuming idle CPU between the
50 Hz policy calls. No environment variables are required for onboard use.

The following environment variables provide explicit overrides:

| Environment variable | Accepted values |
| --- | --- |
| `GROOT_ORT_INTRA_OP_THREADS` | Integer from 1 through 8 |
| `GROOT_ORT_INTER_OP_THREADS` | Integer from 1 through 8 |
| `GROOT_ORT_EXECUTION_MODE` | `sequential` or `parallel` |
| `GROOT_ORT_ALLOW_SPINNING` | `0/1`, `false/true`, `no/yes`, or `off/on` |

Invalid values fail during policy loading. In particular, zero is rejected
because ONNX Runtime interprets a zero thread count as an automatic setting
that can create workers for every available core.

Parallel execution and spinning should only be enabled for controlled
benchmarks. Both settings can raise idle CPU usage and scheduling contention
on the onboard Jetson.

Run the dependency-free unit test with:

```shell
python3 -m unittest \
  decoupled_wbc.tests.control.policy.test_g1_gear_wbc_policy_runtime
```
