import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


POLICY_PATH = (
    Path(__file__).resolve().parents[3]
    / "control"
    / "policy"
    / "g1_gear_wbc_policy.py"
)


class FakeSessionOptions:
    def __init__(self):
        self.intra_op_num_threads = None
        self.inter_op_num_threads = None
        self.execution_mode = None
        self.config_entries = {}

    def add_session_config_entry(self, key, value):
        self.config_entries[key] = value


class FakeInferenceSession:
    instances = []

    def __init__(self, model_path, sess_options=None):
        self.model_path = model_path
        self.sess_options = sess_options
        self.run_calls = []
        self.__class__.instances.append(self)

    def get_inputs(self):
        return [types.SimpleNamespace(name="observations")]

    def run(self, output_names, inputs):
        self.run_calls.append((output_names, inputs))
        return [["policy-output"]]


def load_policy_module():
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = object

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.SessionOptions = FakeSessionOptions
    fake_ort.InferenceSession = FakeInferenceSession
    fake_ort.ExecutionMode = types.SimpleNamespace(
        ORT_SEQUENTIAL="sequential",
        ORT_PARALLEL="parallel",
    )

    fake_torch = types.ModuleType("torch")
    fake_torch.tensor = lambda value, device: {
        "value": value,
        "device": device,
    }

    fake_policy_module = types.ModuleType("decoupled_wbc.control.base.policy")
    fake_policy_module.Policy = type("Policy", (), {})

    fake_utils = types.ModuleType("decoupled_wbc.control.utils.gear_wbc_utils")
    fake_utils.get_gravity_orientation = lambda value: value
    fake_utils.load_config = lambda value: value

    dependencies = {
        "numpy": fake_numpy,
        "onnxruntime": fake_ort,
        "torch": fake_torch,
        "decoupled_wbc.control.base.policy": fake_policy_module,
        "decoupled_wbc.control.utils.gear_wbc_utils": fake_utils,
    }
    spec = importlib.util.spec_from_file_location(
        "_test_g1_gear_wbc_policy", POLICY_PATH
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


class FakeInputTensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class G1GearWbcPolicyRuntimeTest(unittest.TestCase):
    def setUp(self):
        FakeInferenceSession.instances.clear()
        self.module = load_policy_module()

    def test_default_session_options_limit_each_session(self):
        policy = object.__new__(self.module.G1GearWbcPolicy)
        with patch.dict(os.environ, {}, clear=True):
            policy.load_onnx_policy("balance.onnx")
            policy.load_onnx_policy("walk.onnx")

        sessions = FakeInferenceSession.instances
        self.assertEqual(
            [session.model_path for session in sessions],
            ["balance.onnx", "walk.onnx"],
        )
        for session in sessions:
            options = session.sess_options
            self.assertEqual(options.intra_op_num_threads, 1)
            self.assertEqual(options.inter_op_num_threads, 1)
            self.assertEqual(options.execution_mode, "sequential")
            self.assertEqual(
                options.config_entries,
                {
                    "session.intra_op.allow_spinning": "0",
                    "session.inter_op.allow_spinning": "0",
                },
            )
        self.assertIsNot(sessions[0].sess_options, sessions[1].sess_options)

    def test_environment_overrides_are_applied(self):
        environment = {
            "GROOT_ORT_INTRA_OP_THREADS": "2",
            "GROOT_ORT_INTER_OP_THREADS": "3",
            "GROOT_ORT_EXECUTION_MODE": "PARALLEL",
            "GROOT_ORT_ALLOW_SPINNING": "yes",
        }
        with patch.dict(os.environ, environment, clear=True):
            options = self.module._create_ort_session_options()

        self.assertEqual(options.intra_op_num_threads, 2)
        self.assertEqual(options.inter_op_num_threads, 3)
        self.assertEqual(options.execution_mode, "parallel")
        self.assertEqual(
            options.config_entries,
            {
                "session.intra_op.allow_spinning": "1",
                "session.inter_op.allow_spinning": "1",
            },
        )

    def test_invalid_environment_overrides_fail_fast(self):
        cases = (
            ("GROOT_ORT_INTRA_OP_THREADS", "0"),
            ("GROOT_ORT_INTRA_OP_THREADS", "not-a-number"),
            ("GROOT_ORT_INTER_OP_THREADS", "9"),
            ("GROOT_ORT_EXECUTION_MODE", "automatic"),
            ("GROOT_ORT_ALLOW_SPINNING", "sometimes"),
        )
        for name, value in cases:
            with self.subTest(name=name, value=value):
                with patch.dict(os.environ, {name: value}, clear=True):
                    with self.assertRaisesRegex(ValueError, name):
                        self.module._create_ort_session_options()

    def test_load_policy_preserves_inference_contract(self):
        policy = object.__new__(self.module.G1GearWbcPolicy)
        input_value = object()

        with patch.dict(os.environ, {}, clear=True):
            inference = policy.load_onnx_policy("balance.onnx")
            result = inference(FakeInputTensor(input_value))

        session = FakeInferenceSession.instances[-1]
        self.assertEqual(session.model_path, "balance.onnx")
        self.assertIsInstance(session.sess_options, FakeSessionOptions)
        self.assertEqual(
            session.run_calls,
            [(None, {"observations": input_value})],
        )
        self.assertEqual(
            result,
            {
                "value": ["policy-output"],
                "device": "cpu",
            },
        )


if __name__ == "__main__":
    unittest.main()
