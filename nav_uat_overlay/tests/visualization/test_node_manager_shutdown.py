import sys
from pathlib import Path
from threading import Event


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from Node import NodeManager  # noqa: E402


class FakeExecutor:
    def __init__(self):
        self.shutdown_calls = []
        self.removed = []

    def shutdown(self, timeout_sec=None):
        self.shutdown_calls.append(timeout_sec)
        return True

    def remove_node(self, node):
        self.removed.append(node)


class FakeNode:
    def __init__(self):
        self.destroy_calls = 0

    def destroy_node(self):
        self.destroy_calls += 1


class FakeThread:
    def __init__(self):
        self.join_calls = []

    def is_alive(self):
        return True

    def join(self, timeout=None):
        self.join_calls.append(timeout)


class FakeActionExecutor:
    def __init__(self):
        self._stop_flag = Event()
        self._finish_flag = Event()
        self._new_action_event = Event()
        self.shutdown_calls = []

    def shutdown(self, timeout_sec=None):
        self.shutdown_calls.append(timeout_sec)


class FakeStepControl:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1


def test_request_stop_signals_action_executor_and_calls_stop():
    manager = NodeManager.__new__(NodeManager)
    manager.action_excutor = FakeActionExecutor()
    called = {"stop": 0}

    def fake_stop():
        called["stop"] += 1
        return True, "ok", 1

    manager.stop = fake_stop

    manager.request_stop()

    assert called["stop"] == 1
    assert manager.action_excutor._stop_flag.is_set()
    assert manager.action_excutor._finish_flag.is_set()
    assert manager.action_excutor._new_action_event.is_set()


def test_shutdown_is_idempotent_and_destroys_managed_nodes():
    manager = NodeManager.__new__(NodeManager)
    manager._shutdown = False
    manager.executor = FakeExecutor()
    node_a = FakeNode()
    node_b = FakeNode()
    manager._nodes = [node_a, node_b]
    manager.executor_thread = FakeThread()
    manager.action_excutor = FakeActionExecutor()
    manager.step_control_client = FakeStepControl()
    manager.request_stop = lambda: None

    manager.shutdown(timeout_sec=0.5)
    manager.shutdown(timeout_sec=0.5)

    assert manager.executor.shutdown_calls == [0.5]
    assert manager.executor.removed == [node_a, node_b]
    assert node_a.destroy_calls == 1
    assert node_b.destroy_calls == 1
    assert manager.executor_thread.join_calls == [0.5]
    assert manager.action_excutor.shutdown_calls == [0.5]
    assert manager.step_control_client.shutdown_calls == 1
