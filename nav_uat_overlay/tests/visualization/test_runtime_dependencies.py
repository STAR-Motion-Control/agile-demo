import importlib.util


def test_websocket_runtime_library_is_available():
    websockets = importlib.util.find_spec("websockets")
    wsproto = importlib.util.find_spec("wsproto")

    assert websockets is not None or wsproto is not None
