import asyncio
import threading
from contextlib import suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .models import SCHEMA_VERSION
from .map_utils import to_plain_data
from .ws_manager import WebSocketManager

NO_CACHE_HEADERS = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}


def _build_hello_message():
    return {
        "type": "hello",
        "timestamp": 0.0,
        "schema_version": SCHEMA_VERSION,
        "capabilities": {
            "realtime": True,
            "replay": True,
            "map_overlay": True,
        },
    }


def create_app(
    state_hub,
    image_store,
    map_metadata,
    replay_store=None,
    static_dir=None,
    stop_navigation=None,
    submit_text_navigation=None,
    manual_control=None,
):
    app = FastAPI()
    app.state.state_hub = state_hub
    app.state.image_store = image_store
    app.state.map_metadata = to_plain_data(map_metadata)
    app.state.replay_store = replay_store
    app.state.ws_manager = WebSocketManager()
    app.state.static_dir = Path(static_dir) if static_dir else None
    app.state.stop_navigation = stop_navigation
    app.state.submit_text_navigation = submit_text_navigation
    app.state.manual_control = manual_control
    app.state.state_hub.set_broadcast(app.state.ws_manager.broadcast)

    @app.get("/")
    def root():
        return HTMLResponse("<html><body><h1>Navigation visualization backend ready.</h1></body></html>")

    @app.get("/viz/api/map/metadata")
    def get_map_metadata():
        return JSONResponse(app.state.map_metadata)

    @app.get("/viz/api/map/image")
    def get_map_image():
        image_path = app.state.map_metadata.get("image_path")
        if not image_path:
            raise HTTPException(status_code=404, detail="missing map image path")
        path = Path(image_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="map image not found")
        media_type = "image/png" if path.suffix.lower() == ".png" else "application/octet-stream"
        return FileResponse(
            path,
            media_type=media_type,
            headers=NO_CACHE_HEADERS,
        )

    @app.get("/viz/api/frame/{group}/{name}.jpg")
    def get_frame(group: str, name: str):
        key = f"{group}_{name}"
        payload = app.state.image_store.get(key)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"missing frame: {key}")
        return Response(
            content=payload["bytes"],
            media_type=payload["content_type"],
            headers=NO_CACHE_HEADERS,
        )

    @app.get("/viz/api/replay/tasks")
    def get_replay_tasks():
        if app.state.replay_store is None:
            raise HTTPException(status_code=404, detail="replay store unavailable")
        return JSONResponse(app.state.replay_store.list_tasks())

    @app.get("/viz/api/replay/tasks/{task_id}")
    def get_replay_task(task_id: str):
        if app.state.replay_store is None:
            raise HTTPException(status_code=404, detail="replay store unavailable")
        task = app.state.replay_store.get_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return JSONResponse(task)

    @app.get("/viz/api/replay/tasks/{task_id}/events")
    def get_replay_task_events(task_id: str):
        if app.state.replay_store is None:
            raise HTTPException(status_code=404, detail="replay store unavailable")
        events = app.state.replay_store.load_events(task_id)
        if events is None:
            raise HTTPException(status_code=404, detail="task not found")
        return JSONResponse(events)

    @app.get("/viz/api/replay/tasks/{task_id}/snapshot")
    def get_replay_task_snapshot(task_id: str):
        if app.state.replay_store is None:
            raise HTTPException(status_code=404, detail="replay store unavailable")
        snapshot = app.state.replay_store.load_snapshot(task_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="snapshot not found")
        return JSONResponse(snapshot)

    @app.post("/viz/api/navigation/stop")
    def stop_navigation():
        stop_handler = app.state.stop_navigation
        if stop_handler is None:
            raise HTTPException(status_code=503, detail="stop navigation unavailable")
        result = stop_handler()
        return JSONResponse(result)

    @app.post("/viz/api/navigation/text")
    def submit_text_navigation(payload: dict):
        submit_handler = app.state.submit_text_navigation
        if submit_handler is None:
            raise HTTPException(status_code=503, detail="text navigation unavailable")

        goal_text = str(payload.get("goal_text", "")).strip()
        if not goal_text:
            raise HTTPException(status_code=400, detail="goal_text is required")

        dry_run = payload.get("dry_run", False)
        if isinstance(dry_run, str):
            dry_run = dry_run.strip().lower() in {"1", "true", "yes", "on"}
        else:
            dry_run = bool(dry_run)

        result = submit_handler(goal_text, dry_run=dry_run)
        return JSONResponse(result)

    @app.post("/viz/api/navigation/manual")
    def submit_manual_control(payload: dict):
        manual_handler = app.state.manual_control
        if manual_handler is None:
            raise HTTPException(status_code=503, detail="manual control unavailable")

        action = str(payload.get("action", "")).strip().lower()
        if action not in {"forward", "backward", "left", "right", "rotate_left", "rotate_right"}:
            raise HTTPException(status_code=400, detail="invalid manual control action")
        return JSONResponse(manual_handler(action))

    @app.get("/viz/api/replay/tasks/{task_id}/frame/{group}/{name}/{version}.jpg")
    def get_replay_task_frame(task_id: str, group: str, name: str, version: int):
        if app.state.replay_store is None:
            raise HTTPException(status_code=404, detail="replay store unavailable")
        payload = app.state.replay_store.load_frame(task_id, f"{group}_{name}", version)
        if payload is None:
            raise HTTPException(status_code=404, detail="replay frame not found")
        return Response(
            content=payload["bytes"],
            media_type=payload["content_type"],
            headers=NO_CACHE_HEADERS,
        )

    @app.websocket("/viz/ws")
    async def websocket_endpoint(websocket: WebSocket):
        queue = app.state.ws_manager.register()
        await websocket.accept()
        disconnect_task = asyncio.create_task(websocket.receive())
        try:
            await websocket.send_json(_build_hello_message())
            await websocket.send_json(app.state.state_hub.build_snapshot())
            while True:
                if disconnect_task.done():
                    incoming = disconnect_task.result()
                    if incoming.get("type") == "websocket.disconnect":
                        break
                    disconnect_task = asyncio.create_task(websocket.receive())
                message = app.state.ws_manager.get_nowait(queue)
                if message is not None:
                    await websocket.send_json(message)
                    continue
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            pass
        finally:
            if not disconnect_task.done():
                disconnect_task.cancel()
                with suppress(asyncio.CancelledError):
                    await disconnect_task
            app.state.ws_manager.unregister(queue)

    if app.state.static_dir and app.state.static_dir.exists():
        assets_dir = app.state.static_dir / "assets"
        if assets_dir.exists():
            app.mount("/viz/assets", StaticFiles(directory=assets_dir), name="viz-assets")

        @app.get("/viz")
        @app.get("/viz/{path:path}")
        def serve_frontend(path: str = ""):
            index_path = app.state.static_dir / "index.html"
            if not index_path.exists():
                raise HTTPException(status_code=404, detail="frontend build missing")
            return HTMLResponse(index_path.read_text(encoding="utf-8"))

    return app


def launch_server_in_thread(app, host, port):
    import uvicorn

    config = uvicorn.Config(
        app=app,
        host=host,
        port=int(port),
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return server, thread
