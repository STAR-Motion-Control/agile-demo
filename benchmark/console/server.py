#!/usr/bin/env python3
"""G1 benchmark experiment-console server (pure stdlib, no pip deps).

Serves the static site (index.html / manifest.json / videos/*) AND a tiny
render API that runs one custom benchmark trial on demand:

    POST /api/render   {"model":"homie"|"agile", "kind":"squat"|"walk"|"circle",
                        "height":0.4, "rate":0.2, "vx":0.6, "wz":0.3}
    GET  /api/job/<id>  -> job status + log tail (+ videos when done)
    GET  /api/jobs      -> all jobs, newest first

One worker thread executes jobs serially (the GPU/CPU box runs one harness
trial at a time).  Finished videos are captioned (caption_videos.py) and land
in <root>/videos/custom/, served same-origin -> no CORS issues.

Run on the 4090:
    nohup python3 server.py --port 8017 --root /sda/lizhe/g1bench/site \
        > server.log 2>&1 &
"""
import argparse
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------
# CONFIG — 4090 paths.  Edit here when the box layout changes.
# --------------------------------------------------------------------------
CONFIG = {
    "BENCH_DIR": "/sda/lizhe/g1bench",          # cwd for harness runs
    "PYTHON": {
        "homie": "/sda/lizhe/miniforge3/envs/homie/bin/python",
        "agile": "/sda/lizhe/miniforge3/envs/agile/bin/python",
    },
    "BENCH_SCRIPT": {
        "homie": "/sda/lizhe/g1bench/bench_homie.py",
        "agile": "/sda/lizhe/g1bench/bench_agile.py",
    },
    "AGILE_EXTRA_ARGS": [
        "--agile-repo", "/sda/lizhe/g1bench/WBC-AGILE",
        "--mjcf",
        "/sda/lizhe/g1bench/unitree_mujoco/unitree_robots/g1/scene_29dof.xml",
    ],
    "CAPTION_SCRIPT": "/sda/lizhe/g1bench/caption_videos.py",
    "CAPTION_PYTHON": "/sda/lizhe/miniforge3/envs/homie/bin/python",
    "JOB_TIMEOUT_S": 600,
    "LOG_TAIL_LINES": 30,
}

# kind -> (test name, required params).  Ranges are the UI slider ranges.
KINDS = {
    "squat": {"test": "squat_sweep",
              "params": {"height": (0.2, 0.7), "rate": (0.05, 0.8)}},
    "walk": {"test": "walk_speed", "params": {"vx": (0.2, 1.2)}},
    "circle": {"test": "circle_pillar",
               "params": {"vx": (0.2, 1.2), "wz": (0.1, 0.6)}},
}
MODELS = ("homie", "agile")
JSONL_NAME = {"homie": "results.jsonl",
              "agile": "agile_{test}_results.jsonl"}

# --------------------------------------------------------------------------
# Job registry (in-memory + persisted to <root>/jobs.json across restarts)
# --------------------------------------------------------------------------
_JOBS = {}            # id -> dict (replaced wholesale on each update)
_JOBS_LOCK = threading.Lock()
_JOB_QUEUE = queue.Queue()
_ROOT = None          # set in main()


def _jobs_path():
    return os.path.join(_ROOT, "jobs.json")


def _persist_jobs():
    snapshot = sorted(_JOBS.values(), key=lambda j: j["created"], reverse=True)
    tmp = _jobs_path() + ".tmp"
    with open(tmp, "w") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _jobs_path())


def _load_jobs():
    try:
        with open(_jobs_path()) as f:
            for j in json.load(f):
                if j.get("status") in ("queued", "running"):
                    # orphaned by a previous server process
                    j = dict(j, status="failed",
                             error="server restarted mid-job")
                _JOBS[j["id"]] = j
    except (OSError, ValueError):
        pass


def job_update(job_id, **changes):
    """Replace the stored job dict with an updated copy (no mutation)."""
    with _JOBS_LOCK:
        cur = _JOBS.get(job_id)
        if cur is None:
            return None
        new = dict(cur, **changes)
        _JOBS[job_id] = new
        _persist_jobs()
        return new


def job_get(job_id):
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


def jobs_list():
    with _JOBS_LOCK:
        return sorted(_JOBS.values(), key=lambda j: j["created"], reverse=True)


# --------------------------------------------------------------------------
# Request validation
# --------------------------------------------------------------------------
def validate_render_request(body):
    """Returns (params dict, None) or (None, error string)."""
    if not isinstance(body, dict):
        return None, "body must be a JSON object"
    model = body.get("model")
    kind = body.get("kind")
    if model not in MODELS:
        return None, "model must be one of %s" % (MODELS,)
    if kind not in KINDS:
        return None, "kind must be one of %s" % (tuple(KINDS),)
    params = {"model": model, "kind": kind}
    for name, (lo, hi) in KINDS[kind]["params"].items():
        v = body.get(name)
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return None, "missing/non-numeric param '%s'" % name
        v = float(v)
        if not (lo <= v <= hi):
            return None, "param '%s'=%s out of range [%s, %s]" % (name, v,
                                                                  lo, hi)
        params[name] = round(v, 3)
    return params, None


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
def build_bench_cmd(params, out_dir):
    model, kind = params["model"], params["kind"]
    test = KINDS[kind]["test"]
    cmd = [CONFIG["PYTHON"][model], CONFIG["BENCH_SCRIPT"][model],
           "--test", test, "--trials", "1", "--video", "all",
           "--out-dir", out_dir]
    if kind == "squat":
        cmd += ["--custom-height", str(params["height"]),
                "--custom-rate", str(params["rate"])]
    elif kind == "walk":
        cmd += ["--custom-vx", str(params["vx"])]
    else:  # circle
        cmd += ["--custom-vx", str(params["vx"]),
                "--custom-wz", str(params["wz"])]
    if model == "agile":
        cmd += CONFIG["AGILE_EXTRA_ARGS"]
    return cmd


def run_logged(cmd, log_path, timeout_s, cwd):
    """Run cmd appending stdout+stderr to log_path.  Returns (rc, timed_out).
    Kills the whole process group on timeout."""
    env = dict(os.environ, MUJOCO_GL="egl")
    with open(log_path, "a") as lf:
        lf.write("\n$ %s\n" % " ".join(cmd))
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=cwd, env=env, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_s)
            return rc, False
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            proc.wait()
            return -9, True


def log_tail(log_path, n_lines):
    try:
        with open(log_path, "rb") as f:
            data = f.read()[-16384:]
        lines = data.decode("utf-8", "replace").splitlines()
        return lines[-n_lines:]
    except OSError:
        return []


def collect_videos(videos_dir):
    out = []
    if os.path.isdir(videos_dir):
        for name in sorted(os.listdir(videos_dir)):
            m = re.match(r".+_t\d+_(ok|fail)\.mp4$", name)
            if m:
                out.append((os.path.join(videos_dir, name), m.group(1)))
    return out


def execute_job(job):
    job_id = job["id"]
    params = job["params"]
    work_dir = os.path.join(_ROOT, "jobs", job_id)
    videos_dir = os.path.join(work_dir, "videos")
    cap_dir = os.path.join(work_dir, "cap")
    custom_dir = os.path.join(_ROOT, "videos", "custom")
    log_path = os.path.join(work_dir, "job.log")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(custom_dir, exist_ok=True)

    def fail(msg):
        job_update(job_id, status="failed", error=msg,
                   finished=time.time(),
                   log_tail=log_tail(log_path, CONFIG["LOG_TAIL_LINES"]))

    job_update(job_id, status="running", started=time.time())
    deadline = time.time() + CONFIG["JOB_TIMEOUT_S"]

    # 1) harness: one trial, video all -------------------------------------
    cmd = build_bench_cmd(params, work_dir)
    rc, timed_out = run_logged(cmd, log_path, CONFIG["JOB_TIMEOUT_S"],
                               CONFIG["BENCH_DIR"])
    job_update(job_id, log_tail=log_tail(log_path, CONFIG["LOG_TAIL_LINES"]))
    if timed_out:
        return fail("harness timed out after %ds" % CONFIG["JOB_TIMEOUT_S"])
    if rc != 0:
        return fail("harness exited rc=%d (see log_tail)" % rc)

    raw_videos = collect_videos(videos_dir)
    if not raw_videos:
        return fail("harness produced no video in %s" % videos_dir)

    # 2) caption ------------------------------------------------------------
    test = KINDS[params["kind"]]["test"]
    jsonl = os.path.join(work_dir,
                         JSONL_NAME[params["model"]].format(test=test))
    if not os.path.exists(jsonl):
        return fail("results jsonl missing: %s" % jsonl)
    os.makedirs(cap_dir, exist_ok=True)
    cap_cmd = [CONFIG["CAPTION_PYTHON"], CONFIG["CAPTION_SCRIPT"],
               "--pairs", "%s:%s" % (videos_dir, jsonl),
               "--out", cap_dir, "--all"]
    cap_timeout = max(60, int(deadline - time.time()) + 120)
    rc, timed_out = run_logged(cap_cmd, log_path, cap_timeout,
                               CONFIG["BENCH_DIR"])
    job_update(job_id, log_tail=log_tail(log_path, CONFIG["LOG_TAIL_LINES"]))

    # 3) publish into <root>/videos/custom/ (captioned, else raw fallback) --
    published = []
    cap_files = sorted(os.listdir(cap_dir)) if os.path.isdir(cap_dir) else []
    if rc == 0 and not timed_out and cap_files:
        sources = [(os.path.join(cap_dir, n),
                    "ok" if "_ok_" in n or n.endswith("_ok_cap.mp4")
                    else "fail") for n in cap_files if n.endswith(".mp4")]
    else:
        sources = raw_videos  # captioning failed: publish raw videos
    for src, tag in sources:
        dst_name = "%s_%s" % (job_id, os.path.basename(src))
        dst = os.path.join(custom_dir, dst_name)
        shutil.copyfile(src, dst)
        published.append({"file": dst_name,
                          "rel_path": "videos/custom/%s" % dst_name,
                          "outcome": "success" if tag == "ok" else "fail"})

    # 4) trial result row (success flag, fall info) -------------------------
    trial_row = None
    try:
        with open(jsonl) as f:
            first = f.readline().strip()
            trial_row = json.loads(first) if first else None
    except (OSError, ValueError):
        pass
    result = None
    if trial_row:
        result = {"success": bool(trial_row.get("success")),
                  "fall_time": trial_row.get("fall_time"),
                  "fall_phase": trial_row.get("fall_phase"),
                  "metrics": trial_row.get("metrics", {})}

    job_update(job_id, status="done", finished=time.time(),
               videos=published, result=result,
               log_tail=log_tail(log_path, CONFIG["LOG_TAIL_LINES"]))


def worker_loop():
    while True:
        job_id = _JOB_QUEUE.get()
        job = job_get(job_id)
        if job is None:
            continue
        try:
            execute_job(job)
        except Exception as exc:  # noqa: BLE001 — worker must never die
            job_update(job_id, status="failed", finished=time.time(),
                       error="internal: %s: %s" % (type(exc).__name__, exc))


# --------------------------------------------------------------------------
# HTTP handler: static site + JSON API (+ Range support for <video>)
# --------------------------------------------------------------------------
class ConsoleHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=_ROOT, **kw)

    # ---- helpers ----------------------------------------------------------
    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):  # quieter: only API + errors
        if "/api/" in (args[0] if args else "") or "POST" in fmt % args:
            super().log_message(fmt, *args)

    # ---- API routes --------------------------------------------------------
    def do_POST(self):
        if self.path.rstrip("/") != "/api/render":
            return self.send_json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            return self.send_json({"error": "invalid JSON body"}, 400)
        params, err = validate_render_request(body)
        if err:
            return self.send_json({"error": err}, 400)

        job_id = "j%s_%s" % (time.strftime("%Y%m%d_%H%M%S"),
                             uuid.uuid4().hex[:6])
        job = {"id": job_id, "params": params, "status": "queued",
               "created": time.time(), "started": None, "finished": None,
               "videos": [], "result": None, "error": None, "log_tail": []}
        with _JOBS_LOCK:
            _JOBS[job_id] = job
            _persist_jobs()
        _JOB_QUEUE.put(job_id)
        self.send_json({"job_id": job_id, "status": "queued",
                        "queue_len": _JOB_QUEUE.qsize()})

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/job/"):
            job = job_get(path[len("/api/job/"):].strip("/"))
            if job is None:
                return self.send_json({"error": "unknown job"}, 404)
            return self.send_json(job)
        if path.rstrip("/") == "/api/jobs":
            return self.send_json({"jobs": jobs_list()})
        if path.rstrip("/") == "/api/health":
            return self.send_json({"ok": True, "queue_len": _JOB_QUEUE.qsize()})
        return self.serve_static()

    def do_HEAD(self):
        if self.path.startswith("/api/"):
            return self.send_json({}, 405)
        return self.serve_static(head=True)

    # ---- static with single-Range support (Safari needs it for <video>) ---
    def serve_static(self, head=False):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            path = os.path.join(path, "index.html")
        if not os.path.isfile(path):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        ctype = self.guess_type(path)
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)$", rng.strip()) if rng else None
        start, end = 0, size - 1
        partial = False
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            else:  # suffix range: last N bytes
                start = max(0, size - int(m.group(2)))
            if start >= size:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */%d" % size)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            partial = True
        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range",
                             "bytes %d-%d/%d" % (start, end, size))
        if path.endswith((".html", ".json")):
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if head:
            return
        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)


# --------------------------------------------------------------------------
def main():
    global _ROOT
    ap = argparse.ArgumentParser(description="G1 console server")
    ap.add_argument("--port", type=int, default=8017)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--root", default="/sda/lizhe/g1bench/site",
                    help="site root (index.html / manifest.json / videos/)")
    args = ap.parse_args()
    _ROOT = os.path.abspath(args.root)
    if not os.path.isdir(_ROOT):
        raise SystemExit("site root does not exist: %s" % _ROOT)
    os.makedirs(os.path.join(_ROOT, "videos", "custom"), exist_ok=True)
    os.makedirs(os.path.join(_ROOT, "jobs"), exist_ok=True)
    _load_jobs()

    t = threading.Thread(target=worker_loop, daemon=True, name="render-worker")
    t.start()

    srv = ThreadingHTTPServer((args.host, args.port), ConsoleHandler)
    print("console server: http://%s:%d  root=%s  (%d jobs restored)"
          % (args.host, args.port, _ROOT, len(_JOBS)), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
