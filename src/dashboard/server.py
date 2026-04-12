"""
Curiosity Dashboard — FastAPI + SSE backend.
Serves real-time system state for the Curiosity autonomous self-improvement system.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths — all relative to $HOME/curiosity on the DGX host
# ---------------------------------------------------------------------------
HOME = Path.home()
CURIOSITY_DIR = HOME / "curiosity"
LOGS_DIR = CURIOSITY_DIR / "logs"
METRICS_FILE = CURIOSITY_DIR / "metrics" / "capability_scores.jsonl"
OUTCOMES_PASS_DIR = CURIOSITY_DIR / "memory" / "outcomes" / "pass"

DAEMONS = ["assessor", "formulator", "solver", "verifier", "memorizer"]
QUEUES = ["ASSESS_QUEUE", "FORMULATE_QUEUE", "SOLVE_QUEUE", "VERIFY_QUEUE", "MEMORIZE_QUEUE"]
QUEUE_SHORT = ["ASSESS", "FORMULATE", "SOLVE", "VERIFY", "MEMORIZE"]

# ---------------------------------------------------------------------------
# Optional Redis
# ---------------------------------------------------------------------------
try:
    import redis as _redis  # type: ignore

    _rc = _redis.Redis(host="localhost", port=6379, decode_responses=True, socket_timeout=1)
    _rc.ping()
    REDIS_AVAILABLE = True
except Exception:
    _rc = None  # type: ignore
    REDIS_AVAILABLE = False

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Curiosity Dashboard")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

_START_TIME = time.time()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uptime_str() -> str:
    secs = int(time.time() - _START_TIME)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    return f"{m}m {s}s"


def _daemon_status(name: str) -> str:
    pid_file = CURIOSITY_DIR / f"{name}.pid"
    if not pid_file.exists():
        return "stopped"
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return "running"
    except (ValueError, ProcessLookupError, PermissionError):
        return "stopped"


def _queue_depths() -> dict:
    depths = {}
    for short, full in zip(QUEUE_SHORT, QUEUES):
        if REDIS_AVAILABLE and _rc is not None:
            try:
                info = _rc.xinfo_stream(full)
                depths[short] = int(info.get("length", 0))
            except Exception:
                depths[short] = 0
        else:
            depths[short] = 0
    return depths


def _current_problem() -> Optional[dict]:
    """Peek at the latest message in FORMULATE_QUEUE."""
    if not (REDIS_AVAILABLE and _rc is not None):
        return None
    try:
        msgs = _rc.xrevrange("FORMULATE_QUEUE", count=1)
        if not msgs:
            return None
        _msg_id, fields = msgs[0]
        return {
            "domain": fields.get("domain", "unknown"),
            "description": fields.get("description", fields.get("problem", "")),
            "priority": float(fields.get("priority", 0.5)),
        }
    except Exception:
        return None


def _capability_scores() -> dict:
    """Return latest score per domain from capability_scores.jsonl."""
    scores: dict = {}
    if not METRICS_FILE.exists():
        return scores
    try:
        lines = METRICS_FILE.read_text().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                domain = entry.get("domain") or entry.get("name")
                score = entry.get("score") or entry.get("capability_score")
                if domain and score is not None and domain not in scores:
                    scores[domain] = round(float(score), 4)
            except json.JSONDecodeError:
                continue
            if len(scores) >= 10:
                break
    except Exception:
        pass
    return scores


def _recent_solved(limit: int = 10) -> list:
    """Return recent solved problems from the pass outcomes directory."""
    items = []
    if not OUTCOMES_PASS_DIR.exists():
        return items
    try:
        files = sorted(OUTCOMES_PASS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files[:limit]:
            try:
                data = json.loads(f.read_text())
                items.append({
                    "domain": data.get("domain", "unknown"),
                    "score": round(float(data.get("score", data.get("capability_score", 0))), 4),
                    "timestamp": int(f.stat().st_mtime),
                    "description": data.get("description", data.get("problem", ""))[:120],
                })
            except Exception:
                continue
    except Exception:
        pass
    return items


def _build_state() -> dict:
    daemons = {d: _daemon_status(d) for d in DAEMONS}
    queues = _queue_depths()
    scores = _capability_scores()
    recent = _recent_solved(10)
    problem = _current_problem()

    return {
        "type": "status_update",
        "timestamp": int(time.time()),
        "uptime": _uptime_str(),
        "queues": queues,
        "daemons": daemons,
        "current_problem": problem,
        "recent_solved": recent,
        "capability_scores": scores,
        "total_solved": len(list(OUTCOMES_PASS_DIR.glob("*.json"))) if OUTCOMES_PASS_DIR.exists() else 0,
        "redis_available": REDIS_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text())
    return HTMLResponse(content="<h1>Curiosity Dashboard</h1><p>index.html not found.</p>")


@app.get("/api/status")
async def api_status():
    return JSONResponse(_build_state())


@app.get("/api/metrics")
async def api_metrics():
    """Return full capability_scores.jsonl as list."""
    entries = []
    if METRICS_FILE.exists():
        for line in METRICS_FILE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return JSONResponse({"metrics": entries})


@app.get("/api/memory")
async def api_memory():
    return JSONResponse({"recent_solved": _recent_solved(20)})


@app.get("/api/queues")
async def api_queues():
    return JSONResponse({"queues": _queue_depths(), "redis_available": REDIS_AVAILABLE})


async def _event_generator() -> AsyncGenerator[str, None]:
    """Stream JSON status events every 2 seconds."""
    while True:
        try:
            state = _build_state()
            payload = json.dumps(state)
            yield f"data: {payload}\n\n"
        except Exception as exc:
            error_payload = json.dumps({"type": "error", "message": str(exc)})
            yield f"data: {error_payload}\n\n"
        await asyncio.sleep(2)


@app.get("/events")
async def events():
    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
