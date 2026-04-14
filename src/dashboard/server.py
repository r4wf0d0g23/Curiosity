"""
Curiosity Dashboard — Backend v2
Clean rewrite: correct queue lag metrics, in-process throughput tracking,
polling-friendly /api/status endpoint.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests as _requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HOME          = Path.home()
CURIOSITY_DIR = HOME / "curiosity"
LOGS_DIR      = CURIOSITY_DIR / "logs"
METRICS_FILE  = CURIOSITY_DIR / "metrics" / "capability_scores.jsonl"
OUTCOMES_DIR  = CURIOSITY_DIR / "memory" / "outcomes"
PROMPT_FILE   = HOME / "curiosity_code" / "config" / "system_prompt.txt"

DAEMONS = ["assessor", "formulator", "solver", "verifier", "memorizer", "trainer"]

# Queue → consumer group that consumes it (for lag calculation)
# ASSESS_QUEUE and MEMORIZE_QUEUE have no consumer group → use XLEN
QUEUE_CONFIG = {
    "ASSESS_QUEUE":    {"short": "ASSESS",    "group": None},
    "FORMULATE_QUEUE": {"short": "FORMULATE", "group": "solvers"},
    "SOLVE_QUEUE":     {"short": "SOLVE",     "group": "verifiers"},
    "MEMORIZE_QUEUE":  {"short": "MEMORIZE",  "group": None},
    "TRAIN_QUEUE":     {"short": "TRAIN",     "group": "trainers"},
}

# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
try:
    import redis as _redis
    _rc = _redis.Redis(host="localhost", port=6379, decode_responses=True, socket_timeout=2)
    _rc.ping()
    REDIS_OK = True
except Exception:
    _rc = None
    REDIS_OK = False

# ---------------------------------------------------------------------------
# In-process throughput state
# ---------------------------------------------------------------------------
_throughput_state: dict = {}   # {"ts": float, "verify": int, "train": int, "queues": dict}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Curiosity Dashboard v2")
_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
_START = time.time()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uptime() -> str:
    s = int(time.time() - _START)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m" if h else f"{m}m {s}s"


def _daemon_running(name: str) -> bool:
    try:
        r = subprocess.run(
            ["pgrep", "-f", f"src.{name}.{name}"],
            capture_output=True, text=True, timeout=2
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


def _daemon_last_active(name: str) -> str:
    log = LOGS_DIR / f"{name}.log"
    if not log.exists():
        return "never"
    try:
        # Read tail 4KB — fast
        size = log.stat().st_size
        with log.open("rb") as f:
            f.seek(max(0, size - 4096))
            chunk = f.read().decode("utf-8", errors="replace")
        for line in reversed(chunk.splitlines()):
            line = line.strip()
            if len(line) < 23:
                continue
            try:
                dt = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")
                return dt.strftime("%H:%M:%S")
            except ValueError:
                continue
    except Exception:
        pass
    return "?"


def _queue_depths() -> dict[str, int]:
    depths: dict[str, int] = {}
    if not REDIS_OK or _rc is None:
        return {cfg["short"]: 0 for cfg in QUEUE_CONFIG.values()}

    for stream, cfg in QUEUE_CONFIG.items():
        short = cfg["short"]
        group = cfg["group"]
        try:
            if group:
                # Use consumer group lag — messages not yet delivered to any consumer
                groups = _rc.xinfo_groups(stream)
                lag = next(
                    (int(g.get("lag") or 0) for g in groups if g["name"] == group),
                    0
                )
                depths[short] = lag
            else:
                # No consumer group — use raw stream length
                depths[short] = _rc.xlen(stream)
        except Exception:
            depths[short] = 0

    return depths


def _throughput() -> dict[str, int]:
    """
    Items/min per daemon, computed from deltas of Redis counters and queue lags.
    Called once per status build; updates internal state.
    """
    global _throughput_state
    now = time.time()
    result = {d: 0 for d in DAEMONS}

    if not REDIS_OK or _rc is None:
        return result

    # Counters we can read directly
    try:
        verify_now = int(_rc.get("VERIFY_COUNT") or 0)
    except Exception:
        verify_now = 0
    try:
        train_now = int(_rc.get("TRAIN_COUNT") or 0)
    except Exception:
        train_now = 0

    # Queue lag snapshots for solver/formulator throughput estimation
    queue_now: dict[str, int] = {}
    for stream, cfg in QUEUE_CONFIG.items():
        short = cfg["short"]
        group = cfg["group"]
        try:
            if group:
                groups = _rc.xinfo_groups(stream)
                lag = next(
                    (int(g.get("lag") or 0) for g in groups if g["name"] == group),
                    0
                )
                queue_now[short] = lag
            else:
                queue_now[short] = _rc.xlen(stream)
        except Exception:
            queue_now[short] = 0

    prev = _throughput_state
    if prev.get("ts"):
        elapsed = max(5, now - prev["ts"])   # avoid division by tiny number
        per_min = 60 / elapsed

        # Verifier: delta of VERIFY_COUNT
        v_delta = max(0, verify_now - prev.get("verify", verify_now))
        result["verifier"] = round(v_delta * per_min)

        # Trainer: delta of TRAIN_COUNT
        t_delta = max(0, train_now - prev.get("train", train_now))
        result["trainer"] = round(t_delta * per_min)

        # Solver: consumed from FORMULATE_QUEUE lag decrease
        fm_prev = prev.get("queues", {}).get("FORMULATE", queue_now.get("FORMULATE", 0))
        fm_delta = max(0, fm_prev - queue_now.get("FORMULATE", fm_prev))
        result["solver"] = round(fm_delta * per_min)

        # Formulator: consumed from ASSESS_QUEUE lag decrease
        as_prev = prev.get("queues", {}).get("ASSESS", queue_now.get("ASSESS", 0))
        as_delta = max(0, as_prev - queue_now.get("ASSESS", as_prev))
        result["formulator"] = round(as_delta * per_min)

        # Memorizer: from MEMORIZE_QUEUE decrease
        mem_prev = prev.get("queues", {}).get("MEMORIZE", queue_now.get("MEMORIZE", 0))
        mem_delta = max(0, mem_prev - queue_now.get("MEMORIZE", mem_prev))
        result["memorizer"] = round(mem_delta * per_min)

    # Update state
    _throughput_state = {
        "ts": now,
        "verify": verify_now,
        "train": train_now,
        "queues": queue_now,
    }
    return result


def _capability_scores() -> dict:
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
                e = json.loads(line)
                domain = e.get("domain") or e.get("name")
                score = e.get("pass_rate") or e.get("score") or e.get("capability_score")
                if domain and score is not None and domain not in scores:
                    scores[domain] = {
                        "score": round(float(score), 4),
                        "n": int(e.get("sample_size") or 0),
                        "delta": round(float(e.get("delta") or 0), 4),
                    }
            except Exception:
                continue
            if len(scores) >= 15:
                break
    except Exception:
        pass
    return scores


def _recent_outcomes(limit: int = 20) -> list:
    items = []
    files = []
    for sub in ("pass", "fail"):
        d = OUTCOMES_DIR / sub
        if d.exists():
            files.extend(d.glob("*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[:limit]:
        try:
            data = json.loads(f.read_text())
            prob  = data.get("problem", {})
            res   = data.get("result", {})
            sol   = data.get("solution", {})
            items.append({
                "domain":   prob.get("domain") or data.get("domain", "?"),
                "outcome":  res.get("outcome") or ("pass" if "pass" in f.parent.name else "fail"),
                "score":    round(float(res.get("criterion_score") or data.get("score") or 0), 3),
                "approach": sol.get("approach") or data.get("approach") or "?",
                "failure":  (res.get("failure_mode") or "")[:120],
                "ts":       int(f.stat().st_mtime),
            })
        except Exception:
            continue
    return items


def _approach_dist() -> dict:
    counts: dict = {}
    files = []
    for sub in ("pass", "fail"):
        d = OUTCOMES_DIR / sub
        if d.exists():
            files.extend(d.glob("*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for f in files[:200]:
        try:
            data = json.loads(f.read_text())
            a = data.get("solution", {}).get("approach") or "?"
            counts[a] = counts.get(a, 0) + 1
        except Exception:
            continue
    return counts


def _totals() -> tuple[int, int]:
    p = len(list((OUTCOMES_DIR / "pass").glob("*.json"))) if (OUTCOMES_DIR / "pass").exists() else 0
    f = len(list((OUTCOMES_DIR / "fail").glob("*.json"))) if (OUTCOMES_DIR / "fail").exists() else 0
    return p, f


def _vllm() -> dict:
    try:
        r = _requests.get("http://localhost:8001/v1/models", timeout=2)
        d = r.json()
        model = d["data"][0]["id"] if d.get("data") else "?"
        return {"ok": True, "model": model}
    except Exception:
        return {"ok": False, "model": None}


def _prompt_sha() -> str:
    try:
        return hashlib.sha256(PROMPT_FILE.read_text().encode()).hexdigest()[:8]
    except Exception:
        return "?"


def _prompt_preview() -> str:
    try:
        t = PROMPT_FILE.read_text().strip()
        return t[:200] if t else "(empty)"
    except Exception:
        return "?"


def _assessor_cycle() -> int:
    if not METRICS_FILE.exists():
        return 0
    best = 0
    try:
        for line in METRICS_FILE.read_text().splitlines():
            try:
                e = json.loads(line.strip())
                c = int(e.get("cycle") or 0)
                if c > best:
                    best = c
            except Exception:
                continue
    except Exception:
        pass
    return best


def _training_active() -> bool:
    if not REDIS_OK or _rc is None:
        return False
    try:
        return bool(_rc.exists("CURIOSITY_TRAINING_LOCK"))
    except Exception:
        return False


def _training_progress() -> dict:
    """Read live training progress from Redis hash."""
    if not REDIS_OK or _rc is None:
        return {}
    try:
        h = _rc.hgetall("CURIOSITY_TRAINING_PROGRESS")
        if not h:
            return {}
        out = {}
        for k, v in h.items():
            key = k.decode() if isinstance(k, bytes) else k
            val = v.decode() if isinstance(v, bytes) else v
            # cast numeric fields
            if key in ("pairs_done", "pairs_total", "batch", "errors", "ts"):
                try: val = int(val)
                except ValueError: pass
            out[key] = val
        if "pairs_done" in out and "pairs_total" in out and out["pairs_total"]:
            out["pct"] = round(100 * out["pairs_done"] / out["pairs_total"])
        return out
    except Exception:
        return {}


def _active_adapters() -> list:
    if not REDIS_OK or _rc is None:
        return []
    try:
        h = _rc.hgetall("CURIOSITY_ACTIVE_ADAPTERS")
        out = []
        for name, info_str in h.items():
            try:
                out.append(json.loads(info_str))
            except Exception:
                out.append({"name": name})
        return out
    except Exception:
        return []


def _build_status() -> dict:
    queues   = _queue_depths()
    tput     = _throughput()
    tp, tf   = _totals()
    total    = tp + tf
    daemons  = {d: "running" if _daemon_running(d) else "stopped" for d in DAEMONS}
    activity = {d: _daemon_last_active(d) for d in DAEMONS}

    return {
        "ts":        int(time.time()),
        "uptime":    _uptime(),
        "cycle":     _assessor_cycle(),
        "queues":    queues,
        "daemons":   daemons,
        "activity":  activity,
        "throughput": tput,
        "scores":    _capability_scores(),
        "recent":    _recent_outcomes(20),
        "approach":  _approach_dist(),
        "total_pass": tp,
        "total_fail": tf,
        "pass_rate": round(tp / total, 3) if total else 0,
        "vllm":      _vllm(),
        "prompt_sha":     _prompt_sha(),
        "prompt_preview": _prompt_preview(),
        "training_active": _training_active(),
        "training_progress": _training_progress(),
        "adapters":  _active_adapters(),
        "redis_ok":  REDIS_OK,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    f = _STATIC / "index.html"
    return HTMLResponse(content=f.read_text() if f.exists() else "<h1>index.html missing</h1>")


@app.get("/api/status")
async def api_status():
    return JSONResponse(_build_status())


@app.get("/api/prompt")
async def api_prompt():
    return JSONResponse({"sha": _prompt_sha(), "content": _prompt_preview()})


@app.get("/api/queues")
async def api_queues():
    return JSONResponse({"queues": _queue_depths(), "redis": REDIS_OK})


@app.get("/api/recent")
async def api_recent():
    return JSONResponse({"recent": _recent_outcomes(30)})


async def _sse_gen():
    while True:
        try:
            payload = json.dumps(_build_status())
            yield f"data: {payload}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        await asyncio.sleep(3)


@app.get("/events")
async def events():
    return StreamingResponse(
        _sse_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
