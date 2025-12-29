#!/usr/bin/env python3
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "index.html"
STATIC_ALLOWLIST = {
    "Framework.png": ROOT / "Framework.png",
    "classes_LLMGGP.png": ROOT / "classes_LLMGGP.png",
    "packages_LLMGGP.png": ROOT / "packages_LLMGGP.png",
}


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class JobRecord:
    id: str
    task: str
    prompt: str
    scheme: Optional[str] = None
    status: str = "queued"
    created_at: str = field(default_factory=_now_iso)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    traceback: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "prompt": self.prompt,
            "scheme": self.scheme,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback,
        }


class JobRequest(BaseModel):
    task: str = Field(..., pattern="^(ycs|crp)$")
    prompt: str = Field(..., min_length=1, max_length=4000)
    scheme: Optional[str] = Field(default=None, pattern="^(RE|REN|UN)$")


app = FastAPI(title="LLMGGP Demo")

_jobs: Dict[str, JobRecord] = {}
_jobs_lock = Lock()


def _set_job(job_id: str, **updates: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        for key, value in updates.items():
            setattr(job, key, value)


def _run_job(job_id: str, task: str, prompt: str, scheme: Optional[str]) -> None:
    _set_job(job_id, status="running", started_at=_now_iso())
    try:
        prompt = prompt.strip()
        if task == "ycs":
            from ycs.simulator_standalone import build_ycs_simulator  # lazy import to avoid startup failures

            sim = build_ycs_simulator(
                data_dir=ROOT / "ycs" / "train_instances",
                prompt=prompt,
            )
        else:
            from crp.simulator_standalone import build_crp_simulator  # lazy import to avoid startup failures

            sim = build_crp_simulator(
                data_dir=ROOT / "crp" / "clean",
                scheme=scheme or "RE",
                prompt=prompt,
            )
        result = sim.run()
        best = result.hof[0]
        job_result = {
            "best_expr": str(best),
            "sim_score": getattr(best, "sim_score", None),
            "fitness": best.fitness.values[0] if best.fitness.values else None,
        }
        _set_job(
            job_id,
            status="completed",
            finished_at=_now_iso(),
            result=job_result,
        )
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            finished_at=_now_iso(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


@app.get("/", response_class=HTMLResponse)
def read_index() -> HTMLResponse:
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/assets/{filename}")
def read_asset(filename: str):
    path = STATIC_ALLOWLIST.get(filename)
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(path)


@app.post("/api/jobs")
def create_job(payload: JobRequest):
    job_id = uuid4().hex
    job = JobRecord(
        id=job_id,
        task=payload.task,
        prompt=payload.prompt.strip(),
        scheme=payload.scheme,
    )
    with _jobs_lock:
        _jobs[job_id] = job
    thread = Thread(
        target=_run_job,
        args=(job_id, payload.task, payload.prompt, payload.scheme),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict()
