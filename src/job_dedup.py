import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


# ##################################################################
# job status dataclass
# holds all fields for a persisted TTS job status file
@dataclass
class JobStatus:
    status: str  # "running" | "done"
    output_path: str
    caller: str
    voice: str
    started_at: str  # ISO 8601 timestamp
    completed_at: str | None = None


# ##################################################################
# compute job id
# deterministic 16-char hex id derived from output path via sha256
def compute_job_id(output_path: str) -> str:
    return hashlib.sha256(output_path.encode()).hexdigest()[:16]
    # ##################################################################
    # compute job id
    # maps output path to a stable 16-char hex id for status file naming


# ##################################################################
# read job status
# load and parse a job status file; returns None if missing or corrupt
def read_job_status(jobs_dir: Path, job_id: str) -> JobStatus | None:
    job_file = jobs_dir / f"{job_id}.json"
    if not job_file.exists():
        return None
    try:
        data = json.loads(job_file.read_text(encoding="utf-8"))
        return JobStatus(
            status=data.get("status", ""),
            output_path=data.get("output_path", ""),
            caller=data.get("caller", ""),
            voice=data.get("voice", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at"),
        )
    except (json.JSONDecodeError, OSError):
        return None
    # ##################################################################
    # read job status
    # deserialises the status json file for the given job id


# ##################################################################
# write job status
# atomically persist a job status to local/jobs/{job_id}.json via tmp+rename
def write_job_status(jobs_dir: Path, job_id: str, status: JobStatus) -> None:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_file = jobs_dir / f"{job_id}.json"
    tmp_file = jobs_dir / f"{job_id}.json.tmp"
    data = {k: v for k, v in asdict(status).items() if v is not None}
    tmp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.rename(str(tmp_file), str(job_file))
    # ##################################################################
    # write job status
    # writes to .tmp then renames so readers never see a partial file


# ##################################################################
# prune old jobs
# delete done job files whose completed_at is more than 24 hours ago
def prune_old_jobs(jobs_dir: Path) -> None:
    if not jobs_dir.exists():
        return
    cutoff = time.time() - 24 * 3600
    for job_file in jobs_dir.iterdir():
        if job_file.suffix != ".json":
            continue
        try:
            data = json.loads(job_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("status") != "done":
            continue
        completed_at = data.get("completed_at")
        if not completed_at:
            continue
        try:
            ts = datetime.fromisoformat(completed_at).timestamp()
        except (ValueError, TypeError):
            continue
        if ts < cutoff:
            try:
                job_file.unlink()
            except OSError:
                pass
    # ##################################################################
    # prune old jobs
    # iterates jobs dir, removes completed entries older than 24h
