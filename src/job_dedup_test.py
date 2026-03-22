import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.job_dedup import (
    compute_job_id,
    read_job_status,
    write_job_status,
    prune_old_jobs,
    JobStatus,
)


# ##################################################################
# test compute job id is deterministic
# same output path always produces the same job id
def test_compute_job_id_is_deterministic():
    path = "/some/output/audio.mp3"
    assert compute_job_id(path) == compute_job_id(path)


# ##################################################################
# test compute job id uses sha256 prefix
# verify job id matches sha256(path)[:16]
def test_compute_job_id_uses_sha256_prefix():
    path = "/some/output/audio.mp3"
    expected = hashlib.sha256(path.encode()).hexdigest()[:16]
    assert compute_job_id(path) == expected


# ##################################################################
# test compute job id differs for different paths
# different output paths produce different job ids
def test_compute_job_id_differs_for_different_paths():
    a = compute_job_id("/output/a.mp3")
    b = compute_job_id("/output/b.mp3")
    assert a != b


# ##################################################################
# test read job status returns none for missing file
# no job file = no status
def test_read_job_status_returns_none_for_missing_file(tmp_path: Path):
    status = read_job_status(tmp_path, "nonexistent_id")
    assert status is None


# ##################################################################
# test write then read job status running
# written running job can be read back correctly
def test_write_then_read_job_status_running(tmp_path: Path):
    write_job_status(
        tmp_path,
        "abc123",
        JobStatus(
            status="running",
            output_path="/out/test.mp3",
            caller="state-flow/test",
            voice="gary",
            started_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    result = read_job_status(tmp_path, "abc123")
    assert result is not None
    assert result.status == "running"
    assert result.caller == "state-flow/test"
    assert result.voice == "gary"
    assert result.output_path == "/out/test.mp3"


# ##################################################################
# test write then read job status done
# written done job can be read back with completed_at
def test_write_then_read_job_status_done(tmp_path: Path):
    completed = datetime.now(timezone.utc).isoformat()
    write_job_status(
        tmp_path,
        "done123",
        JobStatus(
            status="done",
            output_path="/out/done.mp3",
            caller="state-flow/done_cell",
            voice="aiden",
            started_at=datetime.now(timezone.utc).isoformat(),
            completed_at=completed,
        ),
    )
    result = read_job_status(tmp_path, "done123")
    assert result is not None
    assert result.status == "done"
    assert result.completed_at == completed


# ##################################################################
# test write is atomic (no partial reads)
# status file is written via tmp+rename so partial writes never happen
def test_write_is_atomic(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    job_id = "atomic99"
    write_job_status(
        jobs_dir,
        job_id,
        JobStatus(
            status="running",
            output_path="/out/x.mp3",
            caller="test",
            voice="gary",
            started_at=datetime.now(timezone.utc).isoformat(),
        ),
    )
    # the .tmp file must not exist — only the final file
    tmp_file = jobs_dir / f"{job_id}.json.tmp"
    final_file = jobs_dir / f"{job_id}.json"
    assert not tmp_file.exists()
    assert final_file.exists()


# ##################################################################
# test prune old jobs removes completed older than 24h
# done jobs with completed_at > 24h ago are deleted
def test_prune_old_jobs_removes_old_done(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    # old completed job (25 hours ago)
    old_time = datetime.fromtimestamp(time.time() - 25 * 3600, tz=timezone.utc).isoformat()
    write_job_status(
        jobs_dir,
        "old_done",
        JobStatus(
            status="done",
            output_path="/out/old.mp3",
            caller="test",
            voice="gary",
            started_at=old_time,
            completed_at=old_time,
        ),
    )

    prune_old_jobs(jobs_dir)

    assert not (jobs_dir / "old_done.json").exists()


# ##################################################################
# test prune old jobs keeps recent done jobs
# done jobs completed < 24h ago are kept
def test_prune_old_jobs_keeps_recent_done(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    recent_time = datetime.now(timezone.utc).isoformat()
    write_job_status(
        jobs_dir,
        "recent_done",
        JobStatus(
            status="done",
            output_path="/out/recent.mp3",
            caller="test",
            voice="gary",
            started_at=recent_time,
            completed_at=recent_time,
        ),
    )

    prune_old_jobs(jobs_dir)

    assert (jobs_dir / "recent_done.json").exists()


# ##################################################################
# test prune old jobs keeps running jobs
# running jobs are never pruned regardless of age
def test_prune_old_jobs_keeps_running(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    old_time = datetime.fromtimestamp(time.time() - 48 * 3600, tz=timezone.utc).isoformat()
    write_job_status(
        jobs_dir,
        "old_running",
        JobStatus(
            status="running",
            output_path="/out/running.mp3",
            caller="test",
            voice="gary",
            started_at=old_time,
        ),
    )

    prune_old_jobs(jobs_dir)

    assert (jobs_dir / "old_running.json").exists()


# ##################################################################
# test prune old jobs skips non json files
# non-.json files in jobs dir are not touched
def test_prune_old_jobs_skips_non_json(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    noise = jobs_dir / "something.txt"
    noise.write_text("ignored")

    prune_old_jobs(jobs_dir)

    assert noise.exists()


# ##################################################################
# test job status missing completed_at treated as running
# a status file without completed_at is treated as running during prune
def test_prune_ignores_job_status_without_completed_at(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()

    # write a malformed 'done' status (no completed_at)
    job_file = jobs_dir / "malformed.json"
    job_file.write_text(
        json.dumps(
            {
                "status": "done",
                "output_path": "/out/malformed.mp3",
                "caller": "test",
                "voice": "gary",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    )

    prune_old_jobs(jobs_dir)

    # kept (can't determine age without completed_at)
    assert job_file.exists()


# ##################################################################
# job dedup tests
# unit tests for job deduplication utilities: id hashing, status persistence, pruning
