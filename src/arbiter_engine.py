"""ArbiterTtsEngine - delegates TTS synthesis to spark via the arbiter HTTP API.

Submits all chunks/lines at once to the arbiter queue, then polls until all complete.
The arbiter processes them sequentially per model but this avoids round-trip latency
between chunks.
"""
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

from src.tts_engine import (
    DEFAULT_SPEED,
    DEFAULT_TEMPERATURE,
    TtsEngine,
    concatenate_wav_files,
    convert_wav_to_mp3,
    enhance_output,
    get_voice_description,
    split_into_chunks,
    validate_multi_speaker_voices,
    parse_multi_speaker_jsonl,
    QWEN_VOICES,
)

ARBITER_BASE_URL = "http://10.0.0.254:8400"
SPARK_HOST = "10.0.0.254"
SPARK_USER = "darren"
SPARK_INBOX = "/tmp/arbiter-inbox"
POLL_INTERVAL_S = 0.5
JOB_TIMEOUT_S = 600  # 10 minutes total for all chunks


class ArbiterError(RuntimeError):
    """Base error for arbiter operations."""


class ArbiterUnreachableError(ArbiterError):
    """Arbiter is not reachable."""


def check_arbiter_reachable() -> None:
    """Verify arbiter is reachable."""
    try:
        req = urllib.request.Request(f"{ARBITER_BASE_URL}/v1/health", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if data.get("status") != "ok":
                raise ArbiterError(f"Arbiter unhealthy: {data}")
    except urllib.error.URLError as e:
        raise ArbiterUnreachableError(f"Arbiter not reachable at {ARBITER_BASE_URL}: {e}")
    except TimeoutError:
        raise ArbiterUnreachableError(f"Arbiter timed out at {ARBITER_BASE_URL}")


def _submit_job(job_type: str, params: dict) -> str:
    """Submit a job to arbiter, return job_id."""
    payload = json.dumps({"type": job_type, "params": params}).encode("utf-8")
    req = urllib.request.Request(
        f"{ARBITER_BASE_URL}/v1/jobs",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result["job_id"]
    except urllib.error.URLError as e:
        raise ArbiterError(f"Failed to submit {job_type} job: {e}")


def _poll_all_jobs(job_ids: list[str], timeout: float = JOB_TIMEOUT_S) -> list[dict]:
    """Poll all jobs until all complete, return results in order."""
    deadline = time.monotonic() + timeout
    results: list[dict | None] = [None] * len(job_ids)
    pending = set(range(len(job_ids)))

    while pending and time.monotonic() < deadline:
        for idx in list(pending):
            poll_url = f"{ARBITER_BASE_URL}/v1/jobs/{job_ids[idx]}"
            try:
                with urllib.request.urlopen(poll_url, timeout=30) as resp:
                    job_data = json.loads(resp.read())
            except urllib.error.URLError as e:
                raise ArbiterError(f"Failed to poll job {job_ids[idx]}: {e}")

            status = job_data.get("status")
            if status == "completed":
                results[idx] = job_data.get("result", {})
                pending.discard(idx)
                done = len(job_ids) - len(pending)
                print(f"  {done}/{len(job_ids)} complete", flush=True)
            elif status in ("failed", "cancelled"):
                error_msg = job_data.get("error", "unknown error")
                raise ArbiterError(f"Job {job_ids[idx]} {status}: {error_msg}")

        if pending:
            time.sleep(POLL_INTERVAL_S)

    if pending:
        raise ArbiterError(f"{len(pending)} jobs timed out after {timeout}s")

    return results  # type: ignore[return-value]


def _decode_wav_result(result: dict) -> bytes:
    """Decode base64 WAV data from arbiter result."""
    data_b64 = result.get("data")
    if not data_b64:
        raise ArbiterError("Arbiter returned no audio data")
    return base64.b64decode(data_b64)


def _scp_to_inbox(local_path: str) -> str:
    """SCP a file to spark's arbiter inbox, return remote path."""
    filename = f"{int(time.time())}_{Path(local_path).name}"
    remote_path = f"{SPARK_INBOX}/{filename}"
    result = subprocess.run(
        ["scp", "-q", "-o", "ConnectTimeout=10", local_path,
         f"{SPARK_USER}@{SPARK_HOST}:{remote_path}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise ArbiterError(f"SCP to arbiter inbox failed: {result.stderr.strip()}")
    return remote_path


# ##################################################################
# arbiter tts engine
# delegates synthesis to spark via arbiter HTTP API
# submits ALL chunks at once, then polls until all complete
class ArbiterTtsEngine(TtsEngine):
    def __init__(
        self,
        voice: str = "aiden",
        ref_audio: str | None = None,
        ref_text: str | None = None,
        voice_description: str | None = None,
    ):
        self.voice = voice
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.voice_description = voice_description

    def synthesize(
        self,
        text: str,
        output_path: Path,
        language: str = "English",
        temperature: float = DEFAULT_TEMPERATURE,
        speed: float = DEFAULT_SPEED,
        enhance: bool = False,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        check_arbiter_reachable()

        # split text into chunks
        chunks = split_into_chunks(text)
        total = len(chunks)
        print(f"Synthesizing {total} chunk(s) via arbiter...")

        # determine job type and base params
        job_type, base_params = self._build_job_params(language, temperature)

        # submit ALL chunks at once — arbiter queues them
        job_ids = []
        for chunk in chunks:
            params = {**base_params, "text": chunk}
            job_id = _submit_job(job_type, params)
            job_ids.append(job_id)
        print(f"  Submitted {total} jobs, waiting...", flush=True)

        # poll all until complete
        start = time.time()
        results = _poll_all_jobs(job_ids)
        elapsed = time.time() - start
        print(f"  All {total} chunks done ({elapsed:.1f}s)", flush=True)

        # decode wav data and write chunk files
        chunk_wavs: list[Path] = []
        try:
            for result in results:
                wav_bytes = _decode_wav_result(result)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(wav_bytes)
                    chunk_wavs.append(Path(tmp.name))

            # enhance each chunk if requested
            if enhance:
                for i, chunk_wav in enumerate(chunk_wavs):
                    print(f"  Enhancing chunk {i + 1}/{total}...")
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        enhanced_path = Path(tmp.name)
                    enhance_output(chunk_wav, enhanced_path)
                    chunk_wav.unlink()
                    chunk_wavs[i] = enhanced_path

            # concatenate chunks
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                combined_wav = Path(tmp.name)
            concatenate_wav_files(chunk_wavs, combined_wav)

            # convert to final format
            if output_path.suffix.lower() == ".mp3":
                convert_wav_to_mp3(combined_wav, output_path, normalize=True, speed=speed)
                combined_wav.unlink()
            elif speed != 1.0:
                speed = max(0.5, min(2.0, speed))
                cmd = ["ffmpeg", "-y", "-i", str(combined_wav), "-af", f"atempo={speed}", str(output_path)]
                ffmpeg_result = subprocess.run(cmd, capture_output=True, text=True)
                combined_wav.unlink()
                if ffmpeg_result.returncode != 0:
                    raise RuntimeError(f"Speed adjustment failed: {ffmpeg_result.stderr}")
            else:
                shutil.move(str(combined_wav), str(output_path))

        finally:
            for wav in chunk_wavs:
                if wav.exists():
                    wav.unlink()

        return output_path

    def _build_job_params(self, language: str, temperature: float) -> tuple[str, dict]:
        """Return (job_type, base_params) for the arbiter."""
        base = {"language": language, "temperature": temperature}

        if self.voice_description:
            return "tts-design", {**base, "voice_description": self.voice_description}

        if self.ref_audio:
            # SCP ref_audio to arbiter inbox for efficiency
            remote_path = _scp_to_inbox(self.ref_audio)
            params = {**base, "ref_audio_file": remote_path}
            if self.ref_text:
                params["ref_text"] = self.ref_text
            return "tts-clone", params

        # preset voice
        return "tts-custom", {**base, "speaker": self.voice.title()}


# ##################################################################
# get arbiter engine
# factory function — the default way to get a TTS engine
def get_arbiter_engine(voice: str = "aiden", voice_description: str | None = None) -> ArbiterTtsEngine:
    """Get an ArbiterTtsEngine for the given voice."""
    # zip file (portable voice package)
    if voice.endswith(".zip"):
        import zipfile

        zip_file = Path(voice)
        if not zip_file.exists():
            raise FileNotFoundError(f"Voice package not found: {voice}")
        extract_dir = Path(tempfile.mkdtemp(prefix="voice_pkg_"))
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(extract_dir)
        voice_wav = extract_dir / "voice.wav"
        voice_txt = extract_dir / "voice.txt"
        if not voice_wav.exists():
            raise ValueError(f"Voice package missing voice.wav: {voice}")
        ref_text = voice_txt.read_text().strip() if voice_txt.exists() else None
        return ArbiterTtsEngine(ref_audio=str(voice_wav), ref_text=ref_text)

    # registered custom voice
    custom_voice = get_voice_description(voice)
    if custom_voice:
        voice_type = custom_voice.get("type", "description")
        if voice_type == "clone":
            return ArbiterTtsEngine(
                ref_audio=custom_voice["ref_audio"],
                ref_text=custom_voice.get("ref_text", ""),
            )
        else:
            return ArbiterTtsEngine(voice_description=custom_voice["description"])

    # explicit voice description
    if voice_description:
        return ArbiterTtsEngine(voice_description=voice_description)

    # standard preset voice
    if voice.lower() not in [v.lower() for v in QWEN_VOICES]:
        raise ValueError(f"Unknown voice: {voice}. Supported: {', '.join(QWEN_VOICES)}")
    return ArbiterTtsEngine(voice=voice)


# ##################################################################
# multi-speaker synthesis via arbiter
# submits ALL lines at once, then polls until all complete
def synthesize_multi_speaker_arbiter(
    jsonl_path: Path,
    output_path: Path,
    language: str = "English",
    temperature: float = DEFAULT_TEMPERATURE,
    speed: float = DEFAULT_SPEED,
    enhance: bool = False,
) -> Path:
    """Synthesize multi-speaker dialogue via arbiter."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    segments = parse_multi_speaker_jsonl(jsonl_path)
    if not segments:
        raise ValueError(f"No dialogue found in {jsonl_path}")
    validate_multi_speaker_voices(segments)

    total_lines = len(segments)
    print(f"\nMulti-speaker synthesis via arbiter: {total_lines} lines")

    check_arbiter_reachable()

    work_dir = Path(tempfile.mkdtemp(prefix="arbiter_multi_"))
    total_start = time.time()

    try:
        # submit ALL lines at once
        job_ids: list[str] = []
        for voice, text in segments:
            # build params for this voice
            custom_voice = get_voice_description(voice)
            if custom_voice and custom_voice.get("type") == "clone":
                job_type = "tts-clone"
                remote_ref = _scp_to_inbox(custom_voice["ref_audio"])
                params = {
                    "text": text,
                    "language": language,
                    "temperature": temperature,
                    "ref_audio_file": remote_ref,
                }
                if custom_voice.get("ref_text"):
                    params["ref_text"] = custom_voice["ref_text"]
            elif custom_voice and custom_voice.get("description"):
                job_type = "tts-design"
                params = {
                    "text": text,
                    "language": language,
                    "temperature": temperature,
                    "voice_description": custom_voice["description"],
                }
            else:
                job_type = "tts-custom"
                params = {
                    "text": text,
                    "language": language,
                    "temperature": temperature,
                    "speaker": voice.title(),
                }

            # split long lines into chunks and submit each
            chunks = split_into_chunks(text)
            if len(chunks) == 1:
                job_id = _submit_job(job_type, params)
                job_ids.append(job_id)
            else:
                # for multi-chunk lines, submit each chunk separately
                # we'll track which jobs belong to which line
                chunk_job_ids = []
                for chunk in chunks:
                    chunk_params = {**params, "text": chunk}
                    job_id = _submit_job(job_type, chunk_params)
                    chunk_job_ids.append(job_id)
                # store as tuple marker — we'll handle reassembly below
                job_ids.append(tuple(chunk_job_ids))

        print("  Submitted all jobs, waiting...", flush=True)

        # flatten job_ids for polling
        flat_ids: list[str] = []
        for item in job_ids:
            if isinstance(item, tuple):
                flat_ids.extend(item)
            else:
                flat_ids.append(item)

        # poll all
        flat_results = _poll_all_jobs(flat_ids, timeout=total_lines * 300)

        # reassemble results per line
        flat_idx = 0
        for line_idx, item in enumerate(job_ids):
            line_wav = work_dir / f"line_{line_idx:04d}.wav"
            if isinstance(item, tuple):
                # multi-chunk line — concatenate chunks
                chunk_wavs = []
                for _ in item:
                    wav_bytes = _decode_wav_result(flat_results[flat_idx])
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(work_dir)) as tmp:
                        tmp.write(wav_bytes)
                        chunk_wavs.append(Path(tmp.name))
                    flat_idx += 1
                concatenate_wav_files(chunk_wavs, line_wav)
                for cw in chunk_wavs:
                    cw.unlink()
            else:
                wav_bytes = _decode_wav_result(flat_results[flat_idx])
                line_wav.write_bytes(wav_bytes)
                flat_idx += 1

        # concatenate with silence trimming
        print(f"Trimming silence and concatenating {total_lines} lines...")
        line_wavs = [work_dir / f"line_{i:04d}.wav" for i in range(total_lines)]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            combined_wav = Path(tmp.name)
        concatenate_wav_files(line_wavs, combined_wav, trim_trailing_silence=True)

        if enhance:
            print("Enhancing combined audio...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                enhanced_wav = Path(tmp.name)
            enhance_output(combined_wav, enhanced_wav)
            combined_wav.unlink()
            combined_wav = enhanced_wav

        if output_path.suffix.lower() == ".mp3":
            convert_wav_to_mp3(combined_wav, output_path, normalize=True, speed=speed)
            combined_wav.unlink()
        else:
            shutil.move(str(combined_wav), str(output_path))

        total_elapsed = time.time() - total_start
        print(f"\nMulti-speaker complete! {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
        print(f"Output: {output_path}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return output_path
