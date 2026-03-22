"""SparkTtsEngine - delegates TTS synthesis to spark (NVIDIA CUDA) via SSH/SCP."""

import json
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from src.tts_engine import (
    DEFAULT_SPEED,
    DEFAULT_TEMPERATURE,
    TtsEngine,
    concatenate_wav_files,
    convert_wav_to_mp3,
    enhance_output,
    get_voice_description,
    parse_multi_speaker_jsonl,
    split_into_chunks,
    validate_multi_speaker_voices,
    QWEN_VOICES,
)

SPARK_HOST = "darren@10.0.0.254"
SPARK_TTS_DIR = "/home/darren/src/tts"
SPARK_VENV_PYTHON = f"{SPARK_TTS_DIR}/.venv/bin/python"
SPARK_WORKER = f"{SPARK_TTS_DIR}/worker.py"
SPARK_SETUP_MARKER = f"{SPARK_TTS_DIR}/.setup_complete"
SPARK_LOCK_FILE = f"{SPARK_TTS_DIR}/.tts.lock"

# timeouts (seconds)
SSH_CONNECT_TIMEOUT = 10  # how long to wait for SSH connection
SSH_CMD_TIMEOUT = 30  # short commands (mkdir, test, etc.)
SSH_SYNTHESIS_TIMEOUT = 600  # worker synthesis (10 min max)
SCP_TIMEOUT = 120  # file transfers (2 min max)
SSH_CLEANUP_TIMEOUT = 5  # cleanup calls in finally blocks


class SparkError(RuntimeError):
    """Base error for spark operations."""


class SparkUnreachableError(SparkError):
    """Spark machine is not reachable."""


class SparkWorkerError(SparkError):
    """Worker process failed on spark."""


def check_spark_reachable() -> None:
    """Verify spark is reachable via SSH. Raises SparkUnreachableError if not."""
    try:
        result = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}", SPARK_HOST, "echo ok"],
            capture_output=True,
            text=True,
            timeout=SSH_CONNECT_TIMEOUT + 5,
        )
        if result.returncode != 0:
            raise SparkUnreachableError(f"Spark is not reachable at {SPARK_HOST}: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        raise SparkUnreachableError(f"Spark connection timed out after {SSH_CONNECT_TIMEOUT}s — is spark running?")


# map mlx model names to pytorch model names
MLX_TO_PYTORCH_MODELS = {
    "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}


def _map_model_name(mlx_name: str | None) -> str | None:
    """Convert MLX model name to PyTorch equivalent."""
    if mlx_name is None:
        return None
    return MLX_TO_PYTORCH_MODELS.get(mlx_name, mlx_name)


def _ssh(cmd: str, check: bool = True, timeout: int = SSH_CMD_TIMEOUT) -> subprocess.CompletedProcess:
    """Run a command on spark via SSH with timeout."""
    try:
        result = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}", SPARK_HOST, cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise SparkError(f"SSH command timed out after {timeout}s: {cmd}")
    if check and result.returncode != 0:
        stderr = result.stderr.strip()
        if "Connection refused" in stderr or "No route to host" in stderr or "Connection timed out" in stderr:
            raise SparkUnreachableError(f"Spark is not reachable: {stderr}")
        raise SparkError(f"SSH command failed (exit {result.returncode}): {cmd}\n{stderr}")
    return result


def _ssh_stream(cmd: str, timeout: int = SSH_SYNTHESIS_TIMEOUT) -> tuple[int, str]:
    """Run a command on spark via SSH with streaming stdout, capturing stderr.

    Returns (exit_code, stderr).
    """
    try:
        result = subprocess.run(
            ["ssh", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}", SPARK_HOST, cmd],
            text=True,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise SparkError(f"Spark synthesis timed out after {timeout}s — spark may be overloaded or hung")
    return result.returncode, result.stderr


def _scp_to_spark(local_path: str, remote_path: str) -> None:
    """Copy a file to spark with timeout."""
    try:
        result = subprocess.run(
            ["scp", "-q", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}", local_path, f"{SPARK_HOST}:{remote_path}"],
            capture_output=True,
            text=True,
            timeout=SCP_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise SparkError(f"SCP to spark timed out after {SCP_TIMEOUT}s: {local_path}")
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "Connection refused" in stderr or "No route to host" in stderr:
            raise SparkUnreachableError(f"Spark is not reachable: {stderr}")
        raise SparkError(f"SCP to spark failed: {stderr}")


def _run_spark_worker(config_path: str, batch: bool = False) -> None:
    """Run the TTS worker on spark under flock (serializes all TTS work).

    Acquires a remote file lock so only one TTS job runs at a time.
    Other jobs queue up and wait their turn.
    """
    batch_flag = " --batch" if batch else ""

    # check if another job is running (non-blocking) so we can inform the user
    check = _ssh(
        f"flock -n {SPARK_LOCK_FILE} -c true && echo free || echo busy",
        check=False,
    )
    if check.stdout.strip() == "busy":
        print("Spark TTS is busy — queued, waiting for lock...", flush=True)

    # flock FILE COMMAND [ARGS] — acquires lock, runs command, releases on exit
    flock_cmd = f"flock {SPARK_LOCK_FILE} {SPARK_VENV_PYTHON} {SPARK_WORKER} {config_path}{batch_flag}"

    # long timeout — could be queued behind other jobs (1 hour max wait+run)
    rc, stderr = _ssh_stream(flock_cmd, timeout=3600)
    if rc != 0:
        raise SparkWorkerError(
            f"Spark worker failed (exit code {rc})" + (f"\n{stderr.strip()}" if stderr.strip() else "")
        )


def _scp_from_spark(remote_path: str, local_path: str) -> None:
    """Copy a file from spark with timeout."""
    try:
        result = subprocess.run(
            ["scp", "-q", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT}", f"{SPARK_HOST}:{remote_path}", local_path],
            capture_output=True,
            text=True,
            timeout=SCP_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise SparkError(f"SCP from spark timed out after {SCP_TIMEOUT}s: {remote_path}")
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "Connection refused" in stderr or "No route to host" in stderr:
            raise SparkUnreachableError(f"Spark is not reachable: {stderr}")
        raise SparkError(f"SCP from spark failed: {stderr}")


def ensure_spark_setup() -> None:
    """Ensure spark has the TTS environment set up. Idempotent."""
    # verify spark is reachable before anything else
    check_spark_reachable()

    # check if already set up
    result = _ssh(f"test -f {SPARK_SETUP_MARKER} && echo yes || echo no", check=False)
    if result.stdout.strip() == "yes":
        # always sync worker script (for development)
        _sync_worker()
        return

    print("Setting up TTS on spark (first time)...")

    # create project directory
    _ssh(f"mkdir -p {SPARK_TTS_DIR}")

    # create venv
    print("  Creating virtual environment...")
    _ssh(f"python3 -m venv {SPARK_TTS_DIR}/.venv")

    # install PyTorch with CUDA (critical: must use CUDA index)
    print("  Installing PyTorch with CUDA...")
    _ssh(
        f"{SPARK_TTS_DIR}/.venv/bin/pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu130"
    )

    # install qwen-tts and dependencies
    print("  Installing qwen-tts...")
    _ssh(f"{SPARK_TTS_DIR}/.venv/bin/pip install -q qwen-tts soundfile numpy")

    # try to install flash-attn (optional, gives 2-3x speedup)
    print("  Installing flash-attn (optional, may take a while)...")
    result = _ssh(
        f"{SPARK_TTS_DIR}/.venv/bin/pip install -q flash-attn --no-build-isolation",
        check=False,
    )
    if result.returncode != 0:
        print("  flash-attn install failed (non-critical), continuing without it")

    # copy worker script
    _sync_worker()

    # verify CUDA works
    print("  Verifying CUDA...")
    _ssh(f"{SPARK_TTS_DIR}/.venv/bin/python -c 'import torch; assert torch.cuda.is_available()'")

    # mark setup complete
    _ssh(f"touch {SPARK_SETUP_MARKER}")
    print("  Spark TTS setup complete!")


def _sync_worker() -> None:
    """Copy the worker script to spark."""
    local_worker = Path(__file__).parent / "spark_worker.py"
    _scp_to_spark(str(local_worker), SPARK_WORKER)


# ##################################################################
# spark tts engine
# delegates synthesis to spark machine via SSH/SCP
class SparkTtsEngine(TtsEngine):
    def __init__(
        self,
        voice: str = "aiden",
        ref_audio: str | None = None,
        ref_text: str | None = None,
        voice_description: str | None = None,
        model_name: str | None = None,
    ):
        self.voice = voice
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.voice_description = voice_description
        self.model_name = model_name

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

        # ensure spark is set up
        ensure_spark_setup()

        # split text into chunks
        chunks = split_into_chunks(text)
        print(f"Synthesizing {len(chunks)} chunk(s) on spark...")

        # create remote work directory
        work_id = uuid.uuid4().hex[:12]
        remote_work_dir = f"/tmp/tts_spark_{work_id}"
        _ssh(f"mkdir -p {remote_work_dir}")

        try:
            # determine mode and build config
            config = self._build_config(chunks, remote_work_dir, language, temperature)

            # if voice clone, copy ref_audio to spark
            if config["mode"] == "clone" and self.ref_audio:
                remote_ref = f"{remote_work_dir}/ref_audio.wav"
                _scp_to_spark(self.ref_audio, remote_ref)
                config["ref_audio"] = remote_ref

            # write config to local temp file and SCP to spark
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(config, f)
                local_config = f.name

            remote_config = f"{remote_work_dir}/config.json"
            _scp_to_spark(local_config, remote_config)
            Path(local_config).unlink()

            # run worker on spark (serialized via flock)
            print("Running synthesis on spark...")
            _run_spark_worker(remote_config)

            # copy output back
            remote_output = f"{remote_work_dir}/output.wav"
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                local_wav = tmp.name

            _scp_from_spark(remote_output, local_wav)

            # enhance locally if requested
            if enhance:
                print("Enhancing audio locally...")
                enhanced_wav = local_wav + ".enhanced.wav"
                enhance_output(Path(local_wav), Path(enhanced_wav))
                Path(local_wav).unlink()
                local_wav = enhanced_wav

            # convert to mp3 if needed
            if output_path.suffix.lower() == ".mp3":
                convert_wav_to_mp3(Path(local_wav), output_path, normalize=True, speed=speed)
                Path(local_wav).unlink()
            elif speed != 1.0:
                # apply speed adjustment for non-mp3 output
                speed = max(0.5, min(2.0, speed))
                cmd = ["ffmpeg", "-y", "-i", local_wav, "-af", f"atempo={speed}", str(output_path)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                Path(local_wav).unlink()
                if result.returncode != 0:
                    raise RuntimeError(f"Speed adjustment failed: {result.stderr}")
            else:
                shutil.move(local_wav, str(output_path))

        finally:
            # clean up remote work directory
            _ssh(f"rm -rf {remote_work_dir}", check=False, timeout=SSH_CLEANUP_TIMEOUT)

        return output_path

    def _build_config(
        self,
        chunks: list[str],
        remote_work_dir: str,
        language: str,
        temperature: float,
    ) -> dict:
        """Build the JSON config for the spark worker."""
        config = {
            "chunks": chunks,
            "output": f"{remote_work_dir}/output.wav",
            "language": language,
            "temperature": temperature,
        }

        if self.voice_description:
            # voice design mode
            config["mode"] = "design"
            config["voice_description"] = self.voice_description
            config["model_name"] = _map_model_name(self.model_name) or "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        elif self.ref_audio:
            # voice clone mode
            config["mode"] = "clone"
            config["ref_text"] = self.ref_text
            config["model_name"] = _map_model_name(self.model_name) or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        else:
            # custom voice (preset) mode
            config["mode"] = "custom"
            config["voice"] = self.voice.title()  # capitalize for PyTorch API
            config["model_name"] = _map_model_name(self.model_name) or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

        return config


# ##################################################################
# get spark engine
# factory function mirroring get_engine but for spark backend
def get_spark_engine(voice: str = "aiden", voice_description: str | None = None) -> SparkTtsEngine:
    """Get a SparkTtsEngine for the given voice."""
    # check if voice is a zip file (portable voice package)
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
        return SparkTtsEngine(ref_audio=str(voice_wav), ref_text=ref_text)

    # check if voice is a registered custom voice
    custom_voice = get_voice_description(voice)
    if custom_voice:
        voice_type = custom_voice.get("type", "description")
        if voice_type == "clone":
            return SparkTtsEngine(
                ref_audio=custom_voice["ref_audio"],
                ref_text=custom_voice.get("ref_text", ""),
                model_name=custom_voice.get("model_name"),
            )
        else:
            return SparkTtsEngine(voice_description=custom_voice["description"])

    # explicit voice description provided
    if voice_description:
        return SparkTtsEngine(voice_description=voice_description)

    # standard preset voice
    if voice.lower() not in [v.lower() for v in QWEN_VOICES]:
        raise ValueError(f"Unknown voice: {voice}. Supported: {', '.join(QWEN_VOICES)}")
    return SparkTtsEngine(voice=voice)


# ##################################################################
# multi-speaker synthesis on spark
# batches by speaker (one model load per speaker), reassembles in dialogue order
def synthesize_multi_speaker_spark(
    jsonl_path: Path,
    output_path: Path,
    language: str = "English",
    temperature: float = DEFAULT_TEMPERATURE,
    speed: float = DEFAULT_SPEED,
    enhance: bool = False,
) -> Path:
    """Synthesize multi-speaker dialogue on spark, mirroring local multi-speaker."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # parse and validate
    segments = parse_multi_speaker_jsonl(jsonl_path)
    if not segments:
        raise ValueError(f"No dialogue found in {jsonl_path}")
    validate_multi_speaker_voices(segments)

    # group by speaker: {voice: [(index, text), ...]}
    speaker_batches: dict[str, list[tuple[int, str]]] = {}
    for idx, (voice, text) in enumerate(segments):
        voice_lower = voice.lower()
        if voice_lower not in speaker_batches:
            speaker_batches[voice_lower] = []
        speaker_batches[voice_lower].append((idx, text))

    total_lines = len(segments)
    print(f"\nMulti-speaker synthesis on spark: {total_lines} lines, {len(speaker_batches)} speakers")
    for voice, lines in speaker_batches.items():
        word_count = sum(len(text.split()) for _, text in lines)
        print(f"  {voice}: {len(lines)} lines, {word_count} words")

    ensure_spark_setup()

    # create local and remote work dirs
    work_dir = Path(tempfile.mkdtemp(prefix="spark_multi_"))
    work_id = uuid.uuid4().hex[:12]
    remote_work_dir = f"/tmp/tts_spark_multi_{work_id}"
    _ssh(f"mkdir -p {remote_work_dir}")
    total_start = time.time()

    try:
        # process each speaker batch (one model load per speaker via batch mode)
        for speaker_num, (voice, lines) in enumerate(speaker_batches.items(), 1):
            print(f"\n[Speaker {speaker_num}/{len(speaker_batches)}] {voice}: {len(lines)} lines")

            # get voice config
            custom_voice = get_voice_description(voice)
            if custom_voice and custom_voice.get("type") == "clone":
                mode = "clone"
                ref_audio = custom_voice["ref_audio"]
                ref_text = custom_voice.get("ref_text", "")
                model_name = _map_model_name(custom_voice.get("model_name")) or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            elif custom_voice and custom_voice.get("description"):
                mode = "design"
                ref_audio = None
                ref_text = None
                model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            else:
                mode = "custom"
                ref_audio = None
                ref_text = None
                model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

            # build batch items (one per line, each with its own chunks)
            batch_items = []
            for idx, text in lines:
                chunks = split_into_chunks(text)
                remote_output = f"{remote_work_dir}/line_{idx:04d}.wav"
                batch_items.append({"chunks": chunks, "output": remote_output})

            # build config
            config = {
                "mode": mode,
                "batch": batch_items,
                "language": language,
                "temperature": temperature,
                "model_name": model_name,
            }
            if mode == "custom":
                config["voice"] = voice.title()
            elif mode == "clone" and ref_audio:
                # SCP ref_audio to spark
                remote_ref = f"{remote_work_dir}/ref_{voice}.wav"
                _scp_to_spark(ref_audio, remote_ref)
                config["ref_audio"] = remote_ref
                config["ref_text"] = ref_text or ""
            elif mode == "design" and custom_voice:
                config["voice_description"] = custom_voice.get("description", "A clear neutral voice.")

            # write and SCP config
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(config, f)
                local_config = f.name
            remote_config = f"{remote_work_dir}/config_{voice}.json"
            _scp_to_spark(local_config, remote_config)
            Path(local_config).unlink()

            # run batch worker (serialized via flock)
            speaker_start = time.time()
            _run_spark_worker(remote_config, batch=True)

            elapsed = time.time() - speaker_start
            print(f"[Speaker {speaker_num}/{len(speaker_batches)}] {voice}: {len(lines)} lines in {elapsed:.1f}s")

            # SCP all line wavs back
            for idx, _ in lines:
                remote_wav = f"{remote_work_dir}/line_{idx:04d}.wav"
                local_wav = work_dir / f"line_{idx:04d}.wav"
                _scp_from_spark(remote_wav, str(local_wav))

        # verify all lines
        missing = [i for i in range(total_lines) if not (work_dir / f"line_{i:04d}.wav").exists()]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} lines: {missing[:10]}...")

        # concatenate in dialogue order with silence trimming
        print(f"\nTrimming silence and concatenating {total_lines} lines...")
        line_wavs = [work_dir / f"line_{i:04d}.wav" for i in range(total_lines)]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            combined_wav = Path(tmp.name)
        concatenate_wav_files(line_wavs, combined_wav, trim_trailing_silence=True)

        # enhance if requested
        if enhance:
            print("Enhancing combined audio...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                enhanced_wav = Path(tmp.name)
            enhance_output(combined_wav, enhanced_wav)
            combined_wav.unlink()
            combined_wav = enhanced_wav

        # convert to final format
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
        _ssh(f"rm -rf {remote_work_dir}", check=False)

    return output_path
