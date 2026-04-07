"""ASR Model Benchmark — compare accuracy, speed, and memory across models."""

import json
import sys
import time
from pathlib import Path

import librosa as lr
import numpy as np
from jiwer import wer

# reuse VAD + command processing from voice_cmd
sys.path.insert(0, str(Path(__file__).parent.parent))
from voice_cmd import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    TextBuffer,
    VADConfig,
    VADState,
    process_utterance,
)

FIXTURES = Path(__file__).parent / "fixtures"
GRAY = "\033[90m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
RESET = "\033[0m"


# ---------------------------------------------------------------------------
# VAD segmentation (shared across all models)
# ---------------------------------------------------------------------------

def vad_segment(audio_16k: np.ndarray) -> list[np.ndarray]:
    """Split audio into utterances using VAD."""
    vad = VADState(VADConfig())
    utterances: list[np.ndarray] = []
    current: list[np.ndarray] = []

    for i in range(0, len(audio_16k) - FRAME_SAMPLES, FRAME_SAMPLES):
        frame = audio_16k[i : i + FRAME_SAMPLES]
        event, frames = vad.process_frame(frame)

        if event == "speech_start":
            current = list(frames)
        elif event == "speech_continue":
            current.extend(frames)
        elif event == "speech_end" and current:
            utterances.append(np.concatenate(current))
            current = []

    if current:
        utterances.append(np.concatenate(current))

    return utterances


# ---------------------------------------------------------------------------
# Model wrappers — each returns a transcribe(audio_np) → str callable
# ---------------------------------------------------------------------------

def load_parakeet_beam():
    """Parakeet TDT 0.6B batch mode + beam search."""
    import mlx.core as mx
    from parakeet_mlx import Beam, DecodingConfig, from_pretrained
    from parakeet_mlx.audio import get_logmel

    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    config = DecodingConfig(decoding=Beam(beam_size=5))

    def transcribe(audio: np.ndarray) -> str:
        mel = get_logmel(mx.array(audio.astype(np.float32)), model.preprocessor_config)
        results = model.generate(mel, decoding_config=config)
        return results[0].text.strip()

    return transcribe


def load_faster_whisper(size: str):
    """Faster-whisper via CTranslate2."""
    from faster_whisper import WhisperModel

    model = WhisperModel(size, compute_type="float32")

    def transcribe(audio: np.ndarray) -> str:
        segments, _ = model.transcribe(audio, language="en", beam_size=5)
        return " ".join(s.text for s in segments).strip()

    return transcribe


def load_transformers_asr(model_id: str):
    """HuggingFace transformers pipeline (moonshine, medasr, wav2vec2)."""
    import torch
    from transformers import pipeline

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
    )

    def transcribe(audio: np.ndarray) -> str:
        result = pipe({"raw": audio, "sampling_rate": SAMPLE_RATE})
        return result["text"].strip()

    return transcribe


def load_mlx_whisper(model_id: str):
    """MLX Whisper."""
    from mlx_whisper import transcribe as mlx_transcribe

    # warm up / download model on first call
    _warmed = [False]

    def transcribe(audio: np.ndarray) -> str:
        result = mlx_transcribe(
            audio,
            path_or_hf_repo=model_id,
            language="en",
        )
        return result["text"].strip()

    return transcribe


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = [
    # --- Parakeet (MLX) ---
    {
        "name": "parakeet-tdt-0.6b beam=5",
        "size_mb": 2392,
        "loader": load_parakeet_beam,
        "args": (),
    },
    # --- Faster-Whisper (CTranslate2, CPU) ---
    {
        "name": "fw-tiny.en",
        "size_mb": 72,
        "loader": load_faster_whisper,
        "args": ("tiny.en",),
    },
    {
        "name": "fw-base.en",
        "size_mb": 138,
        "loader": load_faster_whisper,
        "args": ("base.en",),
    },
    {
        "name": "fw-small.en",
        "size_mb": 461,
        "loader": load_faster_whisper,
        "args": ("small.en",),
    },
    {
        "name": "fw-small (multi)",
        "size_mb": 461,
        "loader": load_faster_whisper,
        "args": ("small",),
    },
    {
        "name": "fw-distil-small.en",
        "size_mb": 317,
        "loader": load_faster_whisper,
        "args": ("Systran/faster-distil-whisper-small.en",),
    },
    {
        "name": "fw-distil-medium.en",
        "size_mb": 752,
        "loader": load_faster_whisper,
        "args": ("Systran/faster-distil-whisper-medium.en",),
    },
    # --- MLX Whisper (Apple Silicon native) ---
    {
        "name": "mlx-tiny-4bit",
        "size_mb": 21,
        "loader": load_mlx_whisper,
        "args": ("mlx-community/whisper-tiny-4bit",),
    },
    {
        "name": "mlx-small-4bit",
        "size_mb": 187,
        "loader": load_mlx_whisper,
        "args": ("mlx-community/whisper-small-mlx-4bit",),
    },
    {
        "name": "mlx-small-8bit",
        "size_mb": 282,
        "loader": load_mlx_whisper,
        "args": ("mlx-community/whisper-small-mlx-8bit",),
    },
    {
        "name": "mlx-medium-4bit",
        "size_mb": 489,
        "loader": load_mlx_whisper,
        "args": ("mlx-community/whisper-medium-mlx-4bit",),
    },
    {
        "name": "mlx-large-v3-turbo-4bit",
        "size_mb": 442,
        "loader": load_mlx_whisper,
        "args": ("mlx-community/whisper-large-v3-turbo-4bit",),
    },
    # --- Transformers (torch, MPS) ---
    {
        "name": "moonshine-tiny",
        "size_mb": 103,
        "loader": load_transformers_asr,
        "args": ("UsefulSensors/moonshine-tiny",),
    },
    {
        "name": "moonshine-base",
        "size_mb": 235,
        "loader": load_transformers_asr,
        "args": ("UsefulSensors/moonshine-base",),
    },
    {
        "name": "wav2vec2-base-960h",
        "size_mb": 360,
        "loader": load_transformers_asr,
        "args": ("facebook/wav2vec2-base-960h",),
    },
]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(model_names: list[str] | None = None) -> None:
    with open(FIXTURES / "test_cases.json") as f:
        test_cases = json.load(f)

    # pre-load all audio + VAD segments
    print(f"{GRAY}Loading audio files...{RESET}")
    audio_data: list[dict] = []
    for tc in test_cases:
        path = FIXTURES / tc["file"]
        audio, _ = lr.load(str(path), sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)
        utterances = vad_segment(audio)
        audio_data.append({
            "file": tc["file"],
            "spoken": tc["spoken"],
            "expected": tc["expected"],
            "utterances": utterances,
            "duration": len(audio) / SAMPLE_RATE,
        })
    print(f"{GREEN}Loaded {len(audio_data)} files, VAD segmented{RESET}\n")

    # filter models
    models_to_run = MODELS
    if model_names:
        models_to_run = [m for m in MODELS if any(n.lower() in m["name"].lower() for n in model_names)]

    results: list[dict] = []

    for model_cfg in models_to_run:
        name = model_cfg["name"]
        print(f"{BOLD}{'=' * 70}{RESET}")
        print(f"{BOLD}Model: {name} ({model_cfg['size_mb']}MB){RESET}")
        print(f"{'=' * 70}")

        # load model
        try:
            t0 = time.monotonic()
            transcribe_fn = model_cfg["loader"](*model_cfg["args"])
            load_time = time.monotonic() - t0
            print(f"  {GREEN}Loaded in {load_time:.1f}s{RESET}")
        except Exception as e:
            print(f"  {RED}FAILED to load: {e}{RESET}\n")
            results.append({"name": name, "error": str(e)})
            continue

        file_results = []
        total_time = 0.0

        for ad in audio_data:
            buf = TextBuffer()
            t0 = time.monotonic()

            try:
                for utt in ad["utterances"]:
                    text = transcribe_fn(utt)
                    if text:
                        buf, _, _ = process_utterance(text, buf)
            except Exception as e:
                print(f"  {RED}ERROR on {ad['file']}: {e}{RESET}")
                file_results.append({
                    "file": ad["file"],
                    "got": f"ERROR: {e}",
                    "expected": ad["expected"],
                    "wer": 1.0,
                    "time": 0,
                })
                continue

            elapsed = time.monotonic() - t0
            total_time += elapsed

            got = buf.text.strip()
            expected = ad["expected"].strip()

            # WER needs non-empty reference
            if expected:
                w = wer(expected, got) if got else 1.0
            else:
                w = 0.0 if not got else 1.0

            file_results.append({
                "file": ad["file"],
                "got": got,
                "expected": expected,
                "wer": w,
                "time": elapsed,
            })

            status = f"{GREEN}WER {w:.0%}{RESET}" if w < 0.3 else f"{YELLOW}WER {w:.0%}{RESET}" if w < 0.6 else f"{RED}WER {w:.0%}{RESET}"
            print(f"  {ad['file']:<40} {elapsed:>5.1f}s  {status}")

        avg_wer = np.mean([r["wer"] for r in file_results]) if file_results else 1.0
        avg_time = total_time / len(file_results) if file_results else 0

        results.append({
            "name": name,
            "size_mb": model_cfg["size_mb"],
            "load_time": load_time,
            "avg_time": avg_time,
            "avg_wer": avg_wer,
            "files": file_results,
        })

        print(f"  {BOLD}AVG: WER={avg_wer:.0%}  time={avg_time:.1f}s/file  load={load_time:.1f}s{RESET}\n")

    # summary table
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD}SUMMARY{RESET}")
    print(f"{'=' * 80}")
    print(f"{'Model':<32} {'Load':>6} {'Avg/f':>6} {'WER':>6} {'Size':>7}")
    print(f"{'-' * 32} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 7}")

    for r in sorted(results, key=lambda x: x.get("avg_wer", 1.0)):
        if "error" in r:
            print(f"{r['name']:<32} {'FAIL':>6} {'':>6} {'':>6} {'':>7}")
            continue
        print(f"{r['name']:<32} {r['load_time']:>5.1f}s {r['avg_time']:>5.1f}s {r['avg_wer']:>5.0%} {r['size_mb']:>5}MB")

    # per-file detail
    print(f"\n{BOLD}PER-FILE DETAIL{RESET}")
    for ad in audio_data:
        fname = ad["file"]
        expected = ad["expected"].strip()
        print(f"\n{YELLOW}--- {fname} ---{RESET}")
        print(f"  {GRAY}Expected:{RESET} {expected[:100]}{'...' if len(expected) > 100 else ''}")
        for r in results:
            if "error" in r:
                continue
            for fr in r.get("files", []):
                if fr["file"] == fname:
                    got = fr["got"]
                    color = GREEN if fr["wer"] < 0.3 else YELLOW if fr["wer"] < 0.6 else RED
                    print(f"  {color}{r['name']:<30}{RESET} {got[:100]}{'...' if len(got) > 100 else ''}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR Model Benchmark")
    parser.add_argument("models", nargs="*", help="Filter models by name substring")
    args = parser.parse_args()

    run_benchmark(args.models if args.models else None)
