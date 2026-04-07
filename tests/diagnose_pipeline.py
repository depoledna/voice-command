"""Pipeline diagnostic — find where accuracy drops off.

Stages:
  1. Raw transcription (full file, no VAD) vs spoken text
  2. VAD segmentation stats (how many segments, lengths)
  3. Per-segment raw transcription (no command processing)
  4. Full pipeline (VAD + transcribe + process_utterance) vs expected
"""

import json
import sys
import time
from pathlib import Path

import librosa as lr
import numpy as np
from jiwer import wer

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


def vad_segment(audio_16k: np.ndarray) -> list[np.ndarray]:
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


def load_model(name: str):
    """Load one model, return transcribe(audio) -> str."""
    if name == "whisper-small":
        from faster_whisper import WhisperModel
        model = WhisperModel("small.en", compute_type="float32")
        def transcribe(audio: np.ndarray) -> str:
            segments, _ = model.transcribe(audio, language="en", beam_size=5)
            return " ".join(s.text for s in segments).strip()
        return transcribe

    if name == "moonshine-base":
        from transformers import pipeline
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        pipe = pipeline("automatic-speech-recognition", model="UsefulSensors/moonshine-base", device=device)
        def transcribe(audio: np.ndarray) -> str:
            result = pipe({"raw": audio, "sampling_rate": SAMPLE_RATE})
            return result["text"].strip()
        return transcribe

    raise ValueError(f"Unknown model: {name}")


def run_diagnostic(model_name: str = "whisper-small"):
    with open(FIXTURES / "test_cases.json") as f:
        test_cases = json.load(f)

    print(f"{BOLD}Loading model: {model_name}{RESET}")
    transcribe = load_model(model_name)
    print(f"{GREEN}Model ready{RESET}\n")

    for tc in test_cases:
        path = FIXTURES / tc["file"]
        audio, _ = lr.load(str(path), sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)

        print(f"\n{BOLD}{'=' * 70}{RESET}")
        print(f"{BOLD}{tc['file']}{RESET}  ({len(audio)/SAMPLE_RATE:.1f}s)")
        print(f"  {GRAY}Spoken:   {tc['spoken'][:100]}{RESET}")
        print(f"  {GRAY}Expected: {tc['expected'][:100]}{RESET}")

        # --- Stage 1: Full-file transcription (NO VAD) ---
        t0 = time.monotonic()
        raw_full = transcribe(audio)
        t1 = time.monotonic()
        raw_wer = wer(tc["spoken"].lower(), raw_full.lower()) if raw_full else 1.0
        color = GREEN if raw_wer < 0.15 else YELLOW if raw_wer < 0.3 else RED
        print(f"\n  {BOLD}Stage 1: Full-file transcription (no VAD){RESET}  [{t1-t0:.1f}s]")
        print(f"  {color}WER vs spoken: {raw_wer:.0%}{RESET}")
        print(f"  Got: {raw_full[:120]}")

        # --- Stage 2: VAD segmentation ---
        segments = vad_segment(audio)
        seg_durations = [len(s) / SAMPLE_RATE for s in segments]
        print(f"\n  {BOLD}Stage 2: VAD segmentation{RESET}")
        print(f"  Segments: {len(segments)}  durations: {['%.1fs' % d for d in seg_durations]}")

        # --- Stage 3: Per-segment transcription (no command processing) ---
        print(f"\n  {BOLD}Stage 3: Per-segment transcription (no commands){RESET}")
        seg_texts = []
        for i, seg in enumerate(segments):
            t0 = time.monotonic()
            seg_text = transcribe(seg)
            t1 = time.monotonic()
            seg_texts.append(seg_text)
            print(f"    [{i}] ({seg_durations[i]:.1f}s, {t1-t0:.1f}s) → \"{seg_text}\"")

        combined_raw = " ".join(seg_texts)
        combined_wer = wer(tc["spoken"].lower(), combined_raw.lower()) if combined_raw else 1.0
        color = GREEN if combined_wer < 0.15 else YELLOW if combined_wer < 0.3 else RED
        print(f"  {color}Combined WER vs spoken: {combined_wer:.0%}{RESET}")
        print(f"  Combined: {combined_raw[:120]}")

        # --- Stage 4: Full pipeline (VAD + transcribe + commands) ---
        print(f"\n  {BOLD}Stage 4: Full pipeline (VAD + transcribe + commands){RESET}")
        buf = TextBuffer()
        for i, seg in enumerate(segments):
            seg_text = transcribe(seg)
            if seg_text:
                buf, msg, _cmd = process_utterance(seg_text, buf)
                print(f"    [{i}] \"{seg_text}\" → {msg}")

        final = buf.text.strip()
        expected = tc["expected"].strip()
        pipeline_wer = wer(expected, final) if (expected and final) else (0.0 if not expected and not final else 1.0)
        color = GREEN if pipeline_wer < 0.15 else YELLOW if pipeline_wer < 0.3 else RED
        print(f"  {color}Pipeline WER vs expected: {pipeline_wer:.0%}{RESET}")
        print(f"  Final:    {final[:120]}")
        print(f"  Expected: {expected[:120]}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "whisper-small"
    run_diagnostic(model)
