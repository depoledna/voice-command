"""Microphone capture, VAD, and ASR — wrapped behind a single AudioSource.

Public API:
    AudioSource         — owns the input stream, VAD state, and event queue
    SpeechStart         — sentinel emitted when VAD detects speech onset
    SpeechEnd(audio)    — emitted on speech offset; carries the full utterance

The caller never sees raw audio frames, the VAD, the resampler, or the
sounddevice queue. Mic switching is one method call: AudioSource.set_device.
"""

from __future__ import annotations

import queue
import sys
from collections import deque
from dataclasses import dataclass
from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from silero_vad_lite import SileroVAD


SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # 32ms @ 16kHz

_MODEL_ID = "mlx-community/whisper-small-mlx-8bit"
_INITIAL_PROMPT = (
    "Dennis is dictating text using voice commands. "
    "He says 'period', 'new line', 'comma', 'colon', "
    "'exclamation mark' as literal commands. "
    "He often discusses software development: "
    "TypeScript, JavaScript, Python, React, Next.js, FastAPI, Django, Flask, "
    "PostgreSQL, MongoDB, SQLAlchemy, GraphQL, Redis, Kubernetes, Docker, Nginx. "
    "Tools like GitHub, npm, pip, Homebrew, VS Code, Xcode, Webpack, Vite, ESLint, Prettier. "
    "Concepts: API, REST, RESTful, WebSocket, OAuth, JWT, CRUD, CI/CD, DNS, SSL, TLS, "
    "SSH, HTTP, HTTPS, URL, JSON, YAML, CSV, regex, async, await, middleware, "
    "microservice, endpoint, webhook, deploy, refactor, linting. "
    "Casing matters: camelCase, PascalCase, kebab-case, snake_case, macOS, iOS, npm, GitHub."
)


# ---------------------------------------------------------------------------
# Public event types
# ---------------------------------------------------------------------------

class SpeechStart:
    """Emitted when VAD detects speech onset. No payload."""
    __slots__ = ()

    def __repr__(self) -> str:
        return "SpeechStart()"


@dataclass
class SpeechEnd:
    """Emitted on speech offset. `audio` is the full utterance @ 16 kHz mono."""
    audio: np.ndarray


Event = SpeechStart | SpeechEnd


# ---------------------------------------------------------------------------
# Internal: VAD state machine
# ---------------------------------------------------------------------------

@dataclass
class _VADConfig:
    threshold: float = 0.45
    neg_threshold: float = 0.25
    min_speech_ms: int = 64
    min_silence_ms: int = 600
    speech_pad_ms: int = 300
    frame_ms: int = 32


class _VADState:
    """Silero VAD with hysteresis. Internal to AudioSource."""

    def __init__(self, config: _VADConfig):
        self.cfg = config
        self._vad = SileroVAD(SAMPLE_RATE)
        self._speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
        pad_frames = max(1, self.cfg.speech_pad_ms // self.cfg.frame_ms)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pad_frames)

    def process_frame(self, frame_16k: np.ndarray) -> tuple[str, list[np.ndarray]]:
        """Returns (event, audio_frames). Events: speech_start, speech_continue,
        speech_end, silence."""
        prob = self._vad.process(frame_16k.astype(np.float32).tobytes())

        if not self._speaking:
            self._pre_roll.append(frame_16k.copy())
            if prob >= self.cfg.threshold:
                self._speech_frames += 1
                min_frames = max(1, self.cfg.min_speech_ms // self.cfg.frame_ms)
                if self._speech_frames >= min_frames:
                    self._speaking = True
                    self._silence_frames = 0
                    pre_roll = list(self._pre_roll)
                    self._pre_roll.clear()
                    return "speech_start", pre_roll + [frame_16k]
            else:
                self._speech_frames = 0
            return "silence", []

        if prob < self.cfg.neg_threshold:
            self._silence_frames += 1
            end_frames = max(1, self.cfg.min_silence_ms // self.cfg.frame_ms)
            if self._silence_frames >= end_frames:
                self._speaking = False
                self._speech_frames = 0
                self._silence_frames = 0
                return "speech_end", []
            return "speech_continue", [frame_16k]

        self._silence_frames = 0
        return "speech_continue", [frame_16k]

    @property
    def is_speaking(self) -> bool:
        return self._speaking


# ---------------------------------------------------------------------------
# Internal: device + resampler helpers
# ---------------------------------------------------------------------------

def _find_input_device(preferred: int | None) -> tuple[int, str, int]:
    if preferred is not None:
        info = sd.query_devices(preferred)
        return preferred, info["name"], int(info["default_samplerate"])

    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and "airpod" in d["name"].lower():
            return i, d["name"], int(d["default_samplerate"])

    idx = sd.default.device[0]
    info = sd.query_devices(idx)
    return idx, info["name"], int(info["default_samplerate"])


def _make_resampler(orig_sr: int) -> tuple[int, int, int]:
    """Returns (up, down, blocksize) so callback delivers ~512 samples @ 16kHz."""
    if orig_sr == SAMPLE_RATE:
        return 1, 1, FRAME_SAMPLES
    g = gcd(orig_sr, SAMPLE_RATE)
    up = SAMPLE_RATE // g
    down = orig_sr // g
    blocksize = FRAME_SAMPLES * down // up
    return up, down, blocksize


def _resample_frame(audio: np.ndarray, up: int, down: int) -> np.ndarray:
    if up == 1 and down == 1:
        return audio.flatten().astype(np.float32)
    return resample_poly(audio.flatten().astype(np.float32), up, down).astype(np.float32)


# ---------------------------------------------------------------------------
# AudioSource — the only class the rest of the app sees
# ---------------------------------------------------------------------------

class AudioSource:
    """Microphone + VAD + ASR, behind a poll-and-event interface.

    Lifecycle:
        src = AudioSource(device, vad_threshold, min_silence_ms)
        src.start()
        try:
            for evt in src.poll(timeout=0.05): ...
        finally:
            src.close()

    Hot-swap a mic with `src.set_device(idx)` — close + drain + reopen + VAD
    reset all happen inside the call. On failure the previous device stays.
    """

    def __init__(
        self,
        device: int | None,
        vad_threshold: float,
        min_silence_ms: int,
    ) -> None:
        self._device_pref = device
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream: sd.InputStream | None = None
        self._dev_idx: int = -1
        self._dev_name: str = ""
        self._dev_rate: int = SAMPLE_RATE
        self._up: int = 1
        self._down: int = 1
        self._vad = _VADState(_VADConfig(
            threshold=vad_threshold,
            min_silence_ms=min_silence_ms,
        ))
        self._utterance: list[np.ndarray] = []

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Open the input stream and start capturing."""
        self._open(self._device_pref)
        assert self._stream is not None
        self._stream.start()

    def close(self) -> None:
        """Stop and release the input stream. Idempotent and exception-safe."""
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    # -- runtime control ----------------------------------------------------

    def set_device(self, idx: int) -> str:
        """Switch microphone in place. Returns the new device name on success."""
        if idx == self._dev_idx:
            return self._dev_name
        self.close()
        self._drain_q()
        self._open(idx)
        assert self._stream is not None
        self._stream.start()
        # Preserve current VAD tuning across the swap, but reset detector state.
        self._vad = _VADState(_VADConfig(
            threshold=self._vad.cfg.threshold,
            min_silence_ms=self._vad.cfg.min_silence_ms,
        ))
        self._utterance = []
        return self._dev_name

    def set_vad(
        self,
        *,
        threshold: float | None = None,
        min_silence_ms: int | None = None,
    ) -> None:
        """Adjust VAD parameters live. Takes effect on the next frame."""
        if threshold is not None:
            self._vad.cfg.threshold = threshold
        if min_silence_ms is not None:
            self._vad.cfg.min_silence_ms = min_silence_ms

    # -- event polling ------------------------------------------------------

    def poll(self, timeout: float = 0.05) -> list[Event]:
        """Drain available frames from the mic, run VAD, return events.

        Blocks up to `timeout` seconds for the first frame, then drains the
        queue without blocking. Returns 0+ events in arrival order.
        """
        events: list[Event] = []
        try:
            raw = self._audio_q.get(timeout=timeout)
        except queue.Empty:
            return events
        self._handle_frame(raw, events)
        # drain anything else that piled up while we were polling
        while True:
            try:
                raw = self._audio_q.get_nowait()
            except queue.Empty:
                break
            self._handle_frame(raw, events)
        return events

    # -- properties ---------------------------------------------------------

    @property
    def device_name(self) -> str:
        return self._dev_name

    @property
    def device_index(self) -> int:
        return self._dev_idx

    @property
    def in_speech(self) -> bool:
        return self._vad.is_speaking

    # -- statics ------------------------------------------------------------

    @staticmethod
    def list_devices() -> list[tuple[int, str]]:
        """Return [(index, name)] for every device with input channels."""
        return [
            (i, d["name"])
            for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0
        ]

    @staticmethod
    def transcribe(audio: np.ndarray) -> str:
        """Run ASR on a 16 kHz mono float32 array. Empty array → empty string."""
        if audio.size == 0:
            return ""
        try:
            from mlx_whisper import transcribe as mlx_transcribe

            result = mlx_transcribe(
                audio,
                path_or_hf_repo=_MODEL_ID,
                language="en",
                initial_prompt=_INITIAL_PROMPT,
            )
            return result.get("text", "").strip()
        except Exception as e:
            # stderr — stdout is owned by the alt-screen TUI while running
            print(f"transcription error: {e}", file=sys.stderr)
            return ""

    @staticmethod
    def warmup() -> None:
        """Force the ASR model to load now (avoids latency on first utterance)."""
        AudioSource.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

    # -- internals ----------------------------------------------------------

    def _open(self, preferred: int | None) -> None:
        dev_idx, dev_name, dev_rate = _find_input_device(preferred)
        up, down, blocksize = _make_resampler(dev_rate)
        q = self._audio_q

        def callback(indata, frames, time_info, status):
            try:
                q.put_nowait(indata.copy())
            except queue.Full:
                pass

        self._stream = sd.InputStream(
            device=dev_idx,
            samplerate=dev_rate,
            channels=1,
            blocksize=blocksize,
            callback=callback,
            dtype="float32",
        )
        self._dev_idx = dev_idx
        self._dev_name = dev_name
        self._dev_rate = dev_rate
        self._up = up
        self._down = down
        # remember resolved device so subsequent set_device(same_idx) is a no-op
        self._device_pref = dev_idx

    def _drain_q(self) -> None:
        while True:
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                return

    def _handle_frame(self, raw: np.ndarray, events: list[Event]) -> None:
        frame_16k = _resample_frame(raw, self._up, self._down)
        event, frames = self._vad.process_frame(frame_16k)
        if event == "speech_start":
            self._utterance = list(frames)
            events.append(SpeechStart())
        elif event == "speech_continue":
            self._utterance.extend(frames)
        elif event == "speech_end":
            audio = np.concatenate(self._utterance) if self._utterance else np.zeros(0, dtype=np.float32)
            self._utterance = []
            events.append(SpeechEnd(audio=audio))
        # "silence" → no event
