# voice-command

VAD-driven streaming voice dictation for macOS (Apple Silicon). Speaks into your mic, text appears in a terminal buffer or gets typed directly into any app.

All inference runs locally — Whisper for ASR, Silero for VAD, Qwen for tech-term correction. No cloud APIs.

## Requirements

- macOS with Apple Silicon
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Microphone access (System Settings > Privacy > Microphone)
- Accessibility access for `--type` mode (System Settings > Privacy > Accessibility)

## Install

```bash
git clone https://github.com/BruhTheMomentum/voice-command.git
cd voice-command
uv sync
```

Models download automatically on first run (~300MB for Whisper + ~1GB for Qwen).

## Usage

```bash
# Terminal buffer mode (TUI)
uv run python voice_cmd.py

# Type directly into the focused app
uv run python voice_cmd.py --type

# Transcribe a recording
uv run python voice_cmd.py --file recording.m4a

# List audio devices
uv run python voice_cmd.py --list-devices

# Use a specific device
uv run python voice_cmd.py --device 1
```

## Voice Commands

| Command | Action |
|---------|--------|
| `period` / `comma` / `question mark` | Insert punctuation |
| `new line` | Line break |
| `new paragraph` | Double line break |
| `scratch that` | Delete last ~5 words |
| `delete last N words` | Delete last N words |
| `undo` | Undo last action |
| `clear all` | Clear buffer |
| `stop listening` | Pause |
| `start listening` | Resume |
| `copy all` | Copy to clipboard |
| `done` | Copy to clipboard and exit |
| `show commands` | Show command list |

Commands can appear inline with dictated text: "Send the email **period** **new line** Don't forget the attachment" produces two lines with proper punctuation.

## Pipeline

1. **Audio** - `sounddevice` captures mic input, resampled to 16kHz
2. **VAD** - Silero VAD with hysteresis detects speech boundaries (32ms frames, pre-roll buffering)
3. **ASR** - MLX Whisper (small, 8-bit) with dev-vocabulary prompt
4. **LLM** - Qwen3 1.7B (4-bit) fixes tech terms: "fast api" -> "FastAPI", "type script" -> "TypeScript"
5. **Commands** - Sentence splitting + leading/trailing command extraction
6. **Output** - TUI buffer display or keystroke diff-typing via pynput

## Type Mode

`--type` mode sends keystrokes to the focused app. It detects when its own terminal is focused and skips typing to avoid feedback loops. A 3-second countdown lets you switch to the target app after launching.

## Benchmarks

```bash
# Compare ASR models (requires test fixtures in tests/fixtures/)
uv run python tests/benchmark.py

# Pipeline diagnostics
uv run python tests/diagnose_pipeline.py
```

## License

MIT
