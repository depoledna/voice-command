# voice-command

[![PyPI](https://img.shields.io/pypi/v/voice-command)](https://pypi.org/project/voice-command/)
[![CI](https://github.com/depoledna/voice-command/actions/workflows/ci.yml/badge.svg)](https://github.com/depoledna/voice-command/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab.svg)](https://python.org)
[![macOS](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-black.svg)](https://github.com/depoledna/voice-command)

VAD-driven streaming voice dictation for macOS (Apple Silicon). Speaks into your mic, text appears in a terminal buffer or gets typed directly into any app.

All inference runs locally — Whisper for ASR, Silero for VAD, Qwen for tech-term correction. No cloud APIs.

## Requirements

- macOS with Apple Silicon
- Python 3.12+
- Microphone access (System Settings > Privacy > Microphone)
- Accessibility access for `--type` mode (System Settings > Privacy > Accessibility)

## Install

### From PyPI

```bash
pip install voice-command
# or
uv tool install voice-command
```

### From source

```bash
git clone https://github.com/depoledna/voice-command.git
cd voice-command
uv sync
```

Models download automatically on first run (~300MB for Whisper + ~1GB for Qwen).

## Usage

```bash
# Terminal buffer mode (TUI)
voice-cmd

# Type directly into the focused app
voice-cmd --type

# Transcribe a recording
voice-cmd --file recording.m4a

# List audio devices
voice-cmd --list-devices

# Use a specific device
voice-cmd --device 1
```

When running from source, use `uv run python voice_cmd.py` instead of `voice-cmd`.

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

## Releasing

1. Update the version in `pyproject.toml`
2. Commit: `git commit -am "chore: bump version to X.Y.Z"`
3. Tag: `git tag vX.Y.Z`
4. Push: `git push origin main --tags`

The GitHub Actions workflow builds and publishes to PyPI automatically via trusted publishers (OIDC).

### First-time PyPI setup

1. Go to https://pypi.org/manage/account/publishing/
2. Add a "pending publisher":
   - Package name: `voice-command`
   - Owner: `depoledna`
   - Repository: `voice-command`
   - Workflow: `release.yml`
   - Environment: `pypi`
3. In the GitHub repo, go to Settings > Environments > create `pypi`

## License

MIT
