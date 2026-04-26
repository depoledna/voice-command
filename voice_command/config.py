"""Persistent settings stored at $XDG_CONFIG_HOME/voice-command/settings.json.

Public API:
    Settings       — dataclass with all user-tunable knobs
    path()         — resolved config file path
    load()         — read from disk, fall back to defaults on missing/malformed
    save(s)        — write to disk (creates parent dirs as needed)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    device: int | None = None
    llm_correction: bool = False
    vad_threshold: float = 0.45
    min_silence_ms: int = 600
    inactivity_clear_seconds: float = 5.0   # 0 disables auto-clear (buffer + status message)


def path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "voice-command" / "settings.json"


def load() -> Settings:
    p = path()
    if not p.exists():
        s = Settings()
        save(s)
        return s
    defaults = Settings()
    try:
        data = json.loads(p.read_text())
        return Settings(
            device=data.get("device", defaults.device),
            llm_correction=bool(data.get("llm_correction", defaults.llm_correction)),
            vad_threshold=float(data.get("vad_threshold", defaults.vad_threshold)),
            min_silence_ms=int(data.get("min_silence_ms", defaults.min_silence_ms)),
            inactivity_clear_seconds=float(
                data.get("inactivity_clear_seconds", defaults.inactivity_clear_seconds) or 0
            ),
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
        print(f"warning: ignoring malformed {p}: {e}", file=sys.stderr)
        return defaults


def save(s: Settings) -> None:
    p = path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "device": s.device,
        "llm_correction": s.llm_correction,
        "vad_threshold": s.vad_threshold,
        "min_silence_ms": s.min_silence_ms,
        "inactivity_clear_seconds": s.inactivity_clear_seconds,
    }, indent=2) + "\n")
