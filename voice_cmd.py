"""Backwards-compatible entry shim — all logic lives in `voice_command/`.

The PyPI script entry point is `voice-cmd = voice_cmd:main` (see pyproject.toml),
so this thin module preserves that import path while the actual implementation
lives in the `voice_command` package.
"""

from voice_command import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
