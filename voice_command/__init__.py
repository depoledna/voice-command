"""voice-command — VAD-driven streaming voice dictation for macOS."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from voice_command.app import main

try:
    __version__ = _pkg_version("voice-command")
except PackageNotFoundError:  # editable/source checkout without dist-info
    __version__ = "0.0.0+unknown"

__all__ = ["main", "__version__"]
