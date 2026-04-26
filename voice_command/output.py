"""Output sinks: BufferSink (TUI display) and TypeSink (keystrokes into focused app).

The two modes share the same `OutputSink` protocol so the main loop never
has to branch on settings.mode — it just calls sink.apply, sink.status, etc.

Public API:
    OutputSink   — protocol both sinks satisfy
    BufferSink   — accumulates text in the TUI; auto-clears after idle
    TypeSink     — types each utterance into the focused app via pynput
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Protocol

from voice_command.text import TextBuffer, copy_to_clipboard, normalize_buffer_text
from voice_command.tui import BOLD, GRAY, GREEN, RESET, YELLOW


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class OutputSink(Protocol):
    def before_start(self) -> None: ...
    def status(self, paused: bool, in_speech: bool) -> str: ...
    def apply(self, old_text: str, new_text: str) -> str | None: ...
    def maybe_auto_clear(self, buf: TextBuffer, in_speech: bool) -> TextBuffer | None: ...
    def finalize(self, buf: TextBuffer) -> None: ...


# ---------------------------------------------------------------------------
# Idle-clear helper (shared by both sinks)
# ---------------------------------------------------------------------------

class _IdleClear:
    """Tracks last-utterance timestamp and produces a fresh buffer once the
    user has been silent for `timeout` seconds."""

    def __init__(self, timeout: float) -> None:
        self._timeout = float(timeout)
        self._last = time.monotonic()

    def touch(self) -> None:
        self._last = time.monotonic()

    def maybe_clear(self, buf: TextBuffer, in_speech: bool) -> TextBuffer | None:
        if (self._timeout > 0
                and not in_speech
                and buf.word_count > 0
                and time.monotonic() - self._last > self._timeout):
            return TextBuffer()
        return None


# ---------------------------------------------------------------------------
# BufferSink — text accumulates in the TUI buffer; auto-clears on idle
# ---------------------------------------------------------------------------

class BufferSink:
    """Buffer-mode sink. The TUI renders `buf.text` directly; this sink only
    tracks the inactivity timer and prints the final transcript at exit."""

    def __init__(self, inactivity_clear_seconds: float) -> None:
        self._idle = _IdleClear(inactivity_clear_seconds)

    def before_start(self) -> None:
        pass

    def status(self, paused: bool, in_speech: bool) -> str:
        if paused:
            return f"{YELLOW}⏸ paused{RESET}"
        if in_speech:
            return f"{GREEN}● LISTENING{RESET}"
        return f"{GRAY}○ ready{RESET}"

    def apply(self, old_text: str, new_text: str) -> str | None:
        # any utterance — even a no-op command — resets the idle timer
        self._idle.touch()
        return None

    def maybe_auto_clear(self, buf: TextBuffer, in_speech: bool) -> TextBuffer | None:
        return self._idle.maybe_clear(buf, in_speech)

    def finalize(self, buf: TextBuffer) -> None:
        print(f"{BOLD}Final ({buf.word_count} words):{RESET}")
        if buf.text:
            print(buf.text)
            copy_to_clipboard(buf.text)
            print(f"{GREEN}Copied to clipboard{RESET}")
        else:
            print(f"{GRAY}(empty){RESET}")


# ---------------------------------------------------------------------------
# TypeSink — keystrokes into the focused app
# ---------------------------------------------------------------------------

# macOS terminal app bundle identifiers — used to detect "did the user leave
# focus on the terminal we're running in?", in which case we suppress typing
# (otherwise we'd echo into our own command line).
_TERMINAL_BUNDLES = {
    "com.apple.Terminal",
    "com.googlecode.iterm2",
    "dev.warp.Warp-Stable",
    "co.zeit.hyper",
    "net.kovidgoyal.kitty",
}


class TypeSink:
    """Type-mode sink. Each utterance produces a diff of keystrokes typed into
    whatever app is currently focused (skipped if our own terminal is focused).
    Also accumulates text in the TUI buffer for visual feedback; the buffer
    auto-clears after idle so a fresh utterance types from scratch instead of
    diffing against stale text the user has likely moved on from."""

    def __init__(self, inactivity_clear_seconds: float) -> None:
        # pynput pulls in macOS accessibility plumbing — defer until type mode
        from pynput.keyboard import Controller

        self._kb = Controller()
        self._idle = _IdleClear(inactivity_clear_seconds)

    def before_start(self) -> None:
        for i in range(3, 0, -1):
            print(f"\r{YELLOW}Switch to target app... {i}{RESET}", end="", flush=True)
            time.sleep(1)
        print("\r" + " " * 40 + "\r", end="", flush=True)

    def status(self, paused: bool, in_speech: bool) -> str:
        if paused:
            return f"{YELLOW}⏸ paused{RESET}"
        if in_speech:
            return f"{GREEN}● LISTENING{RESET} → typing"
        return f"{GRAY}○ ready{RESET} (typing into focused window)"

    def apply(self, old_text: str, new_text: str) -> str | None:
        self._idle.touch()
        if old_text == new_text:
            return None
        if _is_own_terminal_focused():
            return "skipped — terminal focused"
        _type_diff(old_text, new_text, self._kb)
        return None

    def maybe_auto_clear(self, buf: TextBuffer, in_speech: bool) -> TextBuffer | None:
        return self._idle.maybe_clear(buf, in_speech)

    def finalize(self, buf: TextBuffer) -> None:
        print(f"{BOLD}Session ended ({buf.word_count} words typed){RESET}")


# ---------------------------------------------------------------------------
# Terminal-focus detection (module-level so it survives across sink instances)
# ---------------------------------------------------------------------------

_own_tty: str | None = None


def _get_own_tty() -> str | None:
    global _own_tty
    if _own_tty is None:
        try:
            _own_tty = os.ttyname(sys.stdin.fileno())
        except OSError:
            _own_tty = ""
    return _own_tty or None


def _is_own_terminal_focused() -> bool:
    """True if the frontmost terminal tab is the one running this script.

    Different tabs have different TTYs, so we ask the terminal app via
    AppleScript and compare against our own /dev/ttysXXX.
    """
    try:
        from AppKit import NSWorkspace

        frontmost = NSWorkspace.sharedWorkspace().frontmostApplication()
        if not frontmost:
            return False

        bundle = frontmost.bundleIdentifier()
        if bundle not in _TERMINAL_BUNDLES:
            return False

        our_tty = _get_own_tty()
        if not our_tty:
            return False

        if bundle == "com.apple.Terminal":
            r = subprocess.run(
                ["osascript", "-e",
                 'tell application "Terminal" to return tty of selected tab of front window'],
                capture_output=True, text=True, timeout=1,
            )
            if r.returncode == 0:
                return r.stdout.strip() == our_tty

        elif bundle == "com.googlecode.iterm2":
            r = subprocess.run(
                ["osascript", "-e",
                 'tell application "iTerm2" to return tty of current session of current window'],
                capture_output=True, text=True, timeout=1,
            )
            if r.returncode == 0:
                return r.stdout.strip() == our_tty

        # unknown terminal — conservative: block typing
        return True

    except Exception:
        return False


# ---------------------------------------------------------------------------
# Keystroke diff helper (private to this module)
# ---------------------------------------------------------------------------

def _type_diff(old_text: str, new_text: str, keyboard) -> None:
    """Type the diff between old and new buffer text via keystrokes.

    Sends Backspace for the deleted suffix of `old_text`, then types the
    new suffix of `new_text`. pynput converts '\\n' to Enter automatically.
    """
    from pynput.keyboard import Key

    old_norm = normalize_buffer_text(old_text)
    new_norm = normalize_buffer_text(new_text)

    common = 0
    for a, b in zip(old_norm, new_norm):
        if a != b:
            break
        common += 1

    for _ in range(len(old_norm) - common):
        keyboard.press(Key.backspace)
        keyboard.release(Key.backspace)

    to_type = new_norm[common:]
    if to_type:
        keyboard.type(to_type)
