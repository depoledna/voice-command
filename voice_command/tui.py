"""Terminal UI: alt-screen lifecycle, frame rendering, key reader, sub-screens.

Public API:
    Tui      — context manager; the only object the app talks to
    Frame    — value passed to Tui.render() each iteration

The Tui owns: alt-screen enter/exit, cursor visibility, termios state,
key-reader thread, ANSI rendering, sub-screen overlays for device + numeric
pickers + help. The caller only sees:

    with Tui() as tui:
        tui.render(Frame(...))
        for key in tui.poll_keys(): ...
        chosen = tui.pick_device(devices, current)
        new_v  = tui.pick_numeric("VAD", 0.45, 0.05, (0.10, 0.95), "{:.2f}")
        tui.show_help(commands)
"""

from __future__ import annotations

import atexit
import io
import os
import queue
import re
import select
import shutil
import sys
import termios
import threading
import tty
from dataclasses import dataclass

from voice_command.config import Settings
from voice_command.text import normalize_buffer_text


# ---------------------------------------------------------------------------
# ANSI codes (full set; re-used by the few callers that build their own ANSI)
# ---------------------------------------------------------------------------

GRAY = "\033[90m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

CLR_EOL = "\033[K"     # erase from cursor to end of line
CLR_BELOW = "\033[J"   # erase from cursor to end of screen
HOME = "\033[H"        # cursor to (1,1)
ALT_SCREEN_ON = "\033[?1049h"
ALT_SCREEN_OFF = "\033[?1049l"
CURSOR_HIDE = "\033[?25l"
CURSOR_SHOW = "\033[?25h"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_ESC_TIMEOUT = 0.05  # 50ms — vim ttimeoutlen, prompt_toolkit


# ---------------------------------------------------------------------------
# Public Frame value
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    settings: Settings
    device_name: str
    body: str          # buffer text — may contain \n
    status: str        # status line content
    message: str = ""  # transient hint shown next to status


# ---------------------------------------------------------------------------
# Render primitives (private)
# ---------------------------------------------------------------------------

def _term_size() -> tuple[int, int]:
    try:
        s = shutil.get_terminal_size()
        return max(20, s.columns), max(5, s.lines)
    except Exception:
        return 80, 24


def _visible_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def _truncate_visible(s: str, max_cols: int) -> str:
    """Truncate to max_cols visible columns, preserving ANSI SGR codes."""
    if max_cols <= 0:
        return ""
    out: list[str] = []
    visible = 0
    i = 0
    n = len(s)
    while i < n:
        m = _ANSI_RE.match(s, i)
        if m:
            out.append(m.group(0))
            i = m.end()
            continue
        if visible >= max_cols:
            break
        out.append(s[i])
        visible += 1
        i += 1
    out.append(RESET)
    return "".join(out)


def _hbar(cols: int) -> str:
    return f"{DIM}{'─' * cols}{RESET}"


def _emit(lines: list[str]) -> None:
    """Single buffered write: HOME, then each line truncated + CLR_EOL."""
    cols, rows = _term_size()
    visible_rows = rows - 1
    buf = io.StringIO()
    buf.write(HOME)
    n = min(len(lines), visible_rows)
    for i in range(n):
        line = _truncate_visible(lines[i], cols)
        buf.write(line)
        buf.write(CLR_EOL)
        if i < n - 1:
            buf.write("\n")
    buf.write("\n")
    buf.write(CLR_BELOW)
    sys.stdout.write(buf.getvalue())
    sys.stdout.flush()


def _header(s: Settings, dev_name: str) -> list[str]:
    cols, _ = _term_size()
    llm = f"{GREEN}on{RESET}" if s.llm_correction else f"{RED}off{RESET}"
    sep = f" {DIM}·{RESET} "
    short_dev = dev_name if len(dev_name) <= 24 else dev_name[:23] + "…"
    h1 = (
        f" {BOLD}voice-command{RESET}{sep}{short_dev}{sep}LLM:{llm}"
        f"{sep}VAD:{s.vad_threshold:.2f}/{s.min_silence_ms}ms"
        f"{sep}mode:{CYAN}{s.mode}{RESET}"
    )
    h2 = f" {DIM}[P]ause  [L]LM  [D]evice  [V]AD  [S]ilence  [?]help  [Q]uit{RESET}"
    return [h1, h2, _hbar(cols)]


def _render_main(frame: Frame) -> None:
    lines = _header(frame.settings, frame.device_name)
    status_line = " " + frame.status + (
        f"   {CYAN}{frame.message}{RESET}" if frame.message else ""
    )
    lines.append(status_line)
    lines.append("")
    if frame.body:
        for line in normalize_buffer_text(frame.body).split("\n"):
            lines.append(f" {GRAY}>{RESET} {line}" if line else "")
    else:
        lines.append(f" {GRAY}(start speaking…){RESET}")
    _emit(lines)


def _render_help(s: Settings, dev_name: str, commands: dict[str, str]) -> None:
    lines = _header(s, dev_name)
    lines.append("")
    lines.append(f" {BOLD}Hotkeys{RESET}")
    lines.append(f"   {YELLOW}P  Space{RESET}  pause / resume listening (live)")
    lines.append(f"   {YELLOW}L{RESET}         toggle LLM tech-term correction (live)")
    lines.append(f"   {YELLOW}D{RESET}         pick audio device          {GRAY}(live){RESET}")
    lines.append(f"   {YELLOW}V / S{RESET}     adjust VAD threshold / min silence (live)")
    lines.append(f"   {YELLOW}?{RESET}         this help")
    lines.append(f"   {YELLOW}Q  Ctrl+C{RESET} quit")
    lines.append("")
    lines.append(f" {BOLD}Voice commands{RESET} {GRAY}(always available){RESET}")
    for cmd, desc in commands.items():
        lines.append(f"   {YELLOW}{cmd:<35}{RESET} {GRAY}{desc}{RESET}")
    lines.append("")
    lines.append(f" {GRAY}press any key to dismiss{RESET}")
    _emit(lines)


def _render_device_picker(
    s: Settings,
    dev_name: str,
    devices: list[tuple[int, str]],
    selected_idx: int,
    current: int | None,
) -> None:
    lines = _header(s, dev_name)
    lines.append("")
    lines.append(f" {BOLD}Select audio device{RESET}  {GRAY}(↑↓ to move · 1–9 jump · Enter save · Esc cancel){RESET}")
    lines.append("")
    for i, (idx, name) in enumerate(devices):
        marker = f"{CYAN}▸{RESET}" if i == selected_idx else " "
        num = f"{YELLOW}{i + 1}{RESET}" if i < 9 else f"{GRAY}·{RESET}"
        suffix = f"  {GRAY}(current){RESET}" if idx == current else ""
        prefix_bold = BOLD if i == selected_idx else ""
        lines.append(f" {marker} {num}. {prefix_bold}{name}{RESET}{suffix}")
    _emit(lines)


def _render_numeric_picker(s: Settings, dev_name: str, label: str, value: str, hint: str) -> None:
    lines = _header(s, dev_name)
    lines.append("")
    lines.append(f" {BOLD}{label}{RESET}: {CYAN}{value}{RESET}")
    lines.append(f" {GRAY}{hint}{RESET}")
    _emit(lines)


# ---------------------------------------------------------------------------
# Key reader (private helper started by Tui.__enter__)
# ---------------------------------------------------------------------------

def _start_key_reader(key_q: queue.Queue, stop):
    """Put stdin in cbreak; spawn a thread that pushes parsed keys onto key_q.

    cbreak (vs raw) keeps ISIG so Ctrl+C still raises SIGINT. Reads in chunks
    so paste/arrow bursts arrive together. Lone ESC waits 50ms for a follow-up
    before firing as a bare Esc.
    """
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    def _read_chunk() -> str:
        try:
            data = os.read(fd, 32)
        except (OSError, BlockingIOError):
            return ""
        return data.decode("utf-8", errors="replace")

    def loop() -> None:
        pending = ""
        while not stop():
            try:
                r, _, _ = select.select([fd], [], [], 0.05)
            except (OSError, ValueError):
                return
            if not r:
                if pending == "\x1b":
                    key_q.put("\x1b")
                    pending = ""
                continue
            chunk = pending + _read_chunk()
            pending = ""
            i = 0
            while i < len(chunk):
                ch = chunk[i]
                if ch == "\x1b":
                    if i + 1 >= len(chunk):
                        try:
                            r2, _, _ = select.select([fd], [], [], _ESC_TIMEOUT)
                        except (OSError, ValueError):
                            r2 = None
                        if r2:
                            chunk += _read_chunk()
                        else:
                            key_q.put("\x1b")
                            i += 1
                            continue
                    if i + 1 < len(chunk) and chunk[i + 1] == "[":
                        j = i + 2
                        while j < len(chunk) and not (0x40 <= ord(chunk[j]) <= 0x7e):
                            j += 1
                        if j < len(chunk):
                            key_q.put(chunk[i:j + 1])
                            i = j + 1
                        else:
                            pending = chunk[i:]
                            break
                    elif i + 1 < len(chunk) and chunk[i + 1] == "O":
                        if i + 2 < len(chunk):
                            key_q.put(chunk[i:i + 3])
                            i += 3
                        else:
                            pending = chunk[i:]
                            break
                    else:
                        key_q.put(chunk[i:i + 2])
                        i += 2
                else:
                    key_q.put(ch)
                    i += 1

    t = threading.Thread(target=loop, daemon=True)
    t.start()

    def cleanup() -> None:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        except (termios.error, OSError):
            pass

    return cleanup


# ---------------------------------------------------------------------------
# Atexit safety net (in case the `with` block was bypassed)
# ---------------------------------------------------------------------------

_alt_screen_active = False


def _force_exit_alt_screen() -> None:
    global _alt_screen_active
    if _alt_screen_active:
        try:
            sys.stdout.write(CURSOR_SHOW + ALT_SCREEN_OFF)
            sys.stdout.flush()
        except Exception:
            pass
        _alt_screen_active = False


atexit.register(_force_exit_alt_screen)


# ---------------------------------------------------------------------------
# Tui — the only class the app sees
# ---------------------------------------------------------------------------

class Tui:
    """Terminal UI lifecycle + render + input + sub-screens.

    Use as a context manager. While inside `with Tui() as tui:` the alt-screen
    is active, the cursor is hidden, and a background thread reads keys.
    """

    def __init__(self) -> None:
        self._key_q: queue.Queue[str] = queue.Queue()
        self._cleanup_keys = None
        self._running = True
        # Cached last frame — sub-screens reuse settings + dev_name from here
        # so we don't make every picker re-take them.
        self._last_settings: Settings | None = None
        self._last_dev_name: str = ""

    def __enter__(self) -> "Tui":
        global _alt_screen_active
        sys.stdout.write(ALT_SCREEN_ON + CURSOR_HIDE + HOME)
        sys.stdout.flush()
        _alt_screen_active = True
        self._cleanup_keys = _start_key_reader(self._key_q, lambda: not self._running)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        global _alt_screen_active
        self._running = False
        if self._cleanup_keys:
            try:
                self._cleanup_keys()
            except Exception:
                pass
        try:
            sys.stdout.write(CURSOR_SHOW + ALT_SCREEN_OFF)
            sys.stdout.flush()
        except Exception:
            pass
        _alt_screen_active = False

    # -- main render --------------------------------------------------------

    def render(self, frame: Frame) -> None:
        self._last_settings = frame.settings
        self._last_dev_name = frame.device_name
        _render_main(frame)

    # -- key polling --------------------------------------------------------

    def poll_keys(self) -> list[str]:
        """Drain the key queue without blocking. Returns 0+ keystrokes."""
        keys: list[str] = []
        while True:
            try:
                keys.append(self._key_q.get_nowait())
            except queue.Empty:
                return keys

    # -- sub-screens --------------------------------------------------------

    def show_help(self, commands: dict[str, str]) -> None:
        s = self._last_settings
        if s is None:
            return
        _render_help(s, self._last_dev_name, commands)
        # block until any key
        while True:
            try:
                self._key_q.get(timeout=0.5)
            except queue.Empty:
                continue
            return

    def pick_device(
        self,
        devices: list[tuple[int, str]],
        current: int | None,
    ) -> int | None:
        s = self._last_settings
        if s is None or not devices:
            return None
        selected = 0
        for i, (idx, _) in enumerate(devices):
            if idx == current:
                selected = i
                break
        _render_device_picker(s, self._last_dev_name, devices, selected, current)
        while True:
            try:
                key = self._key_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if key in ("\x1b", "q", "Q"):
                return None
            if key in ("\x1b[A", "k"):
                selected = max(0, selected - 1)
            elif key in ("\x1b[B", "j"):
                selected = min(len(devices) - 1, selected + 1)
            elif key in ("\r", "\n"):
                return devices[selected][0]
            elif key.isdigit() and key != "0":
                n = int(key) - 1
                if n < len(devices):
                    return devices[n][0]
            else:
                continue
            _render_device_picker(s, self._last_dev_name, devices, selected, current)

    def pick_numeric(
        self,
        label: str,
        value: float,
        step: float,
        bounds: tuple[float, float],
        fmt: str,
    ) -> float | None:
        s = self._last_settings
        if s is None:
            return None
        vmin, vmax = bounds
        val = value
        hint = f"↑↓ / +/- to adjust by {step} · Enter save · Esc cancel"
        _render_numeric_picker(s, self._last_dev_name, label, fmt.format(val), hint)
        while True:
            try:
                key = self._key_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if key in ("\x1b", "q", "Q"):
                return None
            if key in ("\x1b[A", "k", "+", "="):
                val = min(vmax, val + step)
            elif key in ("\x1b[B", "j", "-", "_"):
                val = max(vmin, val - step)
            elif key in ("\r", "\n"):
                return val
            else:
                continue
            _render_numeric_picker(s, self._last_dev_name, label, fmt.format(val), hint)
