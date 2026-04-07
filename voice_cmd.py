"""Voice Command — VAD-driven streaming voice input with editing commands.

Usage:
    uv run python voice_cmd.py                 # live mic → terminal buffer
    uv run python voice_cmd.py --type          # live mic → type into active window
    uv run python voice_cmd.py --device 1
    uv run python voice_cmd.py --list-devices
    uv run python voice_cmd.py --file recording.m4a
"""

import argparse
import os
import queue
import re
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from math import gcd

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from silero_vad_lite import SileroVAD


# ---------------------------------------------------------------------------
# Text buffer (immutable with undo)
# ---------------------------------------------------------------------------

@dataclass
class TextBuffer:
    _words: list[str] = field(default_factory=list)
    _history: deque[list[str]] = field(default_factory=lambda: deque(maxlen=50))

    def _snapshot(self) -> None:
        self._history.append(list(self._words))

    def append_text(self, text: str) -> "TextBuffer":
        words = text.split()
        if not words:
            return self
        self._snapshot()
        return TextBuffer(_words=list(self._words) + words, _history=self._history)

    def delete_last_n(self, n: int) -> "TextBuffer":
        if n <= 0:
            return self
        self._snapshot()
        return TextBuffer(_words=list(self._words)[: max(0, len(self._words) - n)], _history=self._history)

    def clear(self) -> "TextBuffer":
        self._snapshot()
        return TextBuffer(_words=[], _history=self._history)

    def undo(self) -> "TextBuffer":
        if not self._history:
            return self
        prev = self._history.pop()
        return TextBuffer(_words=prev, _history=self._history)

    def append_punctuation(self, punct: str) -> "TextBuffer":
        """Attach punctuation to the last word (no space before it).

        Replaces trailing punctuation if the word already ends with one,
        so "writing," + "." → "writing." not "writing,."
        """
        if not self._words:
            return self
        self._snapshot()
        new_words = list(self._words)
        last = new_words[-1]
        # strip existing trailing punctuation before attaching new one
        if last and last[-1] in ".,!?;:":
            last = last[:-1]
        new_words[-1] = last + punct
        return TextBuffer(_words=new_words, _history=self._history)

    @property
    def text(self) -> str:
        return " ".join(self._words)

    @property
    def word_count(self) -> int:
        return len(self._words)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

COMMANDS: dict[str, str] = {
    "delete last N words": "Remove the last N words",
    "scratch that": "Delete last phrase (~5 words)",
    "undo": "Undo the last action",
    "clear all": "Clear all text",
    "new line": "Insert a line break",
    "new paragraph": "Insert two line breaks",
    "period / comma / question mark": "Insert punctuation",
    "stop listening": "Pause transcription",
    "start listening": "Resume transcription",
    "copy all": "Copy buffer to clipboard",
    "done": "Copy to clipboard and exit",
    "show commands": "Show this list",
}

PUNCTUATION_MAP = {
    "period": ".", "full stop": ".", "comma": ",",
    "question mark": "?", "exclamation mark": "!", "exclamation point": "!",
    "colon": ":", "semicolon": ";", "dash": "—", "hyphen": "-",
    "ellipsis": "...",
}

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twenty": 20,
}

COMMAND_TRIGGERS = {
    "new line", "newline", "new paragraph", "new para",
    "period", "full stop", "comma", "question mark",
    "exclamation mark", "exclamation point", "colon", "semicolon",
    "dash", "hyphen", "ellipsis",
    "scratch that", "scratch this", "delete that",
    "undo", "undo that",
    "clear all", "clear everything", "erase all",
    "show commands", "show command", "list commands", "help",
    "stop listening", "pause", "stop",
    "start listening", "resume", "start",
    "copy all", "copy text", "copy",
    "done", "finish", "exit", "quit",
}


@dataclass
class CommandResult:
    handled: bool
    buffer: TextBuffer
    message: str = ""
    should_pause: bool = False
    should_resume: bool = False
    should_exit: bool = False
    show_commands: bool = False


def _extract_number(text: str) -> int | None:
    for word in text.lower().split():
        if word.isdigit():
            return int(word)
        if word in NUMBER_WORDS:
            return NUMBER_WORDS[word]
    return None


def try_command(text: str, buf: TextBuffer) -> CommandResult:
    """Check if text matches a voice command."""
    t = text.lower().strip().rstrip(".,!?;:")

    if t in ("show commands", "show command", "list commands", "help"):
        return CommandResult(True, buf, show_commands=True)

    if t in ("undo", "undo that"):
        return CommandResult(True, buf.undo(), message="Undone")

    if t in ("scratch that", "scratch this", "delete that"):
        return CommandResult(True, buf.delete_last_n(5), message="Scratched")

    if t in ("clear all", "clear everything", "erase all"):
        return CommandResult(True, buf.clear(), message="Cleared")

    if "delete last" in t and ("word" in t or "words" in t):
        n = _extract_number(t)
        if n:
            return CommandResult(True, buf.delete_last_n(n), message=f"Deleted {n} words")

    if t in ("new line", "newline"):
        new_words = list(buf._words) + ["\n"]
        buf._snapshot()
        return CommandResult(True, TextBuffer(_words=new_words, _history=buf._history), message="New line")

    if t in ("new paragraph", "new para"):
        new_words = list(buf._words) + ["\n\n"]
        buf._snapshot()
        return CommandResult(True, TextBuffer(_words=new_words, _history=buf._history), message="New paragraph")

    if t in ("stop listening", "pause", "stop"):
        return CommandResult(True, buf, should_pause=True, message="Paused")

    if t in ("start listening", "resume", "start"):
        return CommandResult(True, buf, should_resume=True, message="Resumed")

    if t in ("copy all", "copy text", "copy"):
        _copy_to_clipboard(buf.text)
        return CommandResult(True, buf, message="Copied to clipboard")

    if t in ("done", "finish", "exit", "quit"):
        _copy_to_clipboard(buf.text)
        return CommandResult(True, buf, should_exit=True, message="Done")

    for trigger, punct in PUNCTUATION_MAP.items():
        if t == trigger:
            return CommandResult(True, buf.append_punctuation(punct), message=f"+ {trigger}")

    return CommandResult(False, buf)


def _copy_to_clipboard(text: str) -> None:
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Keyboard typing helpers (for --type mode)
# ---------------------------------------------------------------------------

def _normalize_buf_text(text: str) -> str:
    """Normalize TextBuffer output for keyboard typing.

    TextBuffer joins words with spaces, including around \\n tokens.
    Strip spaces adjacent to newlines so we get clean line breaks.
    """
    return re.sub(r" *(\n+) *", r"\1", text)


def _type_diff(old_text: str, new_text: str, keyboard) -> None:
    """Type the difference between old and new buffer states.

    Finds the common prefix, sends Backspace for removed chars,
    then types the new chars.
    """
    from pynput.keyboard import Key

    old_norm = _normalize_buf_text(old_text)
    new_norm = _normalize_buf_text(new_text)

    # find common prefix length
    common = 0
    for a, b in zip(old_norm, new_norm):
        if a != b:
            break
        common += 1

    # delete chars after common prefix
    to_delete = len(old_norm) - common
    for _ in range(to_delete):
        keyboard.press(Key.backspace)
        keyboard.release(Key.backspace)

    # type new chars after common prefix
    to_type = new_norm[common:]
    if to_type:
        # pynput.type() handles Enter for \n automatically
        keyboard.type(to_type)


_TERMINAL_BUNDLES = {
    "com.apple.Terminal",
    "com.googlecode.iterm2",
    "dev.warp.Warp-Stable",
    "co.zeit.hyper",
    "net.kovidgoyal.kitty",
}

# cache our TTY path once (doesn't change during the session)
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
    """Check if the specific terminal TAB running this script is focused.

    Compares our TTY (/dev/ttysXXX) against the frontmost terminal tab's TTY
    via AppleScript. Different terminal windows/tabs have different TTYs.
    """
    try:
        from AppKit import NSWorkspace

        frontmost = NSWorkspace.sharedWorkspace().frontmostApplication()
        if not frontmost:
            return False

        bundle = frontmost.bundleIdentifier()
        if bundle not in _TERMINAL_BUNDLES:
            return False

        # it's a terminal app — compare TTYs to check if it's OUR tab
        our_tty = _get_own_tty()
        if not our_tty:
            return False

        import subprocess

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


def extract_trailing_command(text: str) -> tuple[str, str | None]:
    """Try to find a command at the tail of text."""
    words = text.split()
    for n in range(1, min(6, len(words) + 1)):
        tail = " ".join(words[-n:]).lower().rstrip(".,!?;:")
        if tail in COMMAND_TRIGGERS or ("delete last" in tail and ("word" in tail or "words" in tail)):
            before = " ".join(words[:-n]).strip()
            return before, " ".join(words[-n:])
    return text, None


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks on '. ' boundaries.

    Falls back to comma-splitting when sentence-splitting produces only
    one piece but it contains multiple command words (common with Whisper
    output like "colon, new line, milk, comma, bread, period").
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1 and ", " in text:
        # count command words in the text
        lower = text.lower()
        cmd_count = sum(1 for trigger in COMMAND_TRIGGERS if f", {trigger}" in lower or lower.startswith(trigger))
        if cmd_count >= 2:
            # split on commas instead
            parts = [p.strip() for p in text.split(", ") if p.strip()]

    return parts


def _clean_command_text(text: str) -> str:
    """Strip punctuation that ASR adds around command words."""
    return text.strip().strip(".,!?;:").strip()


def _extract_leading_command(text: str) -> tuple[str | None, str]:
    """Try to find a command at the start of text.

    Returns (command_text, remaining) or (None, text).
    """
    words = text.split()
    for n in range(min(6, len(words)), 0, -1):
        head = " ".join(words[:n])
        cleaned = _clean_command_text(head).lower()
        if cleaned in COMMAND_TRIGGERS or ("delete last" in cleaned and ("word" in cleaned or "words" in cleaned)):
            after = " ".join(words[n:]).strip().lstrip(".,!?;:").strip()
            return head, after
    return None, text


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

CONTRACTION_FIXES = [
    (r"\b(\w+)\s+'(m|s|t|re|ve|ll|d)\b", r"\1'\2"),
    (r"\b(\w+)\s+n't\b", r"\1n't"),
]


def fix_contractions(text: str) -> str:
    result = text
    for pattern, replacement in CONTRACTION_FIXES:
        result = re.sub(pattern, replacement, result)
    return result


# ---------------------------------------------------------------------------
# LLM post-processing for developer vocabulary
# ---------------------------------------------------------------------------

LLM_MODEL_ID = "mlx-community/Qwen3-1.7B-4bit"

_llm_model = None
_llm_tokenizer = None
_llm_sampler = None

LLM_PROMPT_TEMPLATE = (
    "Correct the technical terms in the text below. "
    "Replace: engine x→Nginx, fast api→FastAPI, post gres→Postgres, web socket→WebSocket, "
    "web hook→webhook, type script→TypeScript, java script→JavaScript, next js→Next.js, "
    "node js→Node.js, graph ql→GraphQL, pie test→pytest, sequel alchemy→SQLAlchemy, "
    "x code→Xcode, CI CD→CI/CD. "
    "Fix casing: api→API, json→JSON, http→HTTP, url→URL, rest→REST, sql→SQL, "
    "yaml→YAML, jwt→JWT, oauth→OAuth, dns→DNS, ssl→SSL, tls→TLS, ssh→SSH, cli→CLI, "
    "github→GitHub, docker→Docker, kubernetes→Kubernetes, redis→Redis, "
    "macos→macOS, ios→iOS, python→Python, django→Django, npm→npm. "
    "Output only the corrected text.\n\n{text}"
)


def _looks_like_command(text: str) -> bool:
    """Check if text is likely a voice command (skip LLM for these)."""
    cleaned = text.strip().lower().rstrip(".,!?;:")
    if cleaned in COMMAND_TRIGGERS:
        return True
    if "delete last" in cleaned and ("word" in cleaned or "words" in cleaned):
        return True
    # check for punctuation-only commands
    if cleaned in PUNCTUATION_MAP:
        return True
    return False


def fix_tech_terms(text: str) -> str:
    """Fix technical term casing and misrecognitions using a local LLM."""
    global _llm_model, _llm_tokenizer, _llm_sampler

    if not text.strip():
        return text

    # skip LLM for command-only utterances (LLM hallucinates on short commands)
    if _looks_like_command(text):
        return text

    if _llm_model is None:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        _llm_model, _llm_tokenizer = load(LLM_MODEL_ID)
        _llm_sampler = make_sampler(temp=0.0)

    from mlx_lm import generate

    prompt = LLM_PROMPT_TEMPLATE.format(text=text)
    messages = [{"role": "user", "content": prompt}]
    prompt_text = _llm_tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )
    result = generate(
        _llm_model, _llm_tokenizer,
        prompt=prompt_text,
        max_tokens=len(text.split()) * 3,
        sampler=_llm_sampler,
        verbose=False,
    )

    corrected = result.strip()
    # sanity check: if LLM output is wildly different length, keep original
    if not corrected or len(corrected) > len(text) * 2 or len(corrected) < len(text) * 0.3:
        return text
    return corrected


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

GRAY = "\033[90m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
SEP = f"{DIM}{'─' * 70}{RESET}"


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def render(buf: TextBuffer, message: str = "", paused: bool = False) -> None:
    clear_screen()
    state = f"{RED}PAUSED{RESET}" if paused else f"{GREEN}LISTENING{RESET}"
    print(f"{BOLD}Voice Command{RESET}  {DIM}|{RESET}  {state}  {DIM}|{RESET}  {DIM}Words:{RESET} {buf.word_count}  {DIM}|{RESET}  {GRAY}'show commands' for help{RESET}")
    print(SEP)

    text = buf.text
    if text:
        print(f"\n{text}\n")
    else:
        print(f"\n{GRAY}(start speaking...){RESET}\n")

    print(SEP)
    if message:
        print(f"{CYAN}> {message}{RESET}")
    sys.stdout.flush()


def render_commands() -> None:
    clear_screen()
    print(f"{BOLD}Voice Commands{RESET}")
    print(SEP)
    for cmd, desc in COMMANDS.items():
        print(f"  {YELLOW}{cmd:<35}{RESET} {GRAY}{desc}{RESET}")
    print(SEP)
    print(f"\n{GRAY}Say anything to dismiss...{RESET}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# VAD state machine
# ---------------------------------------------------------------------------

FRAME_SAMPLES = 512  # 32ms @ 16kHz


@dataclass
class VADConfig:
    threshold: float = 0.45
    neg_threshold: float = 0.25
    min_speech_ms: int = 64
    min_silence_ms: int = 600
    speech_pad_ms: int = 300
    frame_ms: int = 32


class VADState:
    """Wraps silero-vad-lite with hysteresis for speech boundary detection."""

    def __init__(self, config: VADConfig | None = None):
        self.cfg = config or VADConfig()
        self._vad = SileroVAD(SAMPLE_RATE)
        self._speaking = False
        self._silence_frames = 0
        self._speech_frames = 0
        pad_frames = max(1, self.cfg.speech_pad_ms // self.cfg.frame_ms)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=pad_frames)

    def process_frame(self, frame_16k: np.ndarray) -> tuple[str, list[np.ndarray]]:
        """Feed one 512-sample frame (16kHz float32).

        Returns (event, audio_frames) where event is:
            "speech_start"    — includes pre-roll + current frame
            "speech_continue" — [current_frame]
            "speech_end"      — []
            "silence"         — []
        """
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

        # currently speaking
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
# ASR helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
MODEL_ID = "mlx-community/whisper-small-mlx-8bit"
INITIAL_PROMPT = (
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


def transcribe_audio(audio: np.ndarray) -> str:
    """Transcribe audio using mlx-whisper. Lazy-imports on first call."""
    try:
        from mlx_whisper import transcribe as mlx_transcribe

        result = mlx_transcribe(
            audio,
            path_or_hf_repo=MODEL_ID,
            language="en",
            initial_prompt=INITIAL_PROMPT,
        )
        return result.get("text", "").strip()
    except Exception as e:
        print(f"{RED}Transcription error: {e}{RESET}")
        return ""


def find_input_device(preferred: int | None) -> tuple[int, str, int]:
    if preferred is not None:
        info = sd.query_devices(preferred)
        return preferred, info["name"], int(info["default_samplerate"])

    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and "airpod" in d["name"].lower():
            return i, d["name"], int(d["default_samplerate"])

    idx = sd.default.device[0]
    info = sd.query_devices(idx)
    return idx, info["name"], int(info["default_samplerate"])


def make_resampler(orig_sr: int) -> tuple[int, int, int]:
    """Return (up, down, callback_blocksize) for orig_sr → 16kHz."""
    if orig_sr == SAMPLE_RATE:
        return 1, 1, FRAME_SAMPLES

    g = gcd(orig_sr, SAMPLE_RATE)
    up = SAMPLE_RATE // g
    down = orig_sr // g
    blocksize = FRAME_SAMPLES * down // up
    return up, down, blocksize


def resample_frame(audio: np.ndarray, up: int, down: int) -> np.ndarray:
    if up == 1 and down == 1:
        return audio.flatten().astype(np.float32)
    return resample_poly(audio.flatten().astype(np.float32), up, down).astype(np.float32)


# ---------------------------------------------------------------------------
# Process a complete utterance
# ---------------------------------------------------------------------------

def _process_single_piece(text: str, buf: TextBuffer) -> tuple[TextBuffer, list[str], CommandResult | None]:
    """Process a single text piece (no sentence splitting).

    Checks: whole command → leading command → trailing command → plain text.
    Returns (buffer, messages, last_cmd).
    """
    # try whole text as command (strip ASR punctuation for matching)
    cmd = try_command(text, buf)
    if cmd.handled:
        return cmd.buffer, [cmd.message], cmd

    # try leading command: "New line, second sentence here"
    lead_cmd, remaining = _extract_leading_command(text)
    if lead_cmd:
        cmd = try_command(lead_cmd, buf)
        if cmd.handled:
            msgs = [cmd.message]
            last = cmd
            if remaining and not cmd.should_exit and not cmd.should_pause:
                buf = cmd.buffer
                buf, sub_msgs, sub_cmd = _process_single_piece(remaining, buf)
                msgs.extend(sub_msgs)
                if sub_cmd:
                    last = sub_cmd
                return buf, msgs, last
            return cmd.buffer, msgs, cmd

    # try trailing command: "some text period"
    before, cmd_text = extract_trailing_command(text)
    if cmd_text:
        msgs = []
        last_cmd = None
        if before:
            buf, sub_msgs, sub_cmd = _process_single_piece(before, buf)
            msgs.extend(sub_msgs)
            if sub_cmd:
                last_cmd = sub_cmd
        cmd = try_command(cmd_text, buf)
        if cmd.handled:
            msgs.append(cmd.message)
            return cmd.buffer, msgs, cmd

    # no command — append as text
    buf = buf.append_text(text)
    return buf, [f'+ "{text}"'], None


def process_utterance(text: str, buf: TextBuffer) -> tuple[TextBuffer, str, CommandResult | None]:
    """Process a complete utterance: normalize, detect commands, append text.

    Splits longer utterances on sentence boundaries and checks each piece
    for commands — both leading and trailing. Handles mid-text commands like
    "First sentence. Period. New line. Second sentence."

    Returns (updated_buffer, message, last_command_result_or_None).
    """
    text = fix_contractions(text)
    text = fix_tech_terms(text)

    # try whole text as a single command first
    cmd = try_command(text, buf)
    if cmd.handled:
        return cmd.buffer, cmd.message, cmd

    # split into sentences and process each
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        buf, msgs, last_cmd = _process_single_piece(text, buf)
        return buf, " → ".join(msgs), last_cmd

    messages: list[str] = []
    last_cmd: CommandResult | None = None

    for sentence in sentences:
        buf, msgs, cmd = _process_single_piece(sentence, buf)
        messages.extend(msgs)
        if cmd:
            last_cmd = cmd
            if cmd.should_exit or cmd.should_pause:
                break

    return buf, " → ".join(messages), last_cmd


# ---------------------------------------------------------------------------
# File test mode
# ---------------------------------------------------------------------------

def run_file_test(path: str) -> None:
    """Transcribe a file using VAD segmentation + per-utterance ASR."""
    import librosa as lr

    audio, _sr = lr.load(path, sr=SAMPLE_RATE, mono=True)
    audio = audio.astype(np.float32)
    print(f"{GRAY}Audio: {len(audio)/SAMPLE_RATE:.1f}s{RESET}")

    print(f"{GRAY}Loading models...{RESET}")
    transcribe_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    fix_tech_terms("test")
    print(f"{GREEN}Models ready{RESET}\n")

    vad = VADState()
    buf = TextBuffer()
    utterance_audio: list[np.ndarray] = []

    for i in range(0, len(audio) - FRAME_SAMPLES, FRAME_SAMPLES):
        frame = audio[i:i + FRAME_SAMPLES]
        event, frames = vad.process_frame(frame)

        if event == "speech_start":
            utterance_audio = list(frames)
        elif event == "speech_continue":
            utterance_audio.extend(frames)
        elif event == "speech_end" and utterance_audio:
            all_audio = np.concatenate(utterance_audio)
            utterance_audio = []
            full_text = transcribe_audio(all_audio)

            if full_text:
                buf, msg, _cmd = process_utterance(full_text, buf)
                print(f"  {GREEN}>{RESET} {msg}")

    # handle trailing speech
    if utterance_audio:
        all_audio = np.concatenate(utterance_audio)
        full_text = transcribe_audio(all_audio)

        if full_text:
            buf, msg, _cmd = process_utterance(full_text, buf)
            print(f"  {GREEN}>{RESET} {msg} {GRAY}(trailing){RESET}")

    print(f"\n{SEP}")
    print(f"{BOLD}Final ({buf.word_count} words):{RESET}")
    print(buf.text)


# ---------------------------------------------------------------------------
# Live mic mode
# ---------------------------------------------------------------------------

RENDER_INTERVAL = 0.15  # max render rate (~7fps)


def run_live(device: int | None, vad_threshold: float, min_silence: int) -> None:
    dev_idx, dev_name, dev_rate = find_input_device(device)

    print(f"{GRAY}Device: {dev_name} ({dev_rate}Hz){RESET}")
    print(f"{GRAY}Loading models...{RESET}")

    transcribe_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    fix_tech_terms("test")
    print(f"{GREEN}Models ready{RESET}")
    time.sleep(0.3)

    # resampling setup
    up, down, callback_blocksize = make_resampler(dev_rate)

    # VAD setup
    vad_cfg = VADConfig(threshold=vad_threshold, min_silence_ms=min_silence)
    vad = VADState(vad_cfg)

    # audio queue: callback → main loop
    audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)

    def audio_callback(indata, frames, time_info, status):
        try:
            audio_q.put_nowait(indata.copy())
        except queue.Full:
            pass  # drop frame on overrun

    # state
    buf = TextBuffer()
    paused = False
    running = True
    message = "Ready"
    showing_commands = False
    last_render = 0.0

    # current utterance
    utterance_audio: list[np.ndarray] = []
    in_speech = False

    def on_sigint(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_sigint)

    render(buf, message=message)

    input_stream = sd.InputStream(
        device=dev_idx,
        samplerate=dev_rate,
        channels=1,
        blocksize=callback_blocksize,
        callback=audio_callback,
        dtype="float32",
    )

    try:
        with input_stream:
            while running:
                try:
                    raw_audio = audio_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                frame_16k = resample_frame(raw_audio, up, down)

                # --- paused mode: only listen for "start listening" ---
                if paused:
                    event, frames = vad.process_frame(frame_16k)

                    if event == "speech_start":
                        utterance_audio = list(frames)
                    elif event == "speech_continue":
                        utterance_audio.extend(frames)
                    elif event == "speech_end" and utterance_audio:
                        all_audio = np.concatenate(utterance_audio)
                        utterance_audio = []
                        text = transcribe_audio(all_audio)
                        if text:
                            cmd = try_command(text, buf)
                            if cmd.should_resume:
                                paused = False
                                message = "Resumed"
                                render(buf, message=message)
                    continue

                # --- active listening ---
                event, frames = vad.process_frame(frame_16k)

                if event == "speech_start":
                    utterance_audio = list(frames)
                    in_speech = True
                    if showing_commands:
                        showing_commands = False
                    render(buf, message="Listening...", paused=False)
                    last_render = time.monotonic()

                elif event == "speech_continue" and in_speech:
                    utterance_audio.extend(frames)

                elif event == "speech_end" and in_speech:
                    in_speech = False
                    if utterance_audio:
                        all_audio = np.concatenate(utterance_audio)
                        utterance_audio = []
                        full_text = transcribe_audio(all_audio)
                    else:
                        full_text = ""

                    if full_text:
                        buf, message, cmd = process_utterance(full_text, buf)
                        if cmd:
                            if cmd.should_pause:
                                paused = True
                            if cmd.should_exit:
                                render(buf, message="Goodbye!")
                                running = False
                                continue
                            if cmd.show_commands:
                                render_commands()
                                showing_commands = True
                                continue

                    render(buf, message=message)
                    last_render = time.monotonic()

                elif event == "silence" and not showing_commands:
                    now = time.monotonic()
                    if now - last_render >= RENDER_INTERVAL * 5:
                        render(buf, message=message)
                        last_render = now

    except KeyboardInterrupt:
        pass

    print(f"\n{SEP}")
    print(f"{BOLD}Final ({buf.word_count} words):{RESET}")
    if buf.text:
        print(buf.text)
        _copy_to_clipboard(buf.text)
        print(f"\n{GREEN}Copied to clipboard{RESET}")
    else:
        print(f"{GRAY}(empty){RESET}")


# ---------------------------------------------------------------------------
# Type mode — keyboard output into active window
# ---------------------------------------------------------------------------

def run_type(device: int | None, vad_threshold: float, min_silence: int) -> None:
    """Like run_live() but types directly into the focused window."""
    from pynput.keyboard import Controller as KBController

    dev_idx, dev_name, dev_rate = find_input_device(device)
    kb = KBController()

    print(f"{GRAY}Device: {dev_name} ({dev_rate}Hz){RESET}")
    print(f"{GRAY}Loading models...{RESET}")

    transcribe_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    fix_tech_terms("test")
    print(f"{GREEN}Models ready{RESET}")
    print(f"\n{BOLD}Voice Command [TYPE MODE]{RESET}  {DIM}|{RESET}  {GREEN}LISTENING{RESET}  {DIM}|{RESET}  {GRAY}'stop listening' to pause  |  Ctrl+C to exit{RESET}")

    # countdown to give the user time to switch focus
    for i in range(3, 0, -1):
        print(f"\r{YELLOW}Switch to target app... {i}{RESET}", end="", flush=True)
        time.sleep(1)
    print(f"\r{GREEN}Listening — speak now!       {RESET}\n")

    # resampling setup
    up, down, callback_blocksize = make_resampler(dev_rate)

    # VAD setup
    vad_cfg = VADConfig(threshold=vad_threshold, min_silence_ms=min_silence)
    vad = VADState(vad_cfg)

    # audio queue
    audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)

    def audio_callback(indata, frames, time_info, status):
        try:
            audio_q.put_nowait(indata.copy())
        except queue.Full:
            pass

    # state
    buf = TextBuffer()
    paused = False
    running = True
    utterance_audio: list[np.ndarray] = []
    in_speech = False

    def on_sigint(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, on_sigint)

    input_stream = sd.InputStream(
        device=dev_idx,
        samplerate=dev_rate,
        channels=1,
        blocksize=callback_blocksize,
        callback=audio_callback,
        dtype="float32",
    )

    try:
        with input_stream:
            while running:
                try:
                    raw_audio = audio_q.get(timeout=0.05)
                except queue.Empty:
                    continue

                frame_16k = resample_frame(raw_audio, up, down)

                # --- paused: only listen for "start listening" ---
                if paused:
                    event, frames = vad.process_frame(frame_16k)

                    if event == "speech_start":
                        utterance_audio = list(frames)
                    elif event == "speech_continue":
                        utterance_audio.extend(frames)
                    elif event == "speech_end" and utterance_audio:
                        all_audio = np.concatenate(utterance_audio)
                        utterance_audio = []
                        text = transcribe_audio(all_audio)
                        if text:
                            cmd = try_command(text, buf)
                            if cmd.should_resume:
                                paused = False
                                print(f"  {GREEN}[LISTENING]{RESET}")
                    continue

                # --- active listening ---
                event, frames = vad.process_frame(frame_16k)

                if event == "speech_start":
                    utterance_audio = list(frames)
                    in_speech = True

                elif event == "speech_continue" and in_speech:
                    utterance_audio.extend(frames)

                elif event == "speech_end" and in_speech:
                    in_speech = False
                    if utterance_audio:
                        all_audio = np.concatenate(utterance_audio)
                        utterance_audio = []
                        full_text = transcribe_audio(all_audio)
                    else:
                        full_text = ""

                    if full_text:
                        old_text = buf.text
                        buf, message, cmd = process_utterance(full_text, buf)

                        if cmd and cmd.should_pause:
                            paused = True
                            print(f"  {YELLOW}[PAUSED]{RESET} say 'start listening' to resume")
                            continue

                        if cmd and cmd.should_exit:
                            print(f"\n{GREEN}Done{RESET}")
                            running = False
                            continue

                        if cmd and cmd.show_commands:
                            print(f"  {GRAY}Commands: {', '.join(COMMANDS.keys())}{RESET}")
                            continue

                        # type the diff into the active window
                        if buf.text != old_text:
                            if _is_own_terminal_focused():
                                print(f"  {YELLOW}> {message} (skipped — click target app first){RESET}")
                            else:
                                _type_diff(old_text, buf.text, kb)
                                print(f"  {CYAN}> {message}{RESET}")
                        else:
                            print(f"  {CYAN}> {message}{RESET}")

    except KeyboardInterrupt:
        pass

    print(f"\n{SEP}")
    print(f"{BOLD}Session ended ({buf.word_count} words typed){RESET}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Command — VAD-driven streaming voice input")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--file", type=str, default=None, help="Transcribe a file instead of live mic")
    parser.add_argument("--type", action="store_true", help="Type directly into active window (requires Accessibility permission)")
    parser.add_argument("--vad-threshold", type=float, default=0.45, help="VAD speech threshold (default: 0.45)")
    parser.add_argument("--min-silence", type=int, default=600, help="Min silence to end utterance in ms (default: 600)")
    args = parser.parse_args()

    if args.list_devices:
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                marker = " <-- default" if i == sd.default.device[0] else ""
                print(f"  {i}: {d['name']} (rate: {d['default_samplerate']}){marker}")
        return

    if args.type:
        run_type(args.device, args.vad_threshold, args.min_silence)
    elif args.file:
        run_file_test(args.file)
    else:
        run_live(args.device, args.vad_threshold, args.min_silence)


if __name__ == "__main__":
    main()
