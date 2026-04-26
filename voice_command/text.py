"""Text buffer, voice-command parsing, and utterance post-processing.

Public API:
    TextBuffer            — immutable word list with undo
    UtteranceResult       — outcome of processing one transcribed utterance
    process_utterance()   — the only entry point callers need
    copy_to_clipboard()   — used by both voice commands and BufferSink.finalize

Hidden: command tables, contraction regex, sentence splitting, leading/
trailing command extraction, LLM model lifecycle, the recursive piece parser.
"""

from __future__ import annotations

import re
import subprocess
from collections import deque
from dataclasses import dataclass, field


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

    def _append_raw(self, token: str) -> "TextBuffer":
        """Append a literal token (e.g. '\\n') without splitting on whitespace."""
        self._snapshot()
        return TextBuffer(_words=list(self._words) + [token], _history=self._history)

    @property
    def text(self) -> str:
        return " ".join(self._words)

    @property
    def word_count(self) -> int:
        return len(self._words)


# ---------------------------------------------------------------------------
# Voice command tables (private)
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

_PUNCTUATION_MAP = {
    "period": ".", "full stop": ".", "comma": ",",
    "question mark": "?", "exclamation mark": "!", "exclamation point": "!",
    "colon": ":", "semicolon": ";", "dash": "—", "hyphen": "-",
    "ellipsis": "...",
}

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twenty": 20,
}

_COMMAND_TRIGGERS = {
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


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class UtteranceResult:
    buffer: TextBuffer
    message: str = ""
    pause: bool = False
    resume: bool = False
    exit: bool = False
    show_help: bool = False
    copied: bool = False


# ---------------------------------------------------------------------------
# Internal command-result type and dispatcher
# ---------------------------------------------------------------------------

@dataclass
class _CommandResult:
    handled: bool
    buffer: TextBuffer
    message: str = ""
    should_pause: bool = False
    should_resume: bool = False
    should_exit: bool = False
    show_commands: bool = False
    copied: bool = False


def _extract_number(text: str) -> int | None:
    for word in text.lower().split():
        if word.isdigit():
            return int(word)
        if word in _NUMBER_WORDS:
            return _NUMBER_WORDS[word]
    return None


def copy_to_clipboard(text: str) -> None:
    """Write `text` to the macOS clipboard via pbcopy. Swallows failures."""
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    except Exception:
        pass


_NEWLINE_PADDING_RE = re.compile(r" *(\n+) *")


def normalize_buffer_text(text: str) -> str:
    """Strip spaces around runs of '\\n'.

    `_append_raw('\\n')`/`_append_raw('\\n\\n')` enter the buffer as standalone
    word tokens, so `TextBuffer.text`'s `' '.join` produces ` ... word \\n word ...`.
    Both display (TUI) and keystroke output (TypeSink) need the surrounding
    spaces removed.
    """
    return _NEWLINE_PADDING_RE.sub(r"\1", text)


def is_resume_command(text: str) -> bool:
    """True iff `text` is a 'resume listening' phrase. Used by paused mode."""
    t = text.lower().strip().rstrip(".,!?;:")
    return t in ("start listening", "resume", "start")


def _try_command(text: str, buf: TextBuffer) -> _CommandResult:
    """Match `text` against the voice-command table; act on the buffer."""
    t = text.lower().strip().rstrip(".,!?;:")

    if t in ("show commands", "show command", "list commands", "help"):
        return _CommandResult(True, buf, show_commands=True)

    if t in ("undo", "undo that"):
        return _CommandResult(True, buf.undo(), message="Undone")

    if t in ("scratch that", "scratch this", "delete that"):
        return _CommandResult(True, buf.delete_last_n(5), message="Scratched")

    if t in ("clear all", "clear everything", "erase all"):
        return _CommandResult(True, buf.clear(), message="Cleared")

    if "delete last" in t and ("word" in t or "words" in t):
        n = _extract_number(t)
        if n:
            return _CommandResult(True, buf.delete_last_n(n), message=f"Deleted {n} words")

    if t in ("new line", "newline"):
        return _CommandResult(True, buf._append_raw("\n"), message="New line")

    if t in ("new paragraph", "new para"):
        return _CommandResult(True, buf._append_raw("\n\n"), message="New paragraph")

    if t in ("stop listening", "pause", "stop"):
        # status icon shows ⏸ paused; no message needed
        return _CommandResult(True, buf, should_pause=True)

    if t in ("start listening", "resume", "start"):
        return _CommandResult(True, buf, should_resume=True, message="▶ resumed")

    if t in ("copy all", "copy text", "copy"):
        copy_to_clipboard(buf.text)
        return _CommandResult(True, buf, copied=True, message="Copied to clipboard")

    if t in ("done", "finish", "exit", "quit"):
        copy_to_clipboard(buf.text)
        return _CommandResult(True, buf, should_exit=True, copied=True, message="Done")

    for trigger, punct in _PUNCTUATION_MAP.items():
        if t == trigger:
            return _CommandResult(True, buf.append_punctuation(punct), message=f"+ {trigger}")

    return _CommandResult(False, buf)


# ---------------------------------------------------------------------------
# Sentence + command splitting (private)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks on '. ' boundaries.

    Falls back to comma-splitting when sentence-splitting produces only
    one piece but it contains multiple command words (common with Whisper
    output like "colon, new line, milk, comma, bread, period").
    """
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1 and ", " in text:
        lower = text.lower()
        cmd_count = sum(1 for trigger in _COMMAND_TRIGGERS if f", {trigger}" in lower or lower.startswith(trigger))
        if cmd_count >= 2:
            parts = [p.strip() for p in text.split(", ") if p.strip()]

    return parts


def _clean_command_text(text: str) -> str:
    return text.strip().strip(".,!?;:").strip()


def _extract_leading_command(text: str) -> tuple[str | None, str]:
    """Find a command at the start of text. Returns (head_text, remaining)."""
    words = text.split()
    for n in range(min(6, len(words)), 0, -1):
        head = " ".join(words[:n])
        cleaned = _clean_command_text(head).lower()
        if cleaned in _COMMAND_TRIGGERS or ("delete last" in cleaned and ("word" in cleaned or "words" in cleaned)):
            after = " ".join(words[n:]).strip().lstrip(".,!?;:").strip()
            return head, after
    return None, text


def _extract_trailing_command(text: str) -> tuple[str, str | None]:
    """Find a command at the tail of text. Returns (before, command_text)."""
    words = text.split()
    for n in range(1, min(6, len(words) + 1)):
        tail = " ".join(words[-n:]).lower().rstrip(".,!?;:")
        if tail in _COMMAND_TRIGGERS or ("delete last" in tail and ("word" in tail or "words" in tail)):
            before = " ".join(words[:-n]).strip()
            return before, " ".join(words[-n:])
    return text, None


# ---------------------------------------------------------------------------
# Contractions (private)
# ---------------------------------------------------------------------------

_CONTRACTION_FIXES = [
    (r"\b(\w+)\s+'(m|s|t|re|ve|ll|d)\b", r"\1'\2"),
    (r"\b(\w+)\s+n't\b", r"\1n't"),
]


def _fix_contractions(text: str) -> str:
    result = text
    for pattern, replacement in _CONTRACTION_FIXES:
        result = re.sub(pattern, replacement, result)
    return result


# ---------------------------------------------------------------------------
# LLM tech-term correction (private; lazy-loaded on first use)
# ---------------------------------------------------------------------------

_LLM_MODEL_ID = "mlx-community/Qwen3-1.7B-4bit"

_llm_model = None
_llm_tokenizer = None
_llm_sampler = None

_LLM_PROMPT_TEMPLATE = (
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
    cleaned = text.strip().lower().rstrip(".,!?;:")
    if cleaned in _COMMAND_TRIGGERS:
        return True
    if "delete last" in cleaned and ("word" in cleaned or "words" in cleaned):
        return True
    if cleaned in _PUNCTUATION_MAP:
        return True
    return False


def _fix_tech_terms(text: str) -> str:
    """Run the local LLM to fix tech-term casing/misrecognitions."""
    global _llm_model, _llm_tokenizer, _llm_sampler

    if not text.strip():
        return text

    # LLM hallucinates on short commands — skip
    if _looks_like_command(text):
        return text

    if _llm_model is None:
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        _llm_model, _llm_tokenizer = load(_LLM_MODEL_ID)
        _llm_sampler = make_sampler(temp=0.0)

    from mlx_lm import generate

    prompt = _LLM_PROMPT_TEMPLATE.format(text=text)
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


def warmup_llm() -> None:
    """Force the LLM to load now (used at app startup)."""
    _fix_tech_terms("test")


# ---------------------------------------------------------------------------
# Per-piece processor (private, recursive)
# ---------------------------------------------------------------------------

def _process_single_piece(text: str, buf: TextBuffer) -> tuple[TextBuffer, list[str], _CommandResult | None]:
    """Process one text fragment.

    Tries: whole text as command → leading command → trailing command → plain.
    Recurses into the leftover after stripping a leading or trailing command.
    """
    cmd = _try_command(text, buf)
    if cmd.handled:
        return cmd.buffer, [cmd.message], cmd

    # leading command: "New line, second sentence here"
    lead_cmd, remaining = _extract_leading_command(text)
    if lead_cmd:
        cmd = _try_command(lead_cmd, buf)
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

    # trailing command: "some text period"
    before, cmd_text = _extract_trailing_command(text)
    if cmd_text:
        msgs: list[str] = []
        last_cmd: _CommandResult | None = None
        if before:
            buf, sub_msgs, sub_cmd = _process_single_piece(before, buf)
            msgs.extend(sub_msgs)
            if sub_cmd:
                last_cmd = sub_cmd
        cmd = _try_command(cmd_text, buf)
        if cmd.handled:
            msgs.append(cmd.message)
            return cmd.buffer, msgs, cmd

    # no command — append as text
    buf = buf.append_text(text)
    return buf, [f'+ "{text}"'], None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_utterance(
    text: str,
    buf: TextBuffer,
    *,
    llm_enabled: bool,
) -> UtteranceResult:
    """Normalize, detect commands, and append to `buf`.

    Splits longer utterances on sentence boundaries and checks each piece
    for commands — both leading and trailing. Handles mid-text commands like
    "First sentence. Period. New line. Second sentence."
    """
    text = _fix_contractions(text)
    if llm_enabled:
        text = _fix_tech_terms(text)

    # try whole text as a single command first
    cmd = _try_command(text, buf)
    if cmd.handled:
        return _result(cmd, message=cmd.message)

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        new_buf, msgs, last_cmd = _process_single_piece(text, buf)
        return _result(last_cmd, buffer=new_buf, message=" → ".join(msgs))

    messages: list[str] = []
    last_cmd: _CommandResult | None = None
    for sentence in sentences:
        buf, msgs, cmd = _process_single_piece(sentence, buf)
        messages.extend(msgs)
        if cmd:
            last_cmd = cmd
            if cmd.should_exit or cmd.should_pause:
                break

    return _result(last_cmd, buffer=buf, message=" → ".join(messages))


def _result(
    cmd: _CommandResult | None,
    *,
    buffer: TextBuffer | None = None,
    message: str = "",
) -> UtteranceResult:
    """Project an internal _CommandResult onto the public UtteranceResult."""
    if cmd is None:
        return UtteranceResult(buffer=buffer if buffer is not None else TextBuffer(), message=message)
    return UtteranceResult(
        buffer=buffer if buffer is not None else cmd.buffer,
        message=message,
        pause=cmd.should_pause,
        resume=cmd.should_resume,
        exit=cmd.should_exit,
        show_help=cmd.show_commands,
        copied=cmd.copied,
    )
