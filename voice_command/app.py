"""Application entry point: warmup → alt-screen → unified run loop → finalize.

This is the only module that talks to all the others. Everywhere else the
modules face inward — text knows nothing about audio, audio nothing about
the TUI, output nothing about settings — and `app.run` wires them together.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from voice_command import config
from voice_command.audio import AudioSource, SpeechEnd
from voice_command.config import Settings
from voice_command.output import OutputSink, TypeSink
from voice_command.text import COMMANDS, TextBuffer, is_resume_command, process_utterance, warmup_llm
from voice_command.tui import GRAY, GREEN, RED, RESET, Frame, Tui


# ---------------------------------------------------------------------------
# Hotkey dispatch
# ---------------------------------------------------------------------------

@dataclass
class _HotkeyCmd:
    quit: bool = False
    toggle_paused: bool = False
    message: str = ""


def _handle_hotkey(key: str, settings: Settings, tui: Tui) -> _HotkeyCmd:
    """Map a single key to a Settings mutation (+ optional sub-screen).

    Hot-swaps and live VAD updates aren't done here — they're picked up by the
    `run()` loop's resync step from settings on the next iteration. That keeps
    settings as the single source of truth.
    """
    if key in ("q", "Q", "\x03"):
        return _HotkeyCmd(quit=True)

    if key in ("p", "P", " "):
        return _HotkeyCmd(toggle_paused=True)

    if key in ("l", "L"):
        settings.llm_correction = not settings.llm_correction
        config.save(settings)
        return _HotkeyCmd(message=f"LLM {'on' if settings.llm_correction else 'off'} · saved")

    if key in ("d", "D"):
        chosen = tui.pick_device(AudioSource.list_devices(), settings.device)
        if chosen is None:
            return _HotkeyCmd()
        settings.device = chosen
        config.save(settings)
        return _HotkeyCmd(message=f"device → #{chosen}")

    if key in ("v", "V"):
        v = tui.pick_numeric("VAD threshold", settings.vad_threshold, 0.05, (0.10, 0.95), "{:.2f}")
        if v is None:
            return _HotkeyCmd()
        settings.vad_threshold = round(v, 2)
        config.save(settings)
        return _HotkeyCmd(message=f"VAD threshold → {settings.vad_threshold:.2f} · saved")

    if key in ("s", "S"):
        v = tui.pick_numeric(
            "min silence (ms)", float(settings.min_silence_ms),
            50.0, (100.0, 3000.0), "{:.0f}",
        )
        if v is None:
            return _HotkeyCmd()
        settings.min_silence_ms = int(v)
        config.save(settings)
        return _HotkeyCmd(message=f"min silence → {settings.min_silence_ms}ms · saved")

    if key == "?":
        tui.show_help(COMMANDS)
        return _HotkeyCmd()

    return _HotkeyCmd()


# ---------------------------------------------------------------------------
# The unified main loop
# ---------------------------------------------------------------------------

def _frame(settings: Settings, audio: AudioSource, sink: OutputSink, buf: TextBuffer,
           paused: bool, message: str) -> Frame:
    return Frame(
        settings=settings,
        device_name=audio.device_name,
        body=buf.text,
        status=sink.status(paused, audio.in_speech),
        message=message,
    )


def run(settings: Settings, sink: OutputSink, audio: AudioSource, tui: Tui) -> TextBuffer:
    """Drive the dictation loop until the user quits. Returns the final buffer."""
    buf = TextBuffer()
    paused = False
    message = ""
    message_set_at = 0.0

    def _set_message(text: str) -> tuple[str, float]:
        return text, time.monotonic()

    tui.render(_frame(settings, audio, sink, buf, paused, message))

    while True:
        # 1. Hotkeys may mutate `settings` and/or open a sub-screen
        for key in tui.poll_keys():
            cmd = _handle_hotkey(key, settings, tui)
            if cmd.quit:
                return buf
            if cmd.toggle_paused:
                paused = not paused
                # status icon shows the paused state; only resume needs a hint
                message, message_set_at = _set_message("" if paused else "▶ resumed")
            if cmd.message:
                message, message_set_at = _set_message(cmd.message)

        # 2. Reconcile audio with settings (handles D/V/S edits)
        if settings.device is not None and settings.device != audio.device_index:
            try:
                new_name = audio.set_device(settings.device)
                message, message_set_at = _set_message(f"device → {new_name}")
            except Exception as e:
                message, message_set_at = _set_message(f"device open failed: {e}")
                settings.device = audio.device_index
        audio.set_vad(threshold=settings.vad_threshold, min_silence_ms=settings.min_silence_ms)

        # 3. Audio events
        for evt in audio.poll(timeout=0.05):
            if not isinstance(evt, SpeechEnd):
                continue
            text = AudioSource.transcribe(evt.audio)
            if not text:
                continue
            if paused:
                # In paused mode we ONLY honor "start listening". Other voice
                # commands and plain text are dropped — matches the original
                # run_live/run_type behavior.
                if is_resume_command(text):
                    paused = False
                    message, message_set_at = _set_message("▶ resumed")
                continue
            old_text = buf.text
            res = process_utterance(text, buf, llm_enabled=settings.llm_correction)
            buf = res.buffer
            message, message_set_at = _set_message(res.message)
            addendum = sink.apply(old_text, buf.text)
            if addendum:
                joined = f"{message} ({addendum})" if message else addendum
                message, message_set_at = _set_message(joined)
            if res.exit:
                return buf
            if res.pause:
                paused = True
            if res.show_help:
                tui.show_help(COMMANDS)

        # 4. Inactivity auto-clear (buffer body + status message, same timeout)
        cleared = sink.maybe_auto_clear(buf, audio.in_speech)
        if cleared is not None:
            buf = cleared
        if (message
                and settings.inactivity_clear_seconds > 0
                and time.monotonic() - message_set_at > settings.inactivity_clear_seconds):
            message = ""

        # 5. Render
        tui.render(_frame(settings, audio, sink, buf, paused, message))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Handle --version / --help before any model loading happens.
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ("-V", "--version"):
            from voice_command import __version__
            print(f"voice-command {__version__}")
            return
        if arg in ("-h", "--help"):
            print(
                "voice-command — VAD-driven streaming voice dictation\n"
                "\n"
                "Usage: voice-cmd [-V|--version] [-h|--help]\n"
                "\n"
                "With no args, launches the dictation TUI. Hotkeys are shown\n"
                "in the header; press '?' inside the app for the full list."
            )
            return
        print(f"voice-cmd: unknown argument: {arg}", file=sys.stderr)
        sys.exit(2)

    settings = config.load()
    sink: OutputSink = TypeSink(settings.inactivity_clear_seconds)

    print(f"{GRAY}Loading models...{RESET}")
    AudioSource.warmup()
    if settings.llm_correction:
        warmup_llm()
    print(f"{GREEN}Models ready{RESET}")

    sink.before_start()

    audio = AudioSource(settings.device, settings.vad_threshold, settings.min_silence_ms)
    try:
        audio.start()
    except Exception as e:
        print(f"{RED}Audio device failed to open: {e}{RESET}")
        return

    # Persist whatever device we actually got (covers None → resolved index)
    settings.device = audio.device_index

    final_buf = TextBuffer()
    try:
        with Tui() as tui:
            try:
                final_buf = run(settings, sink, audio, tui)
            except KeyboardInterrupt:
                pass
    finally:
        audio.close()

    sink.finalize(final_buf)
