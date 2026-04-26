"""
sound_output.py
---------------
Sound output module for ISL Recognition System.
Speaks the recognised letter when confidence is above threshold.

Features:
  - Pi compatible (uses espeak — lightweight, no internet needed)
  - Laptop compatible (uses pyttsx3 as fallback)
  - Cooldown timer — prevents same letter being spoken repeatedly
  - Confidence gate — only speaks above set threshold
  - Non-blocking — runs in background thread so it doesn't freeze video
  - Word building mode — accumulates letters into words and speaks them

Pi installation:
  sudo apt install espeak -y
  pip3 install pyttsx3

Laptop installation:
  pip install pyttsx3
  Windows: pip install pywin32  (for SAPI5 voice)
  Mac:     built-in 'say' command used automatically
"""

import threading
import time
import queue
import os
import platform
import subprocess
from collections import deque


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Minimum confidence to trigger speech (0.0 to 1.0)
SPEECH_CONFIDENCE_THRESHOLD = 0.75

# Seconds to wait before speaking the same letter again
SAME_LETTER_COOLDOWN = 2.5

# Seconds to wait before speaking any letter again
ANY_LETTER_COOLDOWN  = 1.0

# Word building — accumulate letters into words
# When user pauses for WORD_PAUSE seconds, speaks the whole word
WORD_BUILD_MODE  = False   # set True to enable word building
WORD_PAUSE       = 2.0     # seconds of silence = word complete
MAX_WORD_LENGTH  = 10      # max letters before auto-speak

# Speech rate (words per minute)
# Lower = slower and clearer, good for demo
SPEECH_RATE = 150

# Voice engine: 'auto', 'espeak', 'pyttsx3', 'say' (mac)
VOICE_ENGINE = 'auto'


# ─────────────────────────────────────────────
# VOICE ENGINE DETECTION
# ─────────────────────────────────────────────

def detect_engine():
    """Detect best available speech engine for this platform."""
    system = platform.system()

    # Try espeak first — best for Pi
    try:
        result = subprocess.run(
            ['espeak', '--version'],
            capture_output=True, timeout=2
        )
        if result.returncode == 0:
            return 'espeak'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try pyttsx3
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.stop()
        return 'pyttsx3'
    except Exception:
        pass

    # Mac fallback
    if system == 'Darwin':
        return 'say'

    # Linux fallback — festival
    try:
        result = subprocess.run(
            ['festival', '--version'],
            capture_output=True, timeout=2
        )
        if result.returncode == 0:
            return 'festival'
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


# ─────────────────────────────────────────────
# SPEECH ENGINES
# ─────────────────────────────────────────────

def speak_espeak(text, rate=SPEECH_RATE):
    """Use espeak — lightweight, works offline, best for Pi."""
    try:
        subprocess.run(
            ['espeak', '-r', str(rate), '-v', 'en', text],
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f'espeak error: {e}')


def speak_pyttsx3(text, engine_ref):
    """Use pyttsx3 — cross platform."""
    try:
        engine_ref.say(text)
        engine_ref.runAndWait()
    except Exception as e:
        print(f'pyttsx3 error: {e}')


def speak_say(text):
    """Mac built-in say command."""
    try:
        subprocess.run(
            ['say', '-r', str(SPEECH_RATE), text],
            timeout=5,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f'say error: {e}')


def speak_festival(text):
    """Festival TTS — Linux fallback."""
    try:
        proc = subprocess.Popen(
            ['festival', '--tts'],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        proc.communicate(input=text.encode(), timeout=5)
    except Exception as e:
        print(f'festival error: {e}')


# ─────────────────────────────────────────────
# MAIN SOUND OUTPUT CLASS
# ─────────────────────────────────────────────

class SoundOutput:
    """
    Non-blocking sound output for ISL recognition.

    Usage:
        sound = SoundOutput()
        sound.start()

        # In your main loop:
        sound.notify(label='A', confidence=0.92)

        sound.stop()
    """

    def __init__(self,
                 confidence_threshold=SPEECH_CONFIDENCE_THRESHOLD,
                 same_letter_cooldown=SAME_LETTER_COOLDOWN,
                 any_letter_cooldown=ANY_LETTER_COOLDOWN,
                 word_build_mode=WORD_BUILD_MODE):

        self.confidence_threshold = confidence_threshold
        self.same_letter_cooldown = same_letter_cooldown
        self.any_letter_cooldown  = any_letter_cooldown
        self.word_build_mode      = word_build_mode

        self._queue         = queue.Queue(maxsize=5)
        self._thread        = None
        self._running       = False
        self._last_spoken   = None
        self._last_spoken_t = 0
        self._any_last_t    = 0

        # Word building state
        self._word_buffer   = []
        self._last_letter_t = 0

        # Detect and init engine
        engine_name = VOICE_ENGINE if VOICE_ENGINE != 'auto' \
                      else detect_engine()

        self._engine_name   = engine_name
        self._pyttsx3_engine = None

        if engine_name == 'pyttsx3':
            try:
                import pyttsx3
                self._pyttsx3_engine = pyttsx3.init()
                self._pyttsx3_engine.setProperty('rate', SPEECH_RATE)
            except Exception as e:
                print(f'pyttsx3 init error: {e}')
                self._engine_name = None

        if engine_name:
            print(f'🔊  Sound engine: {engine_name}')
        else:
            print('⚠️   No speech engine found.')
            print('     Pi:     sudo apt install espeak -y')
            print('     Laptop: pip install pyttsx3')

    def start(self):
        """Start background speech thread."""
        self._running = True
        self._thread  = threading.Thread(
            target=self._worker,
            daemon=True,
            name='SoundWorker'
        )
        self._thread.start()

    def stop(self):
        """Stop background speech thread."""
        self._running = False
        self._queue.put(None)   # sentinel to unblock worker
        if self._thread:
            self._thread.join(timeout=3)

    def notify(self, label: str, confidence: float):
        """
        Call this every frame with the current prediction.
        Speech is triggered only when:
          1. Label is not '?'
          2. Confidence >= threshold
          3. Cooldown has passed
        """
        if label == '?' or label is None:
            # Silence detected — check if word is ready to speak
            if self.word_build_mode and self._word_buffer:
                silence = time.time() - self._last_letter_t
                if silence >= WORD_PAUSE:
                    word = ''.join(self._word_buffer)
                    self._word_buffer = []
                    self._enqueue(word)
            return

        if confidence < self.confidence_threshold:
            return

        now = time.time()

        if self.word_build_mode:
            # Word building mode — accumulate letters
            if (not self._word_buffer or
                    label != self._word_buffer[-1]):
                self._word_buffer.append(label)
                self._last_letter_t = now

            if len(self._word_buffer) >= MAX_WORD_LENGTH:
                word = ''.join(self._word_buffer)
                self._word_buffer = []
                self._enqueue(word)

        else:
            # Letter mode — speak each letter individually
            same_ok = (label != self._last_spoken or
                       now - self._last_spoken_t >= self.same_letter_cooldown)
            any_ok  = (now - self._any_last_t >= self.any_letter_cooldown)

            if same_ok and any_ok:
                self._last_spoken   = label
                self._last_spoken_t = now
                self._any_last_t    = now
                self._enqueue(label)

    def _enqueue(self, text: str):
        """Add text to speech queue — drop if full."""
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass   # drop if queue is full — don't block video

    def _speak(self, text: str):
        """Speak text using the detected engine."""
        if not self._engine_name:
            return

        if self._engine_name == 'espeak':
            speak_espeak(text)
        elif self._engine_name == 'pyttsx3':
            speak_pyttsx3(text, self._pyttsx3_engine)
        elif self._engine_name == 'say':
            speak_say(text)
        elif self._engine_name == 'festival':
            speak_festival(text)

    def _worker(self):
        """Background worker — processes speech queue."""
        while self._running:
            try:
                text = self._queue.get(timeout=1.0)
                if text is None:
                    break
                self._speak(text)
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f'Sound worker error: {e}')

    @property
    def current_word(self):
        """Return currently accumulated word (for display)."""
        return ''.join(self._word_buffer)

    def clear_word(self):
        """Manually clear word buffer."""
        self._word_buffer = []