"""
Orchestrates the full pipeline:
  Input (text or audio) -> [STT] -> Summarise -> Translate -> TTS -> Output
"""

import os
import time
import requests
from typing import Optional, Tuple

from backend.sunbird_client import (
    transcribe_audio,
    summarise_text,
    translate_text,
    synthesise_speech,
    LANGUAGE_CODES,
)

MAX_AUDIO_DURATION_SECONDS = 300


def check_audio_duration(audio_path: str) -> float:
    try:
        import wave, contextlib
        if audio_path.lower().endswith(".wav"):
            with contextlib.closing(wave.open(audio_path, "r")) as f:
                return f.getnframes() / float(f.getframerate())
        try:
            from mutagen import File as MutagenFile
            audio = MutagenFile(audio_path)
            if audio is not None and audio.info is not None:
                return audio.info.length
        except ImportError:
            pass
    except Exception:
        pass
    return 0.0


def _retry(fn, retries=2, delay=3):
    """Call fn; on timeout, wait and retry up to `retries` times."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(delay)
                continue
            raise


def run_pipeline(
    input_mode: str,
    text_input: Optional[str],
    audio_path: Optional[str],
    target_language: str,
) -> Tuple[Optional[str], str, str, Optional[str], Optional[str]]:

    transcript = None
    language_code = LANGUAGE_CODES.get(target_language, "lug")

    try:
        # Step 1: Get working text
        if input_mode == "Text":
            if not text_input or not text_input.strip():
                return None, None, None, None, "Please enter some text before processing."
            working_text = text_input.strip()
        else:
            if audio_path is None:
                return None, None, None, None, "Please upload an audio file before processing."
            duration = check_audio_duration(audio_path)
            if duration > MAX_AUDIO_DURATION_SECONDS:
                m, s = int(duration // 60), int(duration % 60)
                return None, None, None, None, f"Audio is {m}m {s}s — maximum is 5 minutes."
            transcript = _retry(lambda: transcribe_audio(audio_path))
            if not transcript or not transcript.strip():
                return None, None, None, None, "Transcription returned empty text."
            working_text = transcript

        # Step 2: Summarise
        summary = _retry(lambda: summarise_text(working_text))
        if not summary or not summary.strip():
            return transcript, None, None, None, "Summarisation returned empty result."

        # Step 3: Translate
        translated_summary = _retry(lambda: translate_text(summary, language_code))
        if not translated_summary or not translated_summary.strip():
            return transcript, summary, None, None, "Translation returned empty result."

        # Step 4: TTS
        audio_url = _retry(lambda: synthesise_speech(translated_summary, language_code))

        # Step 5: Download audio
        local_audio_path = None
        if audio_url:
            try:
                resp = requests.get(audio_url, timeout=60)
                resp.raise_for_status()
                import tempfile
                suffix = ".wav" if "wav" in audio_url else ".mp3"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(resp.content)
                tmp.flush()
                local_audio_path = tmp.name
            except Exception as e:
                return transcript, summary, translated_summary, None, f"Audio generated but download failed: {str(e)}"

        return transcript, summary, translated_summary, local_audio_path, None

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        if status == 401:
            msg = "Authentication failed. Check your SUNBIRD_API_TOKEN."
        elif status == 429:
            msg = "Rate limit reached. Please wait and try again."
        elif status == 422:
            msg = "The API rejected the request (422). Check your input."
        else:
            msg = f"Sunbird API error (HTTP {status}): {str(e)}"
        return transcript, None, None, None, msg

    except requests.exceptions.Timeout:
        return transcript, None, None, None, "Request timed out after retrying. Sunbird API is busy — please try again in a moment."

    except requests.exceptions.ConnectionError:
        return transcript, None, None, None, "Could not connect to Sunbird AI API. Check your internet connection."

    except Exception as e:
        return transcript, None, None, None, f"Unexpected error: {str(e)}"
