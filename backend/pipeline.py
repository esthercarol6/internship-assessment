"""
Orchestrates the full pipeline:
  Input (text or audio) -> [STT] -> Summarise -> Translate -> TTS -> Output
"""

import os
import tempfile
import requests
from typing import Optional, Tuple

from backend.sunbird_client import (
    transcribe_audio,
    summarise_text,
    translate_text,
    synthesise_speech,
    LANGUAGE_CODES,
)

# 5-minute audio limit in seconds
MAX_AUDIO_DURATION_SECONDS = 300


def check_audio_duration(audio_path: str) -> float:
    """
    Returns the duration of an audio file in seconds using a lightweight approach.
    Raises ValueError if it exceeds the 5-minute limit.
    """
    try:
        import wave
        import contextlib

        if audio_path.lower().endswith(".wav"):
            with contextlib.closing(wave.open(audio_path, "r")) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                return duration

        # For non-WAV files, use mutagen if available, otherwise skip duration check
        try:
            from mutagen import File as MutagenFile
            audio = MutagenFile(audio_path)
            if audio is not None and audio.info is not None:
                return audio.info.length
        except ImportError:
            pass

        return 0.0  # Unknown duration — let the API handle it

    except Exception:
        return 0.0


def run_pipeline(
    input_mode: str,
    text_input: Optional[str],
    audio_path: Optional[str],
    target_language: str,
) -> Tuple[Optional[str], str, str, Optional[str], Optional[str]]:
    """
    Runs the full pipeline.

    Returns a tuple of:
      (transcript, summary, translated_summary, audio_url, error_message)

    On success, error_message is None.
    On failure, all result fields are None and error_message is set.
    """
    transcript = None
    summary = None
    translated_summary = None
    audio_url = None
    language_code = LANGUAGE_CODES.get(target_language, "lug")

    try:
        # Step 1: Get text — either directly from input or via STT
        if input_mode == "Text":
            if not text_input or not text_input.strip():
                return None, None, None, None, "Please enter some text before processing."
            working_text = text_input.strip()

        else:  # Audio mode
            if audio_path is None:
                return None, None, None, None, "Please upload an audio file before processing."

            # Check duration
            duration = check_audio_duration(audio_path)
            if duration > MAX_AUDIO_DURATION_SECONDS:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                return (
                    None, None, None, None,
                    f"Audio file is {minutes}m {seconds}s long. Maximum allowed duration is 5 minutes. "
                    f"Please upload a shorter file."
                )

            # Transcribe
            transcript = transcribe_audio(audio_path)
            if not transcript or not transcript.strip():
                return None, None, None, None, "Transcription returned empty text. Please try a different audio file."
            working_text = transcript

        # Step 2: Summarise
        summary = summarise_text(working_text)
        if not summary or not summary.strip():
            return transcript, None, None, None, "Summarisation returned an empty result. Please try again."

        # Step 3: Translate
        translated_summary = translate_text(summary, language_code)
        if not translated_summary or not translated_summary.strip():
            return transcript, summary, None, None, "Translation returned an empty result. Please try again."

        # Step 4: TTS
        audio_url = synthesise_speech(translated_summary, language_code)

        return transcript, summary, translated_summary, audio_url, None

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "unknown"
        if status == 401:
            msg = "Authentication failed. Please check your SUNBIRD_API_TOKEN."
        elif status == 429:
            msg = "Rate limit reached. Please wait a moment and try again."
        elif status == 422:
            msg = f"The API rejected the request (422 Unprocessable Entity). Check your input and try again."
        else:
            msg = f"Sunbird API error (HTTP {status}): {str(e)}"
        return transcript, summary, translated_summary, None, msg

    except requests.exceptions.ConnectionError:
        return transcript, summary, translated_summary, None, (
            "Could not connect to the Sunbird AI API. Please check your internet connection."
        )

    except requests.exceptions.Timeout:
        return transcript, summary, translated_summary, None, (
            "The request timed out. The Sunbird AI API may be busy — please try again."
        )

    except Exception as e:
        return transcript, summary, translated_summary, None, f"Unexpected error: {str(e)}"
