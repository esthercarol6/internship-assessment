"""
Thin wrapper around the Sunbird AI API endpoints.
Handles authentication and request/response for STT, TTS, summarisation, and translation.
"""

import os
import requests
from typing import Optional

BASE_URL = "https://api.sunbird.ai"

LANGUAGE_CODES = {
    "Luganda": "lug",
    "Runyankole": "nyn",
    "Ateso": "teo",
    "Lugbara": "lgg",
    "Acholi": "ach",
}

# Speaker IDs for TTS per language
TTS_SPEAKER_IDS = {
    "lug": 248,  # Luganda
    "nyn": 243,  # Runyankole
    "teo": 242,  # Ateso
    "lgg": 245,  # Lugbara
    "ach": 241,  # Acholi
}


def _get_token() -> str:
    token = os.getenv("SUNBIRD_API_TOKEN")
    if not token:
        raise ValueError("SUNBIRD_API_TOKEN is not set. Please add it to your .env file.")
    return token


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {_get_token()}"}


def transcribe_audio(audio_path: str) -> str:
    """
    Send an audio file to Sunbird STT and return the transcribed text.
    Endpoint: POST /tasks/stt
    """
    url = f"{BASE_URL}/tasks/stt"
    headers = _auth_headers()

    with open(audio_path, "rb") as f:
        files = {"audio": f}
        response = requests.post(url, files=files, headers=headers, timeout=120)

    response.raise_for_status()
    data = response.json()

    if "output" in data and isinstance(data["output"], dict):
        return data["output"].get("text") or data["output"].get("transcription") or str(data["output"])
    if "text" in data:
        return data["text"]
    if "transcription" in data:
        return data["transcription"]
    return str(data)


def summarise_text(text: str) -> str:
    """
    Summarise text using the Sunbird summarise endpoint.
    Endpoint: POST /tasks/summarise
    """
    url = f"{BASE_URL}/tasks/summarise"
    headers = {**_auth_headers(), "Content-Type": "application/json"}
    payload = {"text": text}

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "output" in data and isinstance(data["output"], dict):
        return (
            data["output"].get("summarized_text")
            or data["output"].get("summary")
            or data["output"].get("text")
            or data["output"].get("response")
            or str(data["output"])
        )
    if "summarized_text" in data:
        return data["summarized_text"]
    if "summary" in data:
        return data["summary"]
    if "response" in data:
        return data["response"]
    if "text" in data:
        return data["text"]
    return str(data)


def translate_text(text: str, target_language_code: str) -> str:
    """
    Translate text into a Ugandan local language using Sunflower simple inference.
    Endpoint: POST /tasks/sunflower_simple
    Uses form-encoded data (not JSON).
    Response: { "response": "...", "success": true, ... }
    """
    url = f"{BASE_URL}/tasks/sunflower_simple"
    headers = _auth_headers()

    language_names = {v: k for k, v in LANGUAGE_CODES.items()}
    lang_name = language_names.get(target_language_code, target_language_code)

    instruction = (
        f"Translate the following text into {lang_name}. "
        f"Return only the translated text, nothing else.\n\n{text}"
    )

    form_data = {
        "instruction": instruction,
        "model_type": "qwen",
        "temperature": "0.3",
    }

    response = requests.post(url, data=form_data, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()

    if "response" in data:
        return data["response"]
    if "output" in data and isinstance(data["output"], dict):
        return data["output"].get("response") or data["output"].get("text") or str(data["output"])
    return str(data)


def synthesise_speech(text: str, language_code: str) -> str:
    """
    Convert translated text to audio using Sunbird TTS.
    Returns a signed URL to the generated audio file.
    Endpoint: POST /tasks/modal/tts
    Response: { "success": true, "audio_url": "...", ... }
    """
    url = f"{BASE_URL}/tasks/modal/tts"
    headers = {**_auth_headers(), "Content-Type": "application/json"}

    speaker_id = TTS_SPEAKER_IDS.get(language_code, 248)
    payload = {
        "text": text,
        "speaker_id": speaker_id,
        "response_mode": "url",
    }

    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()

    if "audio_url" in data:
        return data["audio_url"]
    if "output" in data and isinstance(data["output"], dict):
        return data["output"].get("audio_url") or str(data["output"])
    raise ValueError(f"Unexpected TTS response shape: {data}")
