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
    "lug": 248,  # Luganda - Female
    "nyn": 243,  # Runyankole - Female
    "teo": 242,  # Ateso - Female
    "lgg": 245,  # Lugbara - Female
    "ach": 241,  # Acholi - Female
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
    return data["output"]["text"]


def summarise_text(text: str) -> str:
    """
    Summarise text using the Sunbird /tasks/summarise endpoint.
    Endpoint: POST /tasks/summarise
    """
    url = f"{BASE_URL}/tasks/summarise"
    headers = {**_auth_headers(), "Content-Type": "application/json"}
    payload = {"text": text}

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["output"]["summary"]


def translate_text(text: str, target_language_code: str) -> str:
    """
    Translate text into a Ugandan local language using Sunflower simple inference.
    Endpoint: POST /tasks/sunflower_simple
    """
    url = f"{BASE_URL}/tasks/sunflower_simple"
    headers = {**_auth_headers(), "Content-Type": "application/json"}

    language_names = {v: k for k, v in LANGUAGE_CODES.items()}
    lang_name = language_names.get(target_language_code, target_language_code)

    instruction = (
        f"Translate the following text into {lang_name}. "
        f"Return only the translated text, nothing else.\n\n{text}"
    )
    payload = {"instruction": instruction}

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()

    # Simple inference returns output.response or output.content depending on model
    output = data.get("output", {})
    if isinstance(output, dict):
        return output.get("response") or output.get("content") or str(output)
    return str(output)


def synthesise_speech(text: str, language_code: str) -> str:
    """
    Convert translated text to audio using Sunbird TTS.
    Returns a URL to the generated audio file.
    Endpoint: POST /tasks/tts
    """
    url = f"{BASE_URL}/tasks/tts"
    headers = {**_auth_headers(), "Content-Type": "application/json"}

    speaker_id = TTS_SPEAKER_IDS.get(language_code, 248)
    payload = {"text": text, "speaker_id": speaker_id}

    response = requests.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["output"]["audio_url"]
