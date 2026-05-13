"""
Thin wrapper around the Sunbird AI API endpoints.
"""

import os
import requests

BASE_URL = "https://api.sunbird.ai"

LANGUAGE_CODES = {
    "Luganda": "lug",
    "Runyankole": "nyn",
    "Ateso": "teo",
    "Lugbara": "lgg",
    "Acholi": "ach",
}

TTS_SPEAKER_IDS = {
    "lug": 248,
    "nyn": 243,
    "teo": 242,
    "lgg": 245,
    "ach": 241,
}


def _get_token() -> str:
    token = os.getenv("SUNBIRD_API_TOKEN")
    if not token:
        raise ValueError("SUNBIRD_API_TOKEN is not set.")
    return token


def _auth_headers() -> dict:
    return {"Authorization": f"Bearer {_get_token()}"}


def transcribe_audio(audio_path: str) -> str:
    """POST /tasks/stt"""
    url = f"{BASE_URL}/tasks/stt"
    with open(audio_path, "rb") as f:
        response = requests.post(url, files={"audio": f}, headers=_auth_headers(), timeout=180)
    response.raise_for_status()
    data = response.json()
    if "output" in data and isinstance(data["output"], dict):
        return data["output"].get("text") or data["output"].get("transcription") or str(data["output"])
    return data.get("text") or data.get("transcription") or str(data)


def summarise_text(text: str) -> str:
    """POST /tasks/summarise"""
    url = f"{BASE_URL}/tasks/summarise"
    headers = {**_auth_headers(), "Content-Type": "application/json"}
    response = requests.post(url, json={"text": text}, headers=headers, timeout=120)
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
    return (
        data.get("summarized_text")
        or data.get("summary")
        or data.get("response")
        or data.get("text")
        or str(data)
    )


def translate_text(text: str, target_language_code: str) -> str:
    """
    POST /tasks/sunflower_inference
    Uses the full chat inference endpoint (JSON, has built-in retry/backoff).
    Response key: data["content"]
    """
    url = f"{BASE_URL}/tasks/sunflower_inference"
    headers = {**_auth_headers(), "Content-Type": "application/json"}

    language_names = {v: k for k, v in LANGUAGE_CODES.items()}
    lang_name = language_names.get(target_language_code, target_language_code)

    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Translate the following text into {lang_name}. "
                    f"Return only the translated text, nothing else.\n\n{text}"
                )
            }
        ],
        "model_type": "qwen",
        "temperature": 0.3,
        "stream": False,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    data = response.json()

    # Response shape: { "content": "...", "model_type": "...", ... }
    if "content" in data:
        return data["content"]
    if "response" in data:
        return data["response"]
    if "output" in data and isinstance(data["output"], dict):
        return data["output"].get("content") or data["output"].get("response") or str(data["output"])
    return str(data)


def synthesise_speech(text: str, language_code: str) -> str:
    """POST /tasks/modal/tts"""
    url = f"{BASE_URL}/tasks/modal/tts"
    headers = {**_auth_headers(), "Content-Type": "application/json"}
    speaker_id = TTS_SPEAKER_IDS.get(language_code, 248)
    payload = {"text": text, "speaker_id": speaker_id, "response_mode": "url"}
    response = requests.post(url, json=payload, headers=headers, timeout=180)
    response.raise_for_status()
    data = response.json()
    if "audio_url" in data:
        return data["audio_url"]
    if "output" in data and isinstance(data["output"], dict):
        return data["output"].get("audio_url") or str(data["output"])
    raise ValueError(f"Unexpected TTS response: {data}")
