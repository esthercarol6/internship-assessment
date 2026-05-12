# Sunbird AI Language Pipeline

A Generative AI web application that takes text or audio input and runs it through a full NLP pipeline — transcription, summarisation, translation into a Ugandan local language, and text-to-speech synthesis — entirely powered by **Sunbird AI**.

\---

## Project Description

This app accepts either typed/pasted text or an uploaded audio file. If audio is provided, it is first transcribed to text using Sunbird's Speech-to-Text API. The text (or transcript) is then summarised using the Sunbird summarisation endpoint. The summary is translated into a user-selected Ugandan local language (Luganda, Runyankole, Ateso, Lugbara, or Acholi) via the Sunflower LLM. Finally, the translated summary is converted to a playable audio clip using Sunbird's Text-to-Speech API. All intermediate results are displayed in the UI.

\---

## Architecture Overview

```
User Input (Text or Audio)
        │
        ▼
\[If Audio] POST /tasks/stt          ← Speech-to-Text
        │
        ▼
POST /tasks/summarise               ← Summarisation
        │
        ▼
POST /tasks/sunflower\_simple        ← Translation (Sunflower LLM)
        │
        ▼
POST /tasks/tts                     ← Text-to-Speech
        │
        ▼
Output: Transcript + Summary + Translated Summary + Audio Player
```

### File structure

```
.
├── app.py                      # Gradio entry point
├── backend/
│   ├── sunbird\_client.py       # Thin wrapper around all Sunbird API endpoints
│   └── pipeline.py             # Orchestrates the full STT → summarise → translate → TTS flow
├── exercises/
│   └── basics.py               # Part 1: collatz and distinct\_numbers implementations
├── tests/
│   └── test\_basics.py          # Pytest test suite for Part 1
├── constants.py                # Large collatz test constants
├── requirements.txt
├── .env.example                # Template for environment variables
└── .gitignore
```

\---

## Local Setup

### 1\. Clone the repository

```bash
git clone https://github.com/<your-username>/internship-assessment.git
cd internship-assessment
```

### 2\. Create and activate a virtual environment

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows
venv\\Scripts\\activate.bat
```

### 3\. Install dependencies

```bash
pip install -r requirements.txt
```

### 4\. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and replace `your\_token\_here` with your actual Sunbird AI API token:

```
SUNBIRD\_API\_TOKEN=eyJ...your\_actual\_token...
```

Get a token by signing up at [https://api.sunbird.ai/](https://api.sunbird.ai/).

### 5\. Run the app

```bash
python app.py
```

Gradio will print a local URL (e.g. `http://127.0.0.1:7860`). Open it in your browser.

### 6\. Run the Part 1 tests (optional)

```bash
pytest
```

All 5 tests should pass.

\---

## Environment Variables

|Variable|Description|
|-|-|
|`SUNBIRD\_API\_TOKEN`|Your Sunbird AI API bearer token. Required for all API calls. Obtain from [https://api.sunbird.ai/](https://api.sunbird.ai/).|

\---

## Usage Walkthrough

### Text input mode

1. Open the app in your browser.
2. Ensure **Input Mode** is set to **Text** (default).
3. Paste or type any English text into the **Text Input** box.
4. Select a target language from the **Target Language** dropdown (e.g. Luganda).
5. Click **Run Pipeline ▶**.
6. The right panel fills in: **Summary** (condensed version of your text), **Translated Summary** (in the chosen language), and an **audio player** you can press play on immediately.

### Audio input mode

1. Click **Audio** under **Input Mode**.
2. Upload an MP3, WAV, OGG, or M4A file — maximum 5 minutes.
3. Select your target language.
4. Click **Run Pipeline ▶**.
5. The **Transcript** field shows the STT output. The remaining fields fill in exactly as above.

\---

## Deployed Link

> \*\*\[https://huggingface.co/spaces/Nabirye/Sunbird-pipeline](https://huggingface.co/spaces/Nabirye/Sunbird-pipeline)\*\*>
> \*(Replace this URL with your actual Hugging Face Space link after deployment.)\*

### Deploying to Hugging Face Spaces

```bash
# Add the Space remote
git remote add space https://huggingface.co/spaces/<your-username>/sunbird-pipeline

# Push
git push space main
```

In your Space settings → **Variables and secrets** → add:

|Name|Value|
|-|-|
|`SUNBIRD\_API\_TOKEN`|your actual token|

Hugging Face will build and serve the app automatically.

\---

## Known Limitations

* **Audio duration cap**: Files longer than 5 minutes are rejected with a clear error. The Sunbird STT API processes up to 10 minutes, but this app enforces a stricter 5-minute limit per the assessment requirements.
* **Summarisation language**: The `/tasks/summarise` endpoint officially supports English and Luganda. Input in other languages may produce lower-quality summaries.
* **Translation quality**: The Sunflower LLM performs best on shorter, well-structured English summaries. Very long or complex text may produce imperfect translations.
* **TTS audio**: The generated audio URL from Sunbird is a temporary signed URL. The app downloads it immediately; however, if the download fails, an error is shown instead of the audio player.
* **Supported audio formats for duration check**: WAV duration checking is native. For MP3/M4A/OGG, duration checking requires the `mutagen` package (included in `requirements.txt`). If `mutagen` is unavailable and a non-WAV file is uploaded, the app skips the client-side duration check and relies on the Sunbird API's own limits.

