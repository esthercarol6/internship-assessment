"""
Sunbird AI Language Pipeline — Gradio entry point.
Pipeline: Input (text or audio) → [STT] → Summarise → Translate → TTS → Output
"""

import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from backend.pipeline import run_pipeline
from backend.sunbird_client import LANGUAGE_CODES

LANGUAGES = list(LANGUAGE_CODES.keys())

CUSTOM_CSS = """
.gradio-container { max-width: 900px !important; margin: 0 auto !important; }
#title-block { text-align: center; padding: 24px 0 8px 0; }
#title-block h1 { font-size: 2rem; font-weight: 700; color: #DC7828; margin-bottom: 4px; }
#title-block p { color: #555; font-size: 0.95rem; }
.pipeline-arrow { text-align: center; font-size: 1.2rem; color: #DC7828; padding: 4px 0; }
.result-box textarea { background: #f9f9f9 !important; border-left: 3px solid #DC7828 !important; }
.error-box textarea { background: #fff3f3 !important; border-left: 3px solid #e53e3e !important; color: #c53030 !important; }
.submit-btn { background: #DC7828 !important; color: white !important; font-weight: 600 !important; }
"""


def process(input_mode, text_input, audio_file, target_language):
    transcript, summary, translated_summary, local_audio, error = run_pipeline(
        input_mode=input_mode,
        text_input=text_input,
        audio_path=audio_file,
        target_language=target_language,
    )
    error_msg = f"⚠️  {error}" if error else ""
    return (
        transcript or "",
        summary or "",
        translated_summary or "",
        local_audio,
        error_msg,
    )


def toggle_input_mode(mode):
    if mode == "Text":
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)


with gr.Blocks(css=CUSTOM_CSS, title="Sunbird AI Language Pipeline") as demo:

    gr.HTML("""
    <div id="title-block">
        <h1>🌻 Sunbird AI Language Pipeline</h1>
        <p>Transcribe → Summarise → Translate into a Ugandan language → Hear it spoken</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_mode = gr.Radio(
                choices=["Text", "Audio"], value="Text", label="Input Mode",
                info="Choose whether to type/paste text or upload an audio file.",
            )
            text_input = gr.Textbox(
                label="Text Input", placeholder="Paste or type your text here...",
                lines=6, visible=True,
            )
            audio_input = gr.Audio(
                label="Audio File (MP3, WAV, OGG, M4A — max 5 minutes)",
                type="filepath", visible=False,
            )
            language_picker = gr.Dropdown(
                choices=LANGUAGES, value="Luganda", label="Target Language",
                info="The local Ugandan language for translation and speech output.",
            )
            submit_btn = gr.Button("Run Pipeline ▶", elem_classes=["submit-btn"])

        with gr.Column(scale=1):
            gr.HTML("<div class='pipeline-arrow'>📝 Transcript (audio input only)</div>")
            transcript_out = gr.Textbox(
                label="Transcript", interactive=False, lines=3,
                placeholder="Transcribed text will appear here (audio mode only).",
                elem_classes=["result-box"],
            )
            gr.HTML("<div class='pipeline-arrow'>⬇️ Summary</div>")
            summary_out = gr.Textbox(
                label="Summary", interactive=False, lines=4,
                placeholder="Summary of the input text will appear here.",
                elem_classes=["result-box"],
            )
            gr.HTML("<div class='pipeline-arrow'>⬇️ Translated Summary</div>")
            translation_out = gr.Textbox(
                label="Translated Summary", interactive=False, lines=4,
                placeholder="Translation into your selected language will appear here.",
                elem_classes=["result-box"],
            )
            gr.HTML("<div class='pipeline-arrow'>⬇️ Generated Speech</div>")
            audio_out = gr.Audio(label="Generated Audio", interactive=False, type="filepath")
            error_out = gr.Textbox(
                label="", interactive=False, lines=2, placeholder="",
                elem_classes=["error-box"], show_label=False,
            )

    gr.HTML("""
    <div style="margin-top:20px; padding:12px 16px; background:#fff8f0;
                border-radius:8px; border:1px solid #f0c080; font-size:0.85rem; color:#555;">
        <strong style="color:#DC7828;">Pipeline:</strong>
        &nbsp; Input → STT (audio only) → Summarise → Translate (Sunflower LLM) → TTS → Output
        &nbsp; | &nbsp; All AI powered by <strong>Sunbird AI</strong>
    </div>
    """)

    input_mode.change(fn=toggle_input_mode, inputs=[input_mode], outputs=[text_input, audio_input])
    submit_btn.click(
        fn=process,
        inputs=[input_mode, text_input, audio_input, language_picker],
        outputs=[transcript_out, summary_out, translation_out, audio_out, error_out],
    )

if __name__ == "__main__":
    demo.launch(show_error=True)
