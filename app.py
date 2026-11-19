"""Shopfloor Assistant: one chat interface for images, audio, documents, code and video.
Everything runs locally (Ollama + faster-whisper + AST). Run:  python app.py
"""
import config  # noqa: F401  (must be first: sets offline flags)

import os
import traceback

import gradio as gr

from handlers import router

EMPTY_BOX = {"text": "", "files": []}


def _path_of(f):
    if isinstance(f, str):
        return f
    if isinstance(f, dict):
        return f.get("path") or f.get("name")
    return getattr(f, "path", None) or getattr(f, "name", None)


def _format(result):
    parts = [result.get("answer", "")]
    if result.get("details"):
        parts.append(result["details"])
    parts.append(f"<sub>⏱ {result.get('latency', 0):.1f} s, {result.get('modality', '')}, "
                 f"{result.get('models', '')}</sub>")
    return "\n\n".join(p for p in parts if p)


def respond(message, history, state, mode):
    message = message or {}
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    new_path = _path_of(files[-1]) if files else None
    history = list(history or [])
    state = dict(state or {})

    if not text and not new_path:
        yield history, EMPTY_BOX, state
        return

    shown = f"📎 {os.path.basename(new_path)}" if new_path else ""
    if text:
        shown = f"{shown}\n\n{text}" if shown else text
    history.append({"role": "user", "content": shown})
    history.append({"role": "assistant", "content": "⏳ Working on it…"})
    yield history, EMPTY_BOX, state

    # A new file becomes the "current" file; follow-up questions reuse it.
    if new_path:
        state["last_file"] = new_path
        path = new_path
    elif router.is_code_request(text):
        path = None
    else:
        path = state.get("last_file")

    try:
        result = router.handle(text, path, mode)
        history[-1] = {"role": "assistant", "content": _format(result)}
    except Exception as e:
        traceback.print_exc()
        history[-1] = {"role": "assistant",
                       "content": f"Something went wrong: {e}\n\nCheck that Ollama is running "
                                  "(`ollama serve`) and try again."}
    yield history, EMPTY_BOX, state


def clear():
    return [], EMPTY_BOX, {"last_file": None}


def build_ui():
    with gr.Blocks(title="Shopfloor Assistant") as demo:
        gr.Markdown(
            "## 🔧 Shopfloor Assistant\n"
            "Attach a photo, sound or voice recording, video, or a PDF/TXT document and ask about it, "
            "or ask for Python code. Follow-up questions use the last file you attached. "
            "Runs entirely on this laptop, no internet needed."
        )
        with gr.Row():
            mode = gr.Radio(["General", "Technician"], value="General", label="Mode", scale=3)
            clear_btn = gr.Button("New conversation", scale=1)

        try:
            chatbot = gr.Chatbot(type="messages", height=520, label="Assistant")
        except TypeError:  # newer Gradio: messages format is the only format
            chatbot = gr.Chatbot(height=520, label="Assistant")

        box_kwargs = dict(
            file_types=["image", "audio", "video", ".pdf", ".txt"],
            placeholder="Ask a question, attach a file, or record audio…",
            show_label=False,
        )
        try:
            box = gr.MultimodalTextbox(sources=["upload", "microphone"], **box_kwargs)
        except TypeError:
            box = gr.MultimodalTextbox(**box_kwargs)

        state = gr.State({"last_file": None})
        box.submit(respond, [box, chatbot, state, mode], [chatbot, box, state])
        clear_btn.click(clear, None, [chatbot, box, state])
    return demo


if __name__ == "__main__":
    print("Warming up models (first run takes a little longer)…")
    try:
        print(f"Models ready in {router.warmup():.1f} s")
    except Exception as e:
        print(f"Warm-up failed: {e}\nIs Ollama running? Start it with: ollama serve")
    build_ui().queue().launch(server_name="0.0.0.0", server_port=7860)
