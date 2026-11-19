"""Shopfloor Assistant: one chat interface for images, audio, documents, code and video.
Everything runs locally (Ollama + faster-whisper + AST). Run:  python app.py
"""
import config  # noqa: F401  (must be first: sets offline flags)

import os
import re
import shutil
import subprocess
import traceback

import gradio as gr

from handlers import audio as audio_handler
from handlers import router

EMPTY_BOX = {"text": "", "files": []}
DEMO_DIR = os.path.join(config.ROOT, "demo_assets")
_say_proc = None


# ---------------- helpers ----------------
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


def speak(text):
    """Read the answer aloud with the Mac's built-in offline voice (hands-free use)."""
    global _say_proc
    if not shutil.which("say") or not text:
        return
    clean = re.sub(r"```.*?```", " Here is the code. ", text, flags=re.S)
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = re.sub(r"[*_`#>|]", "", clean).strip()[:600]
    try:
        if _say_proc and _say_proc.poll() is None:
            _say_proc.terminate()
        _say_proc = subprocess.Popen(["say", clean])
    except Exception:
        pass


def transcribe_question(path):
    wav = audio_handler.to_wav16k(path)
    try:
        return audio_handler.transcribe(wav)
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass


def session_context(state):
    """Last few turns, so files can be used together (Technician mode only)."""
    lines = []
    for h in state.get("history", [])[-3:]:
        src = f"{h['modality']}: {h['file']}" if h.get("file") else h["modality"]
        answer = re.sub(r"\s+", " ", h["answer"])[:300]
        lines.append(f"- [{src}] Q: {h['question'][:150]} A: {answer}")
    return "\n".join(lines)


# ---------------- main chat handler ----------------
def respond(message, history, state, mode, read_aloud, voice_questions):
    message = message or {}
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    new_path = _path_of(files[-1]) if files else None
    history = list(history or [])
    state = dict(state or {})
    state.setdefault("history", [])

    if not text and not new_path:
        yield history, EMPTY_BOX, state
        return

    # Voice question: a recording with no typed text becomes the question itself.
    spoken = None
    if voice_questions and new_path and not text and router.detect(new_path) == "audio":
        history.append({"role": "user", "content": "🎤 …"})
        history.append({"role": "assistant", "content": "⏳ Listening…"})
        yield history, EMPTY_BOX, state
        try:
            spoken = transcribe_question(new_path)
        except Exception as e:
            spoken = ""
            print(f"[voice] transcription failed: {e}")
        history = history[:-2]
        if not spoken:
            history.append({"role": "assistant", "content": "I couldn't hear a question in that recording. "
                                                             "Try again, or type it."})
            yield history, EMPTY_BOX, state
            return
        text, new_path = spoken, None

    if spoken:
        shown = f"🎤 “{spoken}”"
    else:
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

    technician = (mode or "").lower().startswith("tech")
    context = session_context(state) if technician else ""

    try:
        result = router.handle(text, path, mode, context=context)
        history[-1] = {"role": "assistant", "content": _format(result)}
        state["history"] = (state["history"] + [{
            "modality": result.get("modality", ""),
            "file": os.path.basename(path) if path else "",
            "question": text or "(no question)",
            "answer": result.get("answer", ""),
        }])[-6:]
        if read_aloud:
            speak(result.get("answer", ""))
    except Exception as e:
        traceback.print_exc()
        history[-1] = {"role": "assistant",
                       "content": f"Something went wrong: {e}\n\nCheck that Ollama is running "
                                  "(`ollama serve`) and try again."}
    yield history, EMPTY_BOX, state


def clear():
    return [], EMPTY_BOX, {"last_file": None, "history": []}


def demo_examples():
    """One-click examples from demo_assets/ (only the files that exist)."""
    wanted = [
        ("nameplate.jpg", "What's the rated current and voltage?"),
        ("vacuum.mp3", "What's that sound?"),
        ("motor_manual.pdf", "The motor is humming loudly. What should I check?"),
        ("motor_manual.pdf", "What's the torque spec for the mounting bolts?"),
        ("fan.mp4", "What's happening in this video?"),
    ]
    ex = [{"text": q, "files": [os.path.join(DEMO_DIR, f)]}
          for f, q in wanted if os.path.exists(os.path.join(DEMO_DIR, f))]
    ex.append({"text": "Write a Python function that flags a motor as overloaded if the measured "
                       "current is more than 115% of the rated full-load amps.", "files": []})
    return ex


# ---------------- UI ----------------
def build_ui():
    with gr.Blocks(title="Shopfloor Assistant") as demo:
        gr.Markdown(
            "## 🔧 Shopfloor Assistant\n"
            "Attach a photo, sound or voice recording, video, or a PDF/TXT document and ask about it, "
            "or ask for Python code. Follow-up questions use the last file you attached. "
            "Runs entirely on this laptop, no internet needed."
        )
        with gr.Row():
            mode = gr.Radio(["General", "Technician"], value="General", label="Mode", scale=3,
                            info="Technician: shop-floor answers, safety reminders, and remembers "
                                 "earlier files in the conversation")
            with gr.Column(scale=2):
                read_aloud = gr.Checkbox(False, label="🔊 Read answers aloud")
                voice_q = gr.Checkbox(False, label="🎤 Voice questions (a recording is my question)")
            clear_btn = gr.Button("New conversation", scale=1)

        try:
            chatbot = gr.Chatbot(type="messages", height=520, label="Assistant")
        except TypeError:  # newer Gradio: messages format is the only format
            chatbot = gr.Chatbot(height=520, label="Assistant")

        box_kwargs = dict(
            file_types=["image", "audio", "video", ".pdf", ".txt", ".heic", ".heif"],
            placeholder="Ask a question, attach a file, or record audio…",
            show_label=False,
            label="Message",
        )
        try:
            box = gr.MultimodalTextbox(sources=["upload", "microphone"], **box_kwargs)
        except TypeError:
            box = gr.MultimodalTextbox(**box_kwargs)

        try:
            gr.Examples(examples=demo_examples(), inputs=[box], label="Demo examples (click, then press Enter)")
        except Exception as e:
            print(f"[ui] examples not shown: {e}")

        state = gr.State({"last_file": None, "history": []})
        box.submit(respond, [box, chatbot, state, mode, read_aloud, voice_q], [chatbot, box, state])
        clear_btn.click(clear, None, [chatbot, box, state])
    return demo


if __name__ == "__main__":
    print("Warming up models (first run takes a little longer)…")
    try:
        print(f"Models ready in {router.warmup():.1f} s")
    except Exception as e:
        print(f"Warm-up failed: {e}\nIs Ollama running? Start it with: ollama serve")
    cert, key = os.path.join(config.ROOT, "cert.pem"), os.path.join(config.ROOT, "key.pem")
    launch_kwargs = dict(server_name="0.0.0.0", server_port=7860)
    if os.path.exists(cert) and os.path.exists(key):
        launch_kwargs.update(ssl_certfile=cert, ssl_keyfile=key, ssl_verify=False)
        print("HTTPS on: open https://<this-mac's-IP>:7860 (accept the certificate warning)")
    build_ui().queue().launch(**launch_kwargs)
