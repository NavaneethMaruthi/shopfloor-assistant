"""Shopfloor Assistant: one chat interface for images, audio, documents, code and video.
Everything runs locally (Ollama + faster-whisper + AST). Run:  python app.py
"""
import config  # noqa: F401  (must be first: sets offline flags)

import base64
import os
import re
import shutil
import tempfile
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


# Vocabulary hint for spoken questions (not used for audio files, so test scores are unaffected)
VOICE_HINT = ("A maintenance technician asks a question about a machine: What should I check if the motor "
              "is humming, whining, grinding, vibrating or overheating? Voltage, current, amps, nameplate, "
              "overload, bearing, belt tension, torque, manual, warranty, lockout, fan, compressor, pump.")


def transcribe_question(path):
    wav = audio_handler.to_wav16k(path)
    try:
        return audio_handler.transcribe(wav, prompt=VOICE_HINT, language="en")
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
def respond(message, history, state, mode, read_aloud, voice_questions=False, spoken_display=None):
    message = message or {}
    text = (message.get("text") or "").strip()
    files = [p for p in (_path_of(f) for f in (message.get("files") or [])) if p]
    history = list(history or [])
    state = dict(state or {})
    state.setdefault("history", [])

    # Voice question: with 🎤 ticked and nothing typed, a recording is the question and any
    # other attachment in the same message is what the question is about.
    recording = None
    if voice_questions and not text:
        recs = [p for p in files if router.detect(p) == "audio"]
        if recs:
            recording = recs[-1]
            files = [p for p in files if p != recording]
    new_path = files[-1] if files else None

    if not text and not new_path and not recording:
        yield history, EMPTY_BOX, state
        return

    spoken = None
    if recording:
        history.append({"role": "user", "content": "🎤 …"})
        history.append({"role": "assistant", "content": "⏳ Listening…"})
        yield history, EMPTY_BOX, state
        try:
            spoken = transcribe_question(recording)
        except Exception as e:
            spoken = ""
            print(f"[voice] transcription failed: {e}")
        history = history[:-2]
        if not spoken:
            history.append({"role": "assistant", "content": "I couldn't hear a question in that recording. "
                                                             "Try again, or type it."})
            yield history, EMPTY_BOX, state
            return
        text = spoken

    shown = f"📎 {os.path.basename(new_path)}" if new_path else ""
    spoken = spoken or spoken_display
    question_shown = f"🎤 “{spoken}”" if spoken else text
    if question_shown:
        shown = f"{shown}\n\n{question_shown}" if shown else question_shown
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


RECORD_JS = """
async () => {
  const btn = document.getElementById('voice_btn');
  const setLabel = (t) => { if (btn) btn.textContent = t; };
  if (window.__shopRec) { if (window.__shopRec.state !== 'inactive') window.__shopRec.stop(); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
  catch (e) { alert('The microphone is blocked. Allow it in the browser to ask by voice.'); return; }
  const rec = new MediaRecorder(stream);
  window.__shopRec = rec;
  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  const done = new Promise((res) => { rec.onstop = res; });
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 2048;
  ctx.createMediaStreamSource(stream).connect(analyser);
  const buf = new Float32Array(analyser.fftSize);
  const SILENCE_MS = __SILENCE_MS__, NO_SPEECH_MS = 10000, MAX_MS = 45000, THRESHOLD = 0.02;
  const t0 = performance.now();
  let heard = false, lastLoud = t0;
  rec.start(250);
  setLabel('🔴 Listening… stops after a pause (click to stop now)');
  const timer = setInterval(() => {
    analyser.getFloatTimeDomainData(buf);
    let sum = 0; for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length), now = performance.now();
    if (rms > THRESHOLD) { heard = true; lastLoud = now; }
    const silentTooLong = heard && now - lastLoud > SILENCE_MS;
    const nothingSaid = !heard && now - t0 > NO_SPEECH_MS;
    if ((silentTooLong || nothingSaid || now - t0 > MAX_MS) && rec.state !== 'inactive') rec.stop();
  }, 100);
  await done;
  clearInterval(timer);
  window.__shopRec = null;
  stream.getTracks().forEach((t) => t.stop());
  ctx.close();
  setLabel('🎤 Ask by voice');
  if (!heard) { alert("I didn't hear a question. Click and try again."); return; }
  const blob = new Blob(chunks, { type: rec.mimeType || 'audio/webm' });
  const dataUrl = await new Promise((r) => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(blob); });
  const ta = document.querySelector('#voice_b64 textarea');
  ta.value = dataUrl;
  ta.dispatchEvent(new Event('input', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 400));
  document.getElementById('voice_go').click();
}
""".replace("__SILENCE_MS__", str(int(config.VOICE_SILENCE_S * 1000)))


def voice_respond(data_url, pending, history, state, mode, read_aloud):
    """A recording from the 🎤 Ask by voice button: transcribe it, then answer it like a typed
    question about the current file."""
    history = list(history or [])
    if not data_url or "," not in data_url:
        yield history, EMPTY_BOX, state
        return
    header, b64 = data_url.split(",", 1)
    ext = ".mp4" if "mp4" in header else ".ogg" if "ogg" in header else ".webm"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(base64.b64decode(b64))
    tmp.close()
    history.append({"role": "user", "content": "🎤 …"})
    history.append({"role": "assistant", "content": "⏳ Listening…"})
    yield history, EMPTY_BOX, state
    try:
        spoken = transcribe_question(tmp.name)
    except Exception as e:
        spoken = ""
        print(f"[voice] transcription failed: {e}")
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass
    history = history[:-2]
    if not spoken:
        history.append({"role": "assistant", "content": "I couldn't make out a question. Click 🎤 and try again."})
        yield history, EMPTY_BOX, state
        return
    # a file waiting in the message box (attached but not sent yet) is what the question is about
    waiting = (pending or {}).get("files") or [] if isinstance(pending, dict) else []
    yield from respond({"text": spoken, "files": waiting}, history, state, mode, read_aloud,
                       False, spoken_display=spoken)


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
            "Attach a photo, sound or voice recording, video, a PDF, Word or text document, or a zip of them, "
            "and ask about it, "
            "or ask for Python code. Follow-up questions use the last file you attached. "
            "Runs entirely on this laptop, no internet needed."
        )
        with gr.Row():
            mode = gr.Radio(["General", "Technician"], value="General", label="Mode", scale=3,
                            info="Technician: shop-floor answers, safety reminders, and remembers "
                                 "earlier files in the conversation")
            with gr.Column(scale=2):
                read_aloud = gr.Checkbox(False, label="🔊 Read answers aloud")
                voice_btn = gr.Button("🎤 Ask by voice", elem_id="voice_btn", variant="primary")
            clear_btn = gr.Button("New conversation", scale=1)

        try:
            chatbot = gr.Chatbot(type="messages", height=520, label="Assistant")
        except TypeError:  # newer Gradio: messages format is the only format
            chatbot = gr.Chatbot(height=520, label="Assistant")

        box_kwargs = dict(
            file_types=["image", "audio", "video", ".pdf", ".txt", ".md", ".docx", ".zip", ".heic", ".heif"],
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

        # hidden plumbing for the voice button (hidden with CSS so the elements stay in the page)
        gr.HTML("<style>#voice_b64, #voice_go { display: none !important; }</style>")
        voice_b64 = gr.Textbox(elem_id="voice_b64", show_label=False, container=False)
        voice_go = gr.Button("go", elem_id="voice_go")

        state = gr.State({"last_file": None, "history": []})
        box.submit(respond, [box, chatbot, state, mode, read_aloud], [chatbot, box, state])
        voice_btn.click(None, None, None, js=RECORD_JS)
        voice_go.click(voice_respond, [voice_b64, box, chatbot, state, mode, read_aloud], [chatbot, box, state])
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
