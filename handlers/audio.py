"""Audio: transcribe speech, answer questions about it, identify sounds.

Pipeline: ffmpeg -> 16 kHz mono WAV -> AST sound classifier.
  Speech  -> faster-whisper transcript -> (transcript returned directly, or LLM answers from it)
  Sound   -> LLM turns the classifier's labels into a plain answer
"""
import os
import re
import subprocess
import tempfile
from functools import lru_cache

import config  # noqa: F401  (sets offline flags before transformers loads)
import prompts
import soundfile as sf
from handlers import llm

SPEECH_WORDS = ("speech", "speaking", "narration", "conversation", "monologue")
TRANSCRIBE_RE = re.compile(
    r"\b(transcribe|transcript|transcription|write down|word for word|what (is|was) (being )?said exactly)\b",
    re.I,
)


@lru_cache(maxsize=None)
def _whisper():
    from faster_whisper import WhisperModel
    return WhisperModel(config.WHISPER_SIZE, device="cpu", compute_type="int8")


@lru_cache(maxsize=None)
def _ast():
    from transformers import pipeline
    return pipeline("audio-classification", model=config.AST_MODEL, top_k=10)


def warmup():
    _whisper()
    _ast()


def to_wav16k(path):
    """Convert any audio (wav, mp3, m4a, webm...) to 16 kHz mono WAV."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", "-ar", "16000", out],
                   check=True)
    return out


def classify(samples):
    preds = _ast()({"raw": samples, "sampling_rate": 16000}, top_k=10)
    preds = sorted(preds, key=lambda p: p["score"], reverse=True)[:10]
    return [(p["label"], float(p["score"])) for p in preds]


def is_speech(labels):
    top = labels[0][0].lower()
    if any(w in top for w in SPEECH_WORDS):
        return True
    speech_mass = sum(s for l, s in labels if any(w in l.lower() for w in SPEECH_WORDS))
    return speech_mass > 0.4


def transcribe(wav_path):
    segments, _info = _whisper().transcribe(wav_path, beam_size=5)
    return " ".join(s.text.strip() for s in segments).strip()


def handle(question, path, mode="General"):
    q = (question or "").strip()
    wav = to_wav16k(path)
    try:
        samples, _sr = sf.read(wav, dtype="float32")
        labels = classify(samples)
        sounds = ", ".join(f"{l} ({s:.2f})" for l, s in labels)

        if is_speech(labels):
            transcript = transcribe(wav)
            if not q or TRANSCRIBE_RE.search(q):
                return {"answer": transcript or "(no speech recognised)",
                        "details": f"🔊 Detected: {sounds}",
                        "transcript": transcript,
                        "models": f"AST + whisper-{config.WHISPER_SIZE}"}
            answer = llm.chat(
                config.VL_MODEL,
                "Here is an automatic transcript of a speech recording (it may contain small errors):\n"
                f'"""{transcript}"""\n\n'
                "Answer the question using only the transcript. Be concise. "
                "If the transcript does not contain the answer, say so.\n"
                f"Question: {q}",
                system=prompts.system_for(mode),
                think=False,
            )
            return {"answer": answer,
                    "details": f"📝 Transcript: {transcript}\n\n🔊 Detected: {sounds}",
                    "transcript": transcript,
                    "models": f"AST + whisper-{config.WHISPER_SIZE} + {config.VL_MODEL}"}

        # Non-speech sound
        q = q or "What sound is this?"
        answer = llm.chat(
            config.VL_MODEL,
            "An audio classifier (AudioSet labels) analysed a recording. Its top 10 predictions, "
            f"with confidence from 0 to 1, are: {sounds}.\n"
            "Work out the single most likely real-world sound source. Generic labels such as 'Sound "
            "effect', 'Inside, small room', 'Outside, rural or natural', 'Vehicle', 'Noise' or 'Static' "
            "describe the setting, not the source: prefer specific labels even with lower scores, and "
            "combine related labels (e.g. 'Crackle' + 'Fire' = a crackling fire; 'Waves, surf' or 'Ocean' "
            "= ocean waves; 'Bark' + 'Dog' = a dog barking).\n"
            "Answer in ONE short sentence that names the sound (e.g. 'This is a dog barking.'). "
            "Only mention uncertainty if no specific label fits.\n"
            f"Question: {q}",
            system=prompts.system_for(mode),
            think=False,
        )
        return {"answer": answer, "details": f"🔊 Detected: {sounds}",
                "models": f"AST + {config.VL_MODEL}"}
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass
