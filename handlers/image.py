"""Image: describe a photo and answer questions about it (Qwen3-VL)."""
import config
import prompts
from handlers import llm

INSTRUCTIONS = (
    "Look carefully at the image and answer the question.\n"
    "- If the question asks 'how many', count carefully and START your answer with the number "
    "as a digit, then a short explanation.\n"
    "- If asked what text says, copy the text exactly as printed.\n"
    "- If the image is too blurry, dark or cropped to answer, say so and say what you can see.\n"
)


def handle(question, path, mode="General"):
    q = (question or "").strip() or "Describe this image in detail."
    answer = llm.chat(
        config.VL_MODEL,
        f"{INSTRUCTIONS}\nQuestion: {q}",
        system=prompts.system_for(mode),
        images=[llm.image_bytes(path)],
        think=config.THINK_IMAGE,
    )
    return {"answer": answer, "details": "", "models": config.VL_MODEL}
