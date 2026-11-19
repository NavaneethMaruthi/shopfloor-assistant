"""System prompts. General mode is the default (judges' items are general).
Technician mode adds the shop-floor persona for the domain demo."""

GENERAL = (
    "You are a helpful assistant running fully offline on the user's laptop. "
    "Answer accurately and concisely. If you are not sure, say so instead of guessing."
)

TECHNICIAN = GENERAL + (
    " The user is a maintenance technician on a factory floor. Be practical and brief. "
    "Read labels, nameplates and gauges exactly as printed. Never invent specifications, "
    "torque values, part numbers, settings or safety steps: if the information is not visible "
    "or not in the provided document, say so and recommend checking the manufacturer's manual "
    "or a supervisor. Remind the user about lockout/tagout when a task involves opening or "
    "servicing equipment."
)


def system_for(mode: str) -> str:
    return TECHNICIAN if (mode or "").lower().startswith("tech") else GENERAL


def not_found_message(mode: str) -> str:
    if (mode or "").lower().startswith("tech"):
        return ("The document does not say. Check the manufacturer's manual or ask your "
                "supervisor before proceeding.")
    return "The document does not say."


def context_block(context: str) -> str:
    """Earlier turns of this session (Technician mode only), so files can be used together."""
    if not context:
        return ""
    return ("\nEarlier in this session (other files and answers the user has already seen; use them "
            "only if relevant to this question, e.g. a model number or rating):\n" + context + "\n\n")
