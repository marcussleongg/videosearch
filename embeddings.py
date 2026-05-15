import ollama

import config


def get_description(frames: list[str]) -> str:
    """Send frames to Gemma4 and return a description of the video."""
    prompt = (
        "You are generating a search index description for a video. "
        "Here are frames sampled evenly from the video.\n\n"
        "Write 2-3 sentences covering: the setting, who or what appears, "
        "what is happening, and any notable visual details like colours or objects. "
        "Write as flowing prose, not bullet points. "
        "Be specific and use concrete keywords. State if it is game footage/animation. "
        "Do not speculate beyond what is visible.\n\n"
        "Then on separate lines add these two tags in exactly this format:\n"
        "[Angle: <value>] where <value> is one of: eye-level, POV, top-down, low-angle, back-view, aerial, side-on.\n"
        "[Footage: <value>] where <value> is one of: real-life, animated, game."
    )
    response = ollama.chat(
        model=config.VISION_MODEL,
        messages=[{"role": "user", "content": prompt, "images": frames}],
        options={"num_ctx": 8192},
    )
    return response["message"]["content"].strip()


def get_embedding(text: str) -> list[float]:
    """Embed text using dedicated embedding model and get a vector of dim of embedding model."""
    response = ollama.embeddings(model=config.EMBED_MODEL, prompt=text)
    return response["embedding"]
