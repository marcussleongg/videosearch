import base64
import subprocess
import tempfile
from pathlib import Path

import ollama

import config


def extract_video_frames(video_path: str) -> tuple[list[str], float]:
    """Return (list of base64 JPEG frames, duration in seconds)."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-of", "csv=p=0",
         "-show_entries", "format=duration", video_path],
        capture_output=True, text=True,
    )
    try:
        duration = float(probe.stdout.strip())
    except ValueError:
        duration = 60.0

    num_frames = max(1, int(duration / 2))
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_frames):
            t = duration * (i + 0.5) / num_frames
            out = f"{tmpdir}/frame{i}.jpg"
            subprocess.run(
                ["ffmpeg", "-ss", str(t), "-i", video_path,
                 "-frames:v", "1", "-q:v", "2", out],
                capture_output=True,
            )
            p = Path(out)
            if p.exists():
                frames.append(base64.b64encode(p.read_bytes()).decode())

    return frames, duration


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
    """Embed text using embeddinggemma. Returns a 768-dim vector."""
    response = ollama.embeddings(model=config.EMBED_MODEL, prompt=text)
    return response["embedding"]
