#!/usr/bin/env python3
"""
Local Ollama chat CLI with file attachment support.

Usage:
  python chat.py                        # use default model
  python chat.py --model llava          # multimodal model (images/video)
  python chat.py --model llama3.2 --no-history

In-session commands:
  /file <path>   attach a file (image, video, text, code)
  /model <name>  switch model mid-session
  /clear         clear conversation history
  /help          show commands
  /quit          exit
"""

import argparse
import base64
import mimetypes
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import ollama
except ImportError:
    sys.exit("Missing dependency: pip install ollama")


TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".yaml", ".yml",
    ".toml", ".md", ".txt", ".csv", ".html", ".css", ".sh", ".env",
    ".rs", ".go", ".java", ".c", ".cpp", ".h", ".sql", ".xml",
}


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def extract_video_frames(video_path: str, num_frames: int = 4) -> list[str]:
    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-of", "csv=p=0",
             "-show_entries", "format=duration", video_path],
            capture_output=True, text=True,
        )
        try:
            duration = float(probe.stdout.strip())
        except ValueError:
            duration = 60.0

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

    return frames


def process_file(raw_path: str) -> tuple[str, list[str]]:
    """Return (text_to_inject, list_of_base64_images)."""
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        return f"[Error: file not found: {path}]", []

    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or ""

    if mime.startswith("image/"):
        return f"[Image attached: {path.name}]", [encode_image(path)]

    if mime.startswith("video/"):
        print(f"  Extracting frames from {path.name}…", flush=True)
        frames = extract_video_frames(str(path))
        if not frames:
            return f"[Video attached but no frames extracted — is ffmpeg installed?]", []
        return f"[Video attached: {path.name} ({len(frames)} frames sampled)]", frames

    if mime.startswith("text/") or path.suffix.lower() in TEXT_EXTENSIONS:
        try:
            content = path.read_text(errors="replace")
            label = f"[File: {path.name}]"
            return f"{label}\n```\n{content}\n```", []
        except Exception as e:
            return f"[Could not read {path.name}: {e}]", []

    return f"[Unsupported file type ({mime or 'unknown'}) for {path.name}]", []


def ensure_model(client: ollama.Client, model: str) -> bool:
    try:
        running = {m.model for m in client.list().models}
        if model not in running:
            print(f"  Pulling {model}…", flush=True)
            for chunk in client.pull(model, stream=True):
                status = getattr(chunk, "status", "")
                if status:
                    print(f"\r  {status}", end="", flush=True)
            print()
        return True
    except Exception as e:
        print(f"  Error loading model: {e}")
        return False


def keep_model_warm(client: ollama.Client, model: str):
    """Send an empty generate to pin the model in memory."""
    try:
        client.generate(model=model, prompt="", keep_alive=-1)
    except Exception:
        pass


def chat_loop(client: ollama.Client, model: str, use_history: bool):
    history: list[dict] = []
    pending_images: list[str] = []
    pending_text_parts: list[str] = []

    print(f"\n  Model: {model}  |  /help for commands  |  Ctrl-C to exit\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        # --- commands ---
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/quit":
                print("Bye.")
                break

            elif cmd == "/clear":
                history.clear()
                pending_images.clear()
                pending_text_parts.clear()
                print("  History cleared.")

            elif cmd == "/help":
                print(
                    "  /file <path>   attach image, video, or text file\n"
                    "  /model <name>  switch model\n"
                    "  /clear         clear conversation history\n"
                    "  /quit          exit"
                )

            elif cmd == "/model":
                if not arg:
                    print(f"  Current model: {model}")
                else:
                    new_model = arg.strip()
                    if ensure_model(client, new_model):
                        model = new_model
                        keep_model_warm(client, model)
                        history.clear()
                        pending_images.clear()
                        pending_text_parts.clear()
                        print(f"  Switched to {model} (history cleared).")

            elif cmd == "/file":
                if not arg:
                    print("  Usage: /file <path>")
                else:
                    text_part, images = process_file(arg)
                    pending_text_parts.append(text_part)
                    pending_images.extend(images)
                    print(f"  Queued: {text_part.splitlines()[0]}")

            else:
                print(f"  Unknown command: {cmd}  (try /help)")

            continue

        # --- build message ---
        content_parts = pending_text_parts + ([user_input] if user_input else [])
        full_content = "\n".join(content_parts)
        pending_text_parts.clear()

        message: dict = {"role": "user", "content": full_content}
        if pending_images:
            message["images"] = pending_images
            pending_images.clear()

        messages = (history + [message]) if use_history else [message]

        # --- call model ---
        print(f"\n{model}> ", end="", flush=True)
        response_text = ""
        try:
            stream = client.chat(
                model=model,
                messages=messages,
                stream=True,
                options={"keep_alive": -1},
            )
            for chunk in stream:
                token = chunk.message.content or ""
                print(token, end="", flush=True)
                response_text += token
        except ollama.ResponseError as e:
            print(f"\n  Ollama error: {e}")
            continue
        except Exception as e:
            print(f"\n  Error: {e}")
            continue

        print("\n")

        if use_history:
            history.append(message)
            history.append({"role": "assistant", "content": response_text})


def main():
    parser = argparse.ArgumentParser(description="Chat with a local Ollama model.")
    parser.add_argument("--model", "-m", default="llama3.2",
                        help="Ollama model to use (default: llama3.2)")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama host (default: http://localhost:11434)")
    parser.add_argument("--no-history", action="store_true",
                        help="Don't maintain conversation history")
    parser.add_argument("--frames", type=int, default=4,
                        help="Number of frames to sample from videos (default: 4)")
    args = parser.parse_args()

    client = ollama.Client(host=args.host)

    try:
        client.list()
    except Exception:
        sys.exit(
            f"Cannot reach Ollama at {args.host}.\n"
            "Start it with:  ollama serve"
        )

    if not ensure_model(client, args.model):
        sys.exit(1)

    keep_model_warm(client, args.model)

    chat_loop(client, args.model, use_history=not args.no_history)


if __name__ == "__main__":
    main()
