"""
Quick probe: what embedding endpoints does gemma4:e2b actually support?

Tries 5 approaches and prints dimension + first few values on success,
or a clear error on failure. Run with: python test_embeddings.py
"""

import httpx

OLLAMA_HOST = "http://localhost:11434"
MODEL = "embeddinggemma"

# Minimal 1x1 white PNG — no external file needed for the image tests
TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)


def report(label: str, vec: list[float] | None) -> list[float] | None:
    if vec:
        print(f"  SUCCESS  dim={len(vec)}  first5={[round(v, 4) for v in vec[:5]]}")
    return vec


def test_sdk_text() -> list[float] | None:
    print("\n[1] ollama Python SDK — ollama.embeddings(prompt=text)")
    try:
        import ollama
        resp = ollama.embeddings(model=MODEL, prompt="test embedding")
        return report("[1]", resp["embedding"])
    except Exception as e:
        print(f"  FAILED  {e}")
        return None


def test_old_api_text() -> list[float] | None:
    print("\n[2] POST /api/embeddings — text only (older endpoint)")
    try:
        r = httpx.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": MODEL, "prompt": "test embedding"},
            timeout=60,
        )
        r.raise_for_status()
        return report("[2]", r.json()["embedding"])
    except Exception as e:
        print(f"  FAILED  {e}")
        return None


def test_new_api_text() -> list[float] | None:
    print("\n[3] POST /api/embed — text only (newer endpoint)")
    try:
        r = httpx.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": MODEL, "input": "test embedding"},
            timeout=60,
        )
        r.raise_for_status()
        vecs = r.json().get("embeddings", [])
        return report("[3]", vecs[0] if vecs else None)
    except Exception as e:
        print(f"  FAILED  {e}")
        return None


def test_new_api_image() -> list[float] | None:
    print("\n[4] POST /api/embed — image input (newer endpoint)")
    try:
        r = httpx.post(
            f"{OLLAMA_HOST}/api/embed",
            json={"model": MODEL, "input": "describe this image", "images": [TINY_PNG_B64]},
            timeout=60,
        )
        r.raise_for_status()
        vecs = r.json().get("embeddings", [])
        return report("[4]", vecs[0] if vecs else None)
    except Exception as e:
        print(f"  FAILED  {e}")
        return None


def test_old_api_image() -> list[float] | None:
    print("\n[5] POST /api/embeddings — image input (older endpoint)")
    try:
        r = httpx.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": MODEL, "prompt": "describe this image", "images": [TINY_PNG_B64]},
            timeout=60,
        )
        r.raise_for_status()
        return report("[5]", r.json()["embedding"])
    except Exception as e:
        print(f"  FAILED  {e}")
        return None


if __name__ == "__main__":
    print(f"Probing embedding support for {MODEL} at {OLLAMA_HOST}")
    print("=" * 60)

    results = {
        "sdk_text":      test_sdk_text(),
        "old_api_text":  test_old_api_text(),
        "new_api_text":  test_new_api_text(),
        "new_api_image": test_new_api_image(),
        "old_api_image": test_old_api_image(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, vec in results.items():
        status = f"OK  dim={len(vec)}" if vec is not None else "FAILED"
        print(f"  {name:<20} {status}")
