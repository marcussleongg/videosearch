# Video Search System — Implementation Plan

## Context

Build a semantic video search system using local models served via Ollama. The system has two pipelines:

1. **Ingestion (CLI):** extract frames → describe via Gemma4:e2b → embed with embeddinggemma → store in Pinecone + Supabase
2. **Search (Gradio):** embed query with embeddinggemma → Pinecone similarity search → return results with metadata

**Model split (confirmed via testing):**
- `gemma4:e2b` — vision only: takes frames as images, returns a rich text description. Does not support embedding endpoints.
- `embeddinggemma` — embeddings only: Google's purpose-built 300M embedding model from the same Gemma family. Embeds both video descriptions and user queries into the same semantic space.

Pull both before running:
```bash
ollama pull gemma4:e2b
ollama pull embeddinggemma
```

## File Structure

```
videosearch/
├── chat.py              (existing, untouched)
├── config.py            (env-based configuration)
├── embeddings.py        (vision description + embedding utilities)
├── ingest.py            (CLI ingestion pipeline)
├── search_app.py        (Gradio search UI)
├── requirements.txt
└── .env                 (API keys, not committed)
```

## Supabase Table Schema

Run once in the Supabase SQL editor:

```sql
CREATE TABLE videos (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename    TEXT NOT NULL,
  file_path   TEXT,
  description TEXT,
  pinecone_id TEXT UNIQUE NOT NULL,
  duration_s  FLOAT,
  frame_count INTEGER,
  ingested_at TIMESTAMPTZ DEFAULT NOW()
);
```

## config.py

Load all secrets from `.env`. Expose typed constants:

- `OLLAMA_HOST` (default: `http://localhost:11434`)
- `VISION_MODEL` (default: `gemma4:e2b`)
- `EMBED_MODEL` (default: `embeddinggemma`)
- `PINECONE_API_KEY`, `PINECONE_INDEX` (default: `video-search`)
- `PINECONE_DIMENSION` = `768` (confirmed from embeddinggemma)
- `SUPABASE_URL`, `SUPABASE_KEY`
- `FRAMES_PER_VIDEO` (default: `8`)

## embeddings.py

### `get_description_from_frames(frames: list[str]) -> str`
- Calls `ollama.chat()` with `gemma4:e2b`, attaches all base64 frames as images
- Prompt: `"Describe this video in detail: what is happening, who/what appears, the setting, actions, and any notable events. Be thorough for search purposes."`
- Returns the text response

### `get_text_embedding(text: str) -> list[float]`
- Calls `ollama.embeddings(model="embeddinggemma", prompt=text)`
- Returns 1024-dim vector

### `get_video_embedding(frames: list[str]) -> tuple[str, list[float]]`
- Returns `(description, embedding)`
- Calls `get_description_from_frames()` then `get_text_embedding()` on the result

## ingest.py (CLI)

```
python ingest.py <video_path> [--frames N] [--dry-run]
```

Steps:

1. **Extract frames** — reuse `extract_video_frames()` logic from `chat.py`, adapted to return base64 strings
2. **Get video duration** — `ffprobe` subprocess (same pattern as `chat.py`)
3. **Generate embedding** — calls `get_video_embedding()` from `embeddings.py`
4. **Upsert to Pinecone** — vector ID: `video_{uuid4()}`, metadata: description, filename, duration
5. **Insert to Supabase** — stores full metadata including `pinecone_id`
6. Print summary: filename, description preview, Pinecone ID, Supabase row ID

Key reuse from `chat.py`:
- `extract_video_frames()` (adapt to return b64 list)
- `encode_image()`

## search_app.py (Gradio)

Single-page Gradio app:

- **Input:** text search box + "Search" button + top-K slider (1–20, default 5)
- **Action:** `get_text_embedding(query)` → Pinecone `query(vector=..., top_k=k)` → fetch rows from Supabase by `pinecone_id`
- **Output:** table showing filename, description, similarity score
- Launch with `python search_app.py` (Gradio default port 7860)

## requirements.txt

```
ollama
pinecone-client
supabase
gradio
python-dotenv
httpx
```

## Implementation Order

1. ~~Write this plan to `PLAN.md` in the repo root~~ ✓
2. Create `requirements.txt`
3. Create `.env.example`
4. Create `config.py`
5. Create `embeddings.py`
6. Create `ingest.py` (adapting frame extraction from `chat.py`)
7. Create `search_app.py` (Gradio)

## Verification

1. `pip install -r requirements.txt`
2. `ollama pull gemma4:e2b && ollama pull embeddinggemma`
3. Set `.env` with all keys
4. `python ingest.py sample.mp4` — should print description + Pinecone/Supabase IDs
5. `python search_app.py` — open `localhost:7860`, search for content from the video
6. Check Pinecone console for uploaded vectors
7. Check Supabase table for metadata row
