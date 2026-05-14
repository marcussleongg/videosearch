# Video Search System — Implementation Plan

## Context

Build a semantic video search system on top of an existing Ollama/chat codebase. The system has two pipelines:
1. **Ingestion (CLI):** extract frames → generate description via Gemma4:e2b → embed → store in Pinecone + Supabase
2. **Search (Gradio):** embed query → Pinecone similarity search → return results with metadata

The user also wants to explore both **description-based embeddings** (text from vision output) and **raw image embeddings** (pass image directly to the embedding endpoint) from Gemma4:e2b. We design for both and document which works.

## File Structure

```
videosearch/
├── chat.py              (existing, untouched)
├── config.py            (env-based configuration)
├── embeddings.py        (embedding utilities for both approaches)
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
- `OLLAMA_MODEL` (default: `gemma4:e2b`)
- `PINECONE_API_KEY`, `PINECONE_INDEX` (default: `video-search`)
- `SUPABASE_URL`, `SUPABASE_KEY`
- `FRAMES_PER_VIDEO` (default: `8`)

## embeddings.py

Three public functions used by both ingest and search:

### `get_description_from_frames(frames: list[str]) -> str`
- Calls `ollama.chat()` with model `gemma4:e2b`, attaches all base64 frames as images
- Prompt: `"Describe this video in detail: what is happening, who/what appears, the setting, actions, and any notable events. Be thorough for search purposes."`
- Returns the text response

### `get_text_embedding(text: str) -> list[float]`
- Calls `ollama.embeddings(model=..., prompt=text)`
- Returns the embedding vector

### `get_image_embedding(frame_b64: str) -> list[float] | None`
- **Exploratory:** calls the Ollama `/api/embed` endpoint directly via `httpx` with `{"model": ..., "input": "", "images": [frame_b64]}`
- Returns vector if successful, `None` if model doesn't support image embeddings (catches HTTP/model errors gracefully)
- Logs which path was taken so the user can see what's working

### `get_video_embedding(frames: list[str]) -> tuple[str, list[float], list[float] | None]`
- Returns `(description, description_embedding, avg_image_embedding_or_None)`
- If image embeddings work: averages embeddings across all frames
- Pinecone upload uses `description_embedding` by default; image embedding stored as metadata for future experiments

## ingest.py (CLI)

```
python ingest.py <video_path> [--frames N] [--dry-run]
```

Steps:
1. **Extract frames** — reuse `extract_video_frames()` logic from `chat.py`, adapted to return base64 strings + temp files
2. **Get video duration** — `ffprobe` subprocess (same pattern as chat.py)
3. **Generate embedding** — calls `get_video_embedding()` from `embeddings.py`
4. **Upsert to Pinecone** — vector ID: `video_{uuid4()}`, payload: description, filename, duration
5. **Insert to Supabase** — stores full metadata including `pinecone_id`
6. Print summary: filename, description preview, Pinecone ID, Supabase row ID

Key reuse from `chat.py`:
- `extract_video_frames()` at lines ~60–100 (adapt to accept a target count and return b64 list)
- `encode_image()` at lines ~30–35

## search_app.py (Gradio)

Single-page Gradio app:
- **Input:** text search box + "Search" button + top-K slider (1–20, default 5)
- **Action:** `get_text_embedding(query)` → Pinecone `query(vector=..., top_k=k)` → fetch rows from Supabase by `pinecone_id`
- **Output:** Gradio `Dataframe` or `Gallery` showing: filename, description, similarity score
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

## Embedding Dimension Discovery

We don't hard-code the dimension. On first run of `ingest.py`, probe the dimension by calling `get_text_embedding("test")` and use `len(result)`. Create the Pinecone index with that dimension if it doesn't exist yet. Log the discovered dimension.

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
2. Set `.env` with all keys
3. `python ingest.py sample.mp4` — should print description + Pinecone/Supabase IDs
4. `python search_app.py` — open `localhost:7860`, search for content from the video
5. Check Pinecone console for uploaded vectors
6. Check Supabase table for metadata row
