# Locally Hosted Models System for Video Ingestion and Natural Language Search

This project sets up a full pipeline of ingesting videos and natural language querying to search for videos. While the search experience has been built into a user-friendly gradio frontend, the ingestion process is one that requires running CLI commands. Using it requires setting up Supabase and Pinecone databases, and installing Ollama along with the preferred models. Steps taken to set up the full system is documented [here](#setup). The rationale for using locally hosted models for this project is to test the capabilities of new open-weight models (I used [gemma4:e2b](https://huggingface.co/google/gemma-4-E2B)), specifically on limited hardware of a M2 MacBook Air (really put my laptop through its paces here), and thus avoiding costs of API calls. Furthermore, I used [embeddinggemma](https://huggingface.co/google/embeddinggemma-300m) as the dedicated embedding model, the choice here made to simply stay in the Gemma ecosystem with little drawbacks.

The project was built around Ollama's limitations on the modalities of input. Thus the full (video-native, audio) capabilties of Gemma4 was not fully utilized. I discuss this more in the [Considerations](#consideration) section. The system is built to be as model-agnostic as possible, so if a different models on Ollama are to be used, it simply requires changing the model name in .env,

The videos I used as data in this project are subsets of Meta's PE Video Dataset [(1)](#references), along with first and third person videos from Ai2's Charades-Ego dataset [(2)](#references).

# Pipelines

### Full system

```
 INGESTION
 ─────────────────────────────────────────────────────────
 video.mp4
   │
   ├─→ ffmpeg (1 frame per 2s) ──→ base64 JPEG frames
   │
   ├─→ Ollama gemma4:e2b (vision) ──→ text description
   │       "Rolling hills with fog, wind turbines..."
   │       [Angle: aerial] [Footage: real-life]
   │
   ├─→ Ollama embeddinggemma ──→ 768-dim vector
   │
   ├─→ Pinecone upsert (vector + metadata)
   └─→ Supabase insert (description, angle, footage, path...)


 SEARCH
 ─────────────────────────────────────────────────────────
 text query
   │
   ├─→ Ollama embeddinggemma ──→ 768-dim vector
   │
   ├─→ [parallel retrieval]
   │     ├─→ Supabase FTS  ──→ ranked IDs
   │     └─→ Pinecone ANN  ──→ ranked IDs
   │
   ├─→ Reciprocal Rank Fusion ──→ fused ranked IDs
   │
   ├─→ Supabase fetch metadata by ID
   │
   ├─→ [optional] Ollama gemma4:e2b reranker
   │
   └─→ results table (Gradio)
```

### Retrieval detail

```
 text query
   │
   ├──────────────────────────────────────┐
   │                                      │
   ▼                                      ▼
 Supabase FTS                        Pinecone ANN
                                     embeddinggemma → 768-dim vector
 searches description_fts tsvector   cosine similarity, top_k × 3 candidates
 exact + stemmed keyword match        filtered by min_score threshold
   │                                      │
   │  [id, id, id, ...]                   │  [id, id, id, ...]
   │  ranked by text relevance            │  ranked by cosine similarity
   │                                      │
   └──────────────┬───────────────────────┘
                  │
                  ▼
        Reciprocal Rank Fusion
        score = Σ 1 / (60 + rank_i)
        results found by both paths score higher
                  │
                  ▼
        top-K fused IDs
        → fetch full rows from Supabase
                  │
          ┌───────┴────────┐
          │  [optional]    │
          ▼                ▼
    Ollama reranker    return results
    gemma4:e2b         (if rerank off)
    single batch call,
    all descriptions
    in one prompt
          │
          ▼
      reordered top-K
      → return results
```

# Ingestion

```bash
# single video
python ingest.py video.mp4

# test description quality without writing to DBs
python ingest.py video.mp4 --dry-run

# ingest all .mp4s in a folder
python ingest.py --folder /path/to/videos
```

# Considerations:

- Ollama does not have an exposed API for video attachments, hence a manual frame extraction is required
- Use of embeddings from generative model (Gemma4:e2b) vs embeddings from a dedicated embedding model because recent papers. Although it has been the case that dedicated embedding models provide for better vector search results due to contrastive pretraining [(3)](#references), recent papers have claimed that embeddings from generative models' hidden states can do as well if not better [(4,5)](#references).
- Ollama does not have an exposed API to get raw embeddings from hidden layers, rendering the option of embeddings from generative model unviable (same issue for llama.cpp)
- embeddinggemma has 768-dim vectors, not an issue here but potentially an issue at scale
- Gemma4:e2b via Ollama does not take in audio natively, hence this project will not allow for semantic search via audio. Adding this in would simply be either: extracting audio file and sending it directly to a multimodal model to extract the transcription and other audio features, or having a separate model to transcribe speech and a multimodal model to extract the other audio features.
- Could have implemented a "filtered" search, where options like angle and footage type can be specified before the search, but given the limited video library size in this project, it is left out and instead the filter is post retrieval.

# Design decisions:

- 1 frame per 2 seconds for frame extraction
- While a LLM can be used to rerank the results after RRF, its use is dependent on the use case of the search. If precision is necessary with a small number of videos (e.g. top-k 5), then reranking would provide better results. In a use case of having a large set of videos post-RRF, despite sending all video descriptions as one query to the LLM, this can fill up the context window with many video descriptions and reranking might also perform poorly, a LLM rerank might be less viable. In this project, it is included with usage optional. Using it will result in longer waiting times.
- We return top_k \* 3 for the number of results from a Pinecone vector search to allow for a larger pool of items before RRF.
- Allowed for a min vector score filter, that only returns search results with a Pinecone vector search score above the set threshold. This allows users to specify how semantically close they want their results to be to their query.

# References

(1) Daniel Bolya et al., “Perception Encoder: The Best Visual Embeddings Are Not at the Output of the Network,” arXiv:2504.13181, preprint, arXiv, April 28, 2025, https://doi.org/10.48550/arXiv.2504.13181.

(2) Gunnar A. Sigurdsson et al., “Actor and Observer: Joint Modeling of First and Third-Person Videos,” arXiv:1804.09627, preprint, arXiv, April 25, 2018, https://doi.org/10.48550/arXiv.1804.09627.

(3) Arvind Neelakantan et al., “Text and Code Embeddings by Contrastive Pre-Training,” arXiv:2201.10005, preprint, arXiv, January 24, 2022, https://doi.org/10.48550/arXiv.2201.10005.

(4) Matei Benescu and Ivo Pascal de Jong, “Why Large Language Models Can Secretly Outperform Embedding Similarity in Information Retrieval,” arXiv:2603.08077, preprint, arXiv, March 9, 2026, https://doi.org/10.48550/arXiv.2603.08077.

(5) Bo Jiang, “One Model Is Enough: Native Retrieval Embeddings from LLM Agent Hidden States,” arXiv:2603.08429, preprint, arXiv, March 9, 2026, https://doi.org/10.48550/arXiv.2603.08429.

# Setup

### 1. Dependencies

```bash
pip install -r requirements.txt
```

ffmpeg is also required for frame extraction:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

### 2. Ollama

Install Ollama from https://ollama.com, then pull the required models:

```bash
ollama pull gemma4:e2b
ollama pull embeddinggemma
```

Ollama must be running before starting the app or ingesting videos:

```bash
ollama serve
```

### 3. Pinecone

Create a free account at https://pinecone.io and create a new API key. The index is created automatically on first ingest — no manual setup needed. Take note of the vector dimension if manually setting up an index.

### 4. Supabase

Create a free project at https://supabase.com. In the SQL editor, run the following to create the videos table:

```sql
CREATE TABLE videos (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  filename    TEXT NOT NULL,
  file_path   TEXT,
  description TEXT,
  duration_s  FLOAT,
  frame_count INTEGER,
  angle       TEXT,
  footage     TEXT,
  ingested_at TIMESTAMPTZ DEFAULT NOW()
);
```

In this project, I also made angle and footage enumerated types, based on the types I restricted the LLM to output. This is a safety precaution in the event of LLM hallucination outputting a value outside of the enumerated types. This can also be dealt with through deterministic code.

Then add a generated tsvector column for full-text search:

```sql
ALTER TABLE videos
  ADD COLUMN description_fts tsvector
  GENERATED ALWAYS AS (to_tsvector('english', coalesce(description, ''))) STORED;

CREATE INDEX videos_fts_idx ON videos USING GIN (description_fts);
```

### 5. Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```
OLLAMA_HOST=http://localhost:11434
VISION_MODEL=gemma4:e2b
EMBED_MODEL=embeddinggemma
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX=video-search
PINECONE_DIMENSION=768
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
```

# Running the app

```bash
python app.py
# Opens at http://localhost:7860
```
