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
- Use of embeddings from generative model (Gemma4:e2b) vs embeddings from a dedicated embedding model because recent papers. Although it has been the case that dedicated embedding models provide for better vector search results due to contrastive pretraining (1), recent papers have claimed that embeddings from generative models' hidden states can do as well if not better (2,3).
- Ollama does not have an exposed API to get raw embeddings from hidden layers, rendering the option of embeddings from generative model unviable (same issue for llama.cpp)
- embeddinggemma has 768-dim vectors, not an issue here but potentially an issue at scale
- Gemma4:e2b via Ollama does not take in audio natively, hence this project will not allow for semantic search via audio. Adding this in would simply be either: extracting audio file and sending it directly to a multimodal model to extract the transcription and other audio features, or having a separate model to transcribe speech and a multimodal model to extract the other audio features.

# Design decisions:

- 1 frame per 2 seconds for frame extraction

(1) https://arxiv.org/abs/2201.10005
(2) https://arxiv.org/abs/2603.08077
(3) https://arxiv.org/abs/2603.08429
