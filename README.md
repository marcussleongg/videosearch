# Search

```bash
python search_app.py
# Opens at http://localhost:7860
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
- Use of embeddings from generative model (Gemma4:e2b) vs embeddings from a dedicated embedding model because recent papers. Although it has been the case that dedicated embedding models provide for better vector search results due to contrastive pretraining (1), recent papers have claimed that embeddings from generative models' hidden states can do as well if not better (2,3).
- Ollama does not have an exposed API to get raw embeddings from hidden layers, rendering the option of embeddings from generative model unviable (same issue for llama.cpp)
- embeddinggemma has 768-dim vectors, not an issue here but potentially an issue at scale
- Gemma4:e2b via Ollama does not take in audio natively, hence this project will not allow for semantic search via audio. Adding this in would simply be either: extracting audio file and sending it directly to a multimodal model to extract the transcription and other audio features, or having a separate model to transcribe speech and a multimodal model to extract the other audio features.

# Design decisions:

- 1 frame per 2 seconds for frame extraction
- While a LLM can be used to rerank the results after RRF, its use is dependent on the use case of the search. If precision is necessary with a small number of videos (e.g. top-k 5), then reranking would provide better results. In this hypothetical use case of providing a large set of videos that do not have to be too precise (and considering the costs in terms of going through each video for reranking), then a LLM rerank might be less viable. In this project, it is included with usage optional.

(1) https://arxiv.org/abs/2201.10005
(2) https://arxiv.org/abs/2603.08077
(3) https://arxiv.org/abs/2603.08429
