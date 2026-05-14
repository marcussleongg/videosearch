# Considerations:

- Ollama does not have an exposed API for video attachments, hence a manual frame extraction is required
- Use of embeddings from generative model (Gemma4:e2b) vs embeddings from a dedicated embedding model because recent papers. Although it has been the case that dedicated embedding models provide for better vector search results due to contrastive pretraining (1), recent papers have claimed that embeddings from generative models' hidden states can do as well if not better (2,3).

(1) https://arxiv.org/abs/2201.10005
(2) https://arxiv.org/abs/2603.08077
(3) https://arxiv.org/abs/2603.08429
