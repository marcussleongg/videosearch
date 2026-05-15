from concurrent.futures import ThreadPoolExecutor

import ollama

import config
from embeddings import get_embedding
from ingest import get_pinecone_index, get_supabase_client

_supabase = None
_pinecone = None


def _clients():
    global _supabase, _pinecone
    if _supabase is None:
        _supabase = get_supabase_client()
    if _pinecone is None:
        _pinecone = get_pinecone_index()
    return _supabase, _pinecone


def _fts_search(client, query: str, top_k: int) -> list[str]:
    rows = (
        client.table("videos")
        .select("id")
        .text_search("description_fts", query, options={"type": "websearch"})
        .limit(top_k)
        .execute()
        .data
    )
    return [r["id"] for r in rows]


def _vector_search(index, query: str, top_k: int) -> list[str]:
    vector = get_embedding(query)
    results = index.query(vector=vector, top_k=top_k, include_metadata=False)
    return [m["id"] for m in results["matches"]]


def _rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    numbered = "\n".join(
        f"{i + 1}. {c['description']}" for i, c in enumerate(candidates)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Videos (numbered):\n{numbered}\n\n"
        "Return only the numbers in order of relevance to the query, "
        "most relevant first. One number per line. Example:\n3\n1\n2"
    )
    response = ollama.chat(
        model=config.VISION_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    indices = []
    for line in response["message"]["content"].strip().splitlines():
        try:
            indices.append(int(line.strip()) - 1)
        except ValueError:
            pass
    seen = set(indices)
    indices += [i for i in range(len(candidates)) if i not in seen]
    return [candidates[i] for i in indices if i < len(candidates)]


def search(query: str, top_k: int, use_reranker: bool = False) -> list[dict]:
    """
    Returns a list of result dicts with keys:
      filename, description, angle, footage, duration_s, score, source
    """
    query = query.strip()
    if not query:
        return []

    client, index = _clients()
    candidate_k = top_k * 3

    with ThreadPoolExecutor(max_workers=2) as pool:
        fts_future = pool.submit(_fts_search, client, query, candidate_k)
        vec_future = pool.submit(_vector_search, index, query, candidate_k)
    fts_ids = fts_future.result()
    vec_ids = vec_future.result()

    fused = _rrf_fuse([fts_ids, vec_ids])[:top_k]
    if not fused:
        return []

    ids = [doc_id for doc_id, _ in fused]
    rows_by_id = {
        r["id"]: r
        for r in client.table("videos")
        .select("id,filename,description,angle,footage,duration_s")
        .in_("id", ids)
        .execute()
        .data
    }

    fts_set = set(fts_ids)
    vec_set = set(vec_ids)

    results = []
    for doc_id, score in fused:
        row = rows_by_id.get(doc_id)
        if not row:
            continue
        sources = []
        if doc_id in fts_set:
            sources.append("fts")
        if doc_id in vec_set:
            sources.append("vector")
        desc = row["description"] or ""
        results.append({
            "filename": row["filename"],
            "description": desc[:200] + ("…" if len(desc) > 200 else ""),
            "angle": row["angle"] or "",
            "footage": row["footage"] or "",
            "duration_s": row["duration_s"],
            "score": round(score, 4),
            "source": "+".join(sources) or "unknown",
        })

    if use_reranker and results:
        results = _rerank(query, results)

    return results
