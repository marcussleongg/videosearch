#!/usr/bin/env python3
"""
Gradio search UI for the video search index.

Usage:
  python search_app.py
  # Opens at http://localhost:7860
"""

import gradio as gr

from search import search


def _run_search(query: str, top_k: int, use_reranker: bool):
    results = search(query, top_k, use_reranker)
    return [[
        r["filename"], r["description"], r["angle"],
        r["footage"], r["duration_s"], r["score"], r["source"],
    ] for r in results]


with gr.Blocks(title="Video Search") as app:
    gr.Markdown("## Video Search")
    with gr.Row():
        query_box = gr.Textbox(label="Search query", placeholder="e.g. person running on a beach", scale=4)
        top_k_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top K")
    reranker_checkbox = gr.Checkbox(label="Rerank with Ollama", value=False)
    search_btn = gr.Button("Search", variant="primary")
    results_table = gr.Dataframe(
        headers=["filename", "description", "angle", "footage", "duration_s", "score", "source"],
        datatype=["str", "str", "str", "str", "number", "number", "str"],
        interactive=False,
    )

    search_btn.click(fn=_run_search, inputs=[query_box, top_k_slider, reranker_checkbox], outputs=results_table)
    query_box.submit(fn=_run_search, inputs=[query_box, top_k_slider, reranker_checkbox], outputs=results_table)

if __name__ == "__main__":
    app.launch()
