#!/usr/bin/env python3
"""
Gradio search UI for the video search index.

Usage:
  python app.py
  # Opens at http://localhost:7860
"""

import sys
from pathlib import Path

import gradio as gr
import ollama

from search import search


def _check_ollama():
    try:
        ollama.list()
    except Exception:
        sys.exit("Cannot reach Ollama. Start it with: ollama serve")


def _run_search(query: str, top_k: int, use_reranker: bool, min_score: float):
    try:
        results, stats = search(query, top_k, use_reranker, min_score=min_score)
    except Exception as e:
        raise gr.Error(str(e))
    rows = [[
        r["filename"],
        r["description"][:120] + ("…" if len(r["description"]) > 120 else ""),
        r["angle"], r["footage"], r["duration_s"], r["score"], r["source"],
    ] for r in results]
    return rows, stats, results, None, ""


def _on_row_select(results, evt: gr.SelectData):
    if not results or evt.index[0] >= len(results):
        return None, ""
    r = results[evt.index[0]]
    file_path = r.get("file_path", "")
    if not file_path:
        gr.Warning("No file path stored for this video.")
        return None, r.get("description", "")
    if not Path(file_path).exists():
        gr.Warning(f"Video file not found: {file_path}")
        return None, r.get("description", "")
    return str(file_path), r.get("description", "")


_check_ollama()

with gr.Blocks(title="Video Search", theme=gr.themes.Default(primary_hue="purple")) as app:
    gr.Markdown("## Video Search")
    results_state = gr.State([])
    with gr.Row():
        query_box = gr.Textbox(label="Search query", placeholder="e.g. person running on a beach", scale=4)
        top_k_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Top K")
        min_score_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label="Min vector score")
    reranker_checkbox = gr.Checkbox(label="Post RRF Rerank", value=False)
    search_btn = gr.Button("Search", variant="primary")
    stats_line = gr.Markdown()
    results_table = gr.Dataframe(
        headers=["filename", "description", "angle", "footage type", "duration (s)", "score", "source"],
        datatype=["str", "str", "str", "str", "number", "number", "str"],
        column_widths=["14%", "30%", "10%", "10%", "10%", "10%", "11%"],
        interactive=False,
    )
    with gr.Row():
        desc_box = gr.Textbox(label="Full description", lines=4, interactive=False, scale=1)
        player = gr.Video(label="Playback", interactive=False, scale=1)

    search_btn.click(fn=_run_search, inputs=[query_box, top_k_slider, reranker_checkbox, min_score_slider], outputs=[results_table, stats_line, results_state, player, desc_box])
    query_box.submit(fn=_run_search, inputs=[query_box, top_k_slider, reranker_checkbox, min_score_slider], outputs=[results_table, stats_line, results_state, player, desc_box])
    results_table.select(fn=_on_row_select, inputs=[results_state], outputs=[player, desc_box])

if __name__ == "__main__":
    app.launch(allowed_paths=["/"])
