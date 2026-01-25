from pathlib import Path
from typing import Optional

import gradio as gr

from src.config import DEFAULT_SAMPLE_FPS, FRAMES_DIR, LANCEDB_DIR
from src.indexer import index_video
from src.query import VideoSearcher


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _resolve_video_path(video_file) -> Optional[str]:
    if video_file is None:
        return None
    if isinstance(video_file, str):
        return video_file
    if isinstance(video_file, dict) and "name" in video_file:
        return video_file["name"]
    if isinstance(video_file, (list, tuple)) and len(video_file) > 0:
        return video_file[0]
    return None


def handle_index(video_file, sample_fps: int, max_frames: Optional[int]):
    video_path = _resolve_video_path(video_file)
    if not video_path:
        return "Lutfen bir video yukleyin."

    count = index_video(
        video_path=video_path,
        sample_fps=sample_fps,
        max_frames=max_frames if max_frames and max_frames > 0 else None,
    )
    return f"{count} kare indekslendi. DB: {LANCEDB_DIR}"


def handle_search(query: str, top_k: int):
    if not Path(LANCEDB_DIR).exists():
        return None, "Once bir video indeksle."

    try:
        searcher = VideoSearcher()
        results = searcher.search(query, top_k=top_k)
    except Exception:
        return None, "Indeks bulunamadi. Once video indeksle."
    if not results:
        return None, "Sonuc bulunamadi."

    best = results[0]
    info_lines = []
    for idx, item in enumerate(results, start=1):
        ts = format_timestamp(item["timestamp"])
        info_lines.append(f"{idx}. {ts} - {item['caption']}")

    info = "\n".join(info_lines)
    return best["frame_path"], info


with gr.Blocks(title="Chat with Video") as demo:
    gr.Markdown("# Chat with Video: Multimodal Video Search Engine")

    with gr.Tab("Indeksleme"):
        video_input = gr.Video(label="Video yukle")
        fps_input = gr.Slider(1, 5, value=DEFAULT_SAMPLE_FPS, step=1, label="Sample FPS")
        max_frames_input = gr.Number(value=None, label="Max frame (opsiyonel)")
        index_btn = gr.Button("Indeksle")
        index_status = gr.Textbox(label="Durum")

        index_btn.click(
            handle_index,
            inputs=[video_input, fps_input, max_frames_input],
            outputs=index_status,
        )

    with gr.Tab("Sorgu"):
        query_input = gr.Textbox(
            label="Soru",
            placeholder="Orn: Kirmizi sapkali adam ne zaman kadraja girdi?",
        )
        top_k_input = gr.Slider(1, 5, value=3, step=1, label="Top K")
        search_btn = gr.Button("Ara")
        result_image = gr.Image(label="En iyi kare")
        result_info = gr.Textbox(label="Sonuclar")

        search_btn.click(
            handle_search,
            inputs=[query_input, top_k_input],
            outputs=[result_image, result_info],
        )


if __name__ == "__main__":
    demo.launch()
