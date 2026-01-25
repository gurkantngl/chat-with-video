from pathlib import Path
from typing import Optional
import uuid

import lancedb
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import (
    FRAMES_DIR,
    LANCEDB_DIR,
    EMBED_MODEL_NAME,
    DEFAULT_SAMPLE_FPS,
)
from .video_utils import extract_frames
from .vision_captioner import FlorenceCaptioner


def index_video(
    video_path: str,
    sample_fps: int = DEFAULT_SAMPLE_FPS,
    max_frames: Optional[int] = None,
    frames_dir: Path = FRAMES_DIR,
    db_dir: Path = LANCEDB_DIR,
) -> int:
    frames = extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        sample_fps=sample_fps,
        max_frames=max_frames,
    )

    captioner = FlorenceCaptioner()
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    records = []
    for frame in tqdm(frames, desc="Caption ve embedding", unit="frame"):
        caption = captioner.caption(frame["frame_path"])
        vector = embedder.encode(caption, normalize_embeddings=True)
        vector = np.asarray(vector, dtype=np.float32)

        records.append(
            {
                "id": str(uuid.uuid4()),
                "timestamp": frame["timestamp"],
                "frame_path": frame["frame_path"],
                "caption": caption,
                "vector": vector,
            }
        )

    db = lancedb.connect(str(db_dir))
    db.create_table("frames", data=records, mode="overwrite")

    return len(records)
