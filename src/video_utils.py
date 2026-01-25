from pathlib import Path
from typing import List, Dict, Optional

import cv2


def extract_frames(
    video_path: str,
    output_dir: Path,
    sample_fps: int = 1,
    max_frames: Optional[int] = None,
) -> List[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Video acilamadi: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_fps = max(sample_fps, 1)
    frame_interval = max(int(round(native_fps / sample_fps)), 1)

    frames = []
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / native_fps
            frame_path = output_dir / f"frame_{frame_idx:08d}.jpg"
            cv2.imwrite(str(frame_path), frame)

            frames.append(
                {
                    "timestamp": float(timestamp),
                    "frame_path": str(frame_path),
                }
            )
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames
