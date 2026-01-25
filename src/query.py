from typing import List, Dict

import lancedb
from sentence_transformers import SentenceTransformer

from .config import LANCEDB_DIR, EMBED_MODEL_NAME


class VideoSearcher:
    def __init__(self, db_dir: str = str(LANCEDB_DIR)):
        self.db = lancedb.connect(db_dir)
        self.table = self.db.open_table("frames")
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_vec = self.embedder.encode(query, normalize_embeddings=True)
        results = self.table.search(query_vec).limit(top_k).to_list()

        formatted = []
        for item in results:
            formatted.append(
                {
                    "timestamp": item.get("timestamp"),
                    "frame_path": item.get("frame_path"),
                    "caption": item.get("caption"),
                    "score": item.get("_distance"),
                }
            )
        return formatted
