import numpy as np
import json
import os
from typing import List, Tuple
from src.core.models import Chunk

class LocalPersistentVectorStore:
    def __init__(self, storage_dir: str = "data/vector_store", embedding_dim: int = 384):
        self.storage_dir = storage_dir
        self.chunks_file = os.path.join(storage_dir, "chunks.json")
        self.embeddings_file = os.path.join(storage_dir, "embeddings.npy")
        self.embedding_dim = embedding_dim
        
        # ساخت پوشه در صورت عدم وجود
        os.makedirs(storage_dir, exist_ok=True)
        
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray = np.empty((0, embedding_dim))
        
        # خواندن داده‌ها از هارد در زمان مقداردهی اولیه
        self._load_from_disk()

    def _load_from_disk(self):
        if os.path.exists(self.chunks_file) and os.path.exists(self.embeddings_file):
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
                self.chunks = [Chunk(**data) for data in chunks_data]
            
            self.embeddings = np.load(self.embeddings_file)
            print(f"✅ Loaded {len(self.chunks)} chunks from local disk ({self.storage_dir}).")

    def _save_to_disk(self):
        with open(self.chunks_file, "w", encoding="utf-8") as f:
            json.dump([chunk.model_dump() for chunk in self.chunks], f, ensure_ascii=False, indent=2)
        
        np.save(self.embeddings_file, self.embeddings)

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        self.chunks.extend(chunks)
        if self.embeddings.shape[0] == 0:
            self.embeddings = np.array(embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        # ذخیره خودکار روی هارد بعد از هر الحاق
        self._save_to_disk()

    def search(self, query_embedding: List[float], top_k: int = 2, allowed_roles: List[str] = None, score_threshold: float = 0.3) -> List[Tuple[Chunk, float]]:
        if len(self.chunks) == 0:
            return []
            
        q_vec = np.array(query_embedding)
        similarities = np.dot(self.embeddings, q_vec)
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            chunk = self.chunks[idx]
            
            if score < score_threshold:
                break
                
            if allowed_roles:
                has_access = any(role in chunk.metadata.allowed_roles for role in allowed_roles)
                if not has_access:
                    continue
                    
            results.append((chunk, score))
            if len(results) == top_k:
                break
                
        return results