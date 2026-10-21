import numpy as np
from typing import List, Tuple
from src.core.models import Chunk

class InMemoryVectorStore:
    def __init__(self, embedding_dim: int = 384):
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray = np.empty((0, embedding_dim))

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]):
        self.chunks.extend(chunks)
        if self.embeddings.shape[0] == 0:
            self.embeddings = np.array(embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query_embedding: List[float], top_k: int = 2, allowed_roles: List[str] = None, score_threshold: float = 0.3) -> List[Tuple[Chunk, float]]:
        if len(self.chunks) == 0:
            return []
            
        q_vec = np.array(query_embedding)
        similarities = np.dot(self.embeddings, q_vec)
        
        # مرتب‌سازی نزولی (بیشترین شباهت اول)
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            chunk = self.chunks[idx]
            
            # 1. فیلتر شباهت: اگر شباهت از حد مجاز کمتر است، چون آرایه نزولی است، بقیه هم قطعا کمتر هستند
            if score < score_threshold:
                break
                
            # 2. فیلتر امنیتی (RBAC)
            if allowed_roles:
                has_access = any(role in chunk.metadata.allowed_roles for role in allowed_roles)
                if not has_access:
                    continue
                    
            results.append((chunk, score))
            
            if len(results) == top_k:
                break
                
        return results