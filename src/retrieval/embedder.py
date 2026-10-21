from sentence_transformers import SentenceTransformer
from typing import List

class DocumentEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        # این مدل دفعه اول حدود 130 مگابایت دانلود می‌شود
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        # پارامتر normalize_embeddings باعث می‌شود طول بردار ۱ شود، که محاسبه شباهت را بسیار سریع‌تر می‌کند
        embeddings = self.model.encode(text, normalize_embeddings=True)
        return embeddings.tolist()
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()