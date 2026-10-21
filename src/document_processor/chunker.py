import uuid
from typing import List
from src.core.models import Document, Chunk

class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, doc: Document) -> List[Chunk]:
        chunks = []
        text = doc.content
        start = 0
        text_length = len(text)

        # اگر متن خالی باشد، لیست خالی برمی‌گرداند
        if not text.strip():
            return chunks

        while start < text_length:
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            chunk = Chunk(
                id=str(uuid.uuid4()),
                text=chunk_text.strip(),
                metadata=doc.metadata
            )
            chunks.append(chunk)
            
            # حرکت به جلو با در نظر گرفتن هم‌پوشانی (Overlap)
            start += (self.chunk_size - self.chunk_overlap)
            
        return chunks