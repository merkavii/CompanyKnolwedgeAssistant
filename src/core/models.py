from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentMetadata(BaseModel):
    source_file: str
    allowed_roles: List[str] = Field(default_factory=lambda: ["employee"])
    # در آینده می‌توانیم شماره صفحه یا تاریخ را هم اینجا اضافه کنیم

class Chunk(BaseModel):
    id: str
    text: str
    metadata: DocumentMetadata

class Document(BaseModel):
    content: str
    metadata: DocumentMetadata