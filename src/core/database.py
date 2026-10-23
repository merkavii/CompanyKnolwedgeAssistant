# وظیفه: اتصال به دیتابیس، فعال‌سازی pgvector و تعریف Schema
from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from src.core.settings import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DBChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    text = Column(Text, nullable=False)
    source_file = Column(String, index=True, nullable=False)
    # استفاده از JSONB مخصوص Postgres برای جست‌وجوی بسیار سریع روی آرایه‌ها
    allowed_roles = Column(JSONB, nullable=False)
    # تعریف ستون برداری با ابعاد 384 (مطابق خروجی مدل BAAI/bge-small)
    embedding = Column(Vector(384))