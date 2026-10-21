from src.core.models import Document, DocumentMetadata
from src.document_processor.chunker import TextChunker

def run_test():
    # ۱. ساخت یک داکیومنت فرضی با متادیتا
    dummy_text = (
        "This is a sample company policy. " * 20 + 
        "Only managers can approve remote work. " * 10
    )
    
    metadata = DocumentMetadata(
        source_file="hr_policy_test.txt",
        allowed_roles=["employee", "manager"]
    )
    
    doc = Document(content=dummy_text, metadata=metadata)
    
    # ۲. راه‌اندازی Chunker با سایز کوچک برای تست
    chunker = TextChunker(chunk_size=150, chunk_overlap=20)
    
    # ۳. قطعه‌بندی متن
    chunks = chunker.split_document(doc)
    
    # ۴. نمایش نتایج
    print(f"Total chunks created: {len(chunks)}\n")
    for i, chunk in enumerate(chunks[:3]): # چاپ ۳ قطعه اول
        print(f"--- Chunk {i+1} (ID: {chunk.id[:8]}...) ---")
        print(f"Text length: {len(chunk.text)}")
        print(f"Allowed roles: {chunk.metadata.allowed_roles}")
        print(f"Content: {chunk.text}\n")

if __name__ == "__main__":
    run_test()