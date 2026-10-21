from src.core.models import Document, DocumentMetadata
from src.document_processor.chunker import TextChunker
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import InMemoryVectorStore

def run_test():
    print("Loading embedding model (this may take a moment to download on first run)...")
    embedder = DocumentEmbedder()
    vector_store = InMemoryVectorStore()
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    
    # 1. ساخت اسناد نمونه
    doc1 = Document(
        content="The company cafeteria is open from 12 PM to 2 PM. We serve vegetarian options.",
        metadata=DocumentMetadata(source_file="general_info.txt", allowed_roles=["employee", "manager"])
    )
    doc2 = Document(
        content="Project X is highly confidential. The launch date is set for November 2026.",
        metadata=DocumentMetadata(source_file="secret_project.txt", allowed_roles=["manager"]) # فقط مدیران
    )
    
    # 2. چانک کردن و تولید امبدینگ
    all_chunks = chunker.split_document(doc1) + chunker.split_document(doc2)
    chunk_texts = [chunk.text for chunk in all_chunks]
    
    print(f"Embedding {len(all_chunks)} chunks...")
    embeddings = embedder.embed_batch(chunk_texts)
    
    # 3. ذخیره در وکتور استور
    vector_store.add_chunks(all_chunks, embeddings)
    
    # 4. تست سناریوی جست‌وجو
    query = "When does the cafeteria open?"
    print(f"\nQuery: {query}")
    query_emb = embedder.embed_text(query)
    
    print("\n--- Test 1: Employee Search (Normal Access) ---")
    results = vector_store.search(query_emb, top_k=2, allowed_roles=["employee"])
    for chunk, score in results:
        print(f"Score: {score:.3f} | File: {chunk.metadata.source_file} | Text: {chunk.text}")
        
    print("\n--- Test 2: Manager Searching for Secret Project ---")
    query2 = "What is the launch date for Project X?"
    query2_emb = embedder.embed_text(query2)
    results2 = vector_store.search(query2_emb, top_k=2, allowed_roles=["manager"])
    for chunk, score in results2:
        print(f"Score: {score:.3f} | File: {chunk.metadata.source_file} | Text: {chunk.text}")
        
    print("\n--- Test 3: Employee Searching for Secret Project (Should be BLOCKED) ---")
    results3 = vector_store.search(query2_emb, top_k=2, allowed_roles=["employee"])
    if not results3:
        print("Blocked successfully! No results found for employee.")
    else:
        for chunk, score in results3:
            print(f"Score: {score:.3f} | Text: {chunk.text}")

if __name__ == "__main__":
    run_test()