import json
from src.core.models import Document, DocumentMetadata
from src.document_processor.chunker import TextChunker
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import LocalPersistentVectorStore

def load_knowledge_base(chunker, embedder, vector_store):
    # این تابع دیتابیس ما را با اسناد تستی پر می‌کند
    docs = [
        Document(
            content="Employees can take up to 3 days of paid emergency leave per year. Manager approval is required in writing.",
            metadata=DocumentMetadata(source_file="hr_leave_policy_2026.pdf", allowed_roles=["employee", "manager"])
        ),
        Document(
            content="Standard working hours are 9 AM to 5 PM. Only managers can approve remote work.",
            metadata=DocumentMetadata(source_file="hr_policy_test.txt", allowed_roles=["employee", "manager"])
        ),
        Document(
            content="The server root password is 'Admin@123!'. This is strictly for the DevOps team.",
            metadata=DocumentMetadata(source_file="it_security.txt", allowed_roles=["devops", "manager"])
        )
    ]
    
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.split_document(doc))
        
    chunk_texts = [chunk.text for chunk in all_chunks]
    embeddings = embedder.embed_batch(chunk_texts)
    vector_store.add_chunks(all_chunks, embeddings)
    print(f"Knowledge base loaded with {len(all_chunks)} chunks.\n")

def evaluate_retrieval():
    print("Initializing Evaluation System...")
    embedder = DocumentEmbedder()
    vector_store = LocalPersistentVectorStore()
    chunker = TextChunker(chunk_size=150, chunk_overlap=20)
    
    load_knowledge_base(chunker, embedder, vector_store)
    
    # 1. خواندن دیتاست ارزیابی (Golden Dataset)
    with open("data/evaluation/qa_dataset.json", "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    successful_retrievals = 0
    total_questions = len(eval_data)
    
    print("-" * 40)
    
    # 2. تست کردن هر سوال
    for item in eval_data:
        query = item["question"]
        expected_file = item["expected_source_file"]
        required_snippet = item["required_text_snippet"].lower()
        roles = item["user_roles"]
        
        query_emb = embedder.embed_text(query)
        
        # ما Recall@2 را می‌سنجیم (آیا جواب در ۲ نتیجه‌ی اول هست؟)
        results = vector_store.search(query_emb, top_k=2, allowed_roles=roles, score_threshold=0.3)
        
        # بررسی اینکه آیا منبع درست بازیابی شده است یا خیر
        match_found = False
        for chunk, score in results:
            if expected_file == chunk.metadata.source_file and required_snippet in chunk.text.lower():
                match_found = True
                break
                
        if match_found:
            print(f"✅ PASS: {query}")
            successful_retrievals += 1
        else:
            print(f"❌ FAIL: {query}")
            
    # 3. محاسبه و نمایش متریک نهایی
    recall_at_k = (successful_retrievals / total_questions) * 100
    print("-" * 40)
    print(f"Metrics Report:")
    print(f"Total Questions: {total_questions}")
    print(f"Retrieval Recall@2: {recall_at_k:.2f}%")

if __name__ == "__main__":
    evaluate_retrieval()