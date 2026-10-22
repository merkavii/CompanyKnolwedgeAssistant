from src.core.models import Document, DocumentMetadata
from src.document_processor.chunker import TextChunker
from src.retrieval.embedder import DocumentEmbedder
from src.retrieval.vector_store import InMemoryVectorStore
from src.generation.generator import AnswerGenerator

def run_test():
    print("Initializing components (Embedding model, Vector DB, LLM Generator)...")
    
    # نمونه‌سازی از تمام کلاس‌های پایه‌ای که تا الان ساختیم
    embedder = DocumentEmbedder()
    vector_store = InMemoryVectorStore()
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    generator = AnswerGenerator() # اینجا API Key به صورت خودکار از Settings خوانده می‌شود
    
    # ۱. شبیه‌سازی اسناد شرکت و متادیتاها (ایجاد دیتابیس دانشی)
    doc1 = Document(
        content="Employees can take up to 3 days of paid emergency leave per year. Manager approval is required in writing.",
        metadata=DocumentMetadata(source_file="hr_leave_policy_2026.pdf", allowed_roles=["employee", "manager"])
    )
    doc2 = Document(
        content="The server root password is 'Admin@123!'. This is strictly for the DevOps team.",
        metadata=DocumentMetadata(source_file="it_security.txt", allowed_roles=["devops"])
    )
    
    # ۲. تبدیل اسناد به Chunk (قطعه‌بندی)
    all_chunks = chunker.split_document(doc1) + chunker.split_document(doc2)
    chunk_texts = [chunk.text for chunk in all_chunks]
    
    print("Embedding chunks and storing in Vector Database...")
    # ۳. تبدیل متن‌ها به بردار ریاضی و ذخیره در دیتابیس حافظه‌ای
    embeddings = embedder.embed_batch(chunk_texts)
    vector_store.add_chunks(all_chunks, embeddings)
    
    # ۴. تابع کمکی برای شبیه‌سازی کامل یک درخواست از سمت کاربر
    def ask_question(query: str, user_roles: list[str]):
        print(f"\n[User Role: {user_roles}] Query: {query}")
        
        # الف: تبدیل سوال کاربر به بردار
        query_emb = embedder.embed_text(query)
        
        # ب: جست‌وجو در دیتابیس (فیلتر شده بر اساس نقش کاربر و Threshold)
        retrieved_chunks = vector_store.search(
            query_embedding=query_emb, 
            top_k=2, 
            allowed_roles=user_roles,
            score_threshold=0.3  # فقط نتایج مرتبط را برمی‌گرداند
        )
        
        # ج: ارسال سوال و نتایج فیلترشده به مدل زبانی (LLM) برای تولید پاسخ انسانی
        answer = generator.generate_answer(query, [chunk for chunk, score in retrieved_chunks])
        print(f"Assistant: {answer}")

    print("\n" + "="*50)
    
    # سناریو ۱: کارمند عادی درباره مرخصی می‌پرسد (باید جواب دقیق با Citation فایل بدهد)
    ask_question("How many days of emergency leave can I take?", ["employee"])
    
    # سناریو ۲: کارمند عادی رمز سرور را می‌پرسد (فیلتر RBAC سند را مسدود می‌کند، LLM باید بگوید نمی‌دانم)
    ask_question("What is the server root password?", ["employee"])
    
    # سناریو ۳: سوالی که اصلا در اسناد نیست (فیلتر Threshold اسناد نامرتبط را حذف می‌کند، LLM باید بگوید پیدا نکردم)
    ask_question("Does the company provide free lunch?", ["employee"])

if __name__ == "__main__":
    run_test()