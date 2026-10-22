from openai import OpenAI
from typing import List
from src.core.models import Chunk
from src.core.settings import settings # اضافه شدن مدیریت متمرکز تنظیمات

class AnswerGenerator:
    def __init__(self, model_name: str = "qwen/qwen3.8-27b"):
        # خواندن API Key به صورت Type-Safe از کلاس Settings
        self.client = OpenAI(
            api_key=settings.GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = model_name

    def generate_answer(self, query: str, retrieved_chunks: List[Chunk]) -> str:
        # بررسی وجود داکیومنت قبل از ارسال به مدل
        if not retrieved_chunks:
            return "I could not find the answer in the provided documents or you do not have permission to view them."

        # ساخت ساختار متنی از قطعات بازیابی‌شده به همراه نام فایل برای ارجاع‌دهی
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            context_parts.append(f"--- Document: {chunk.metadata.source_file} ---\n{chunk.text}\n")
        
        context_text = "\n".join(context_parts)

        # پرامپت سیستمی برای محدود کردن مدل به متن و جلوگیری از توهم
        system_prompt = (
            "You are a highly strictly professional Company Knowledge Assistant. "
            "Your task is to answer the user's question ONLY using the provided context documents. "
            "Follow these rules strictly:\n"
            "1. If the answer cannot be deduced from the provided context, explicitly output exactly: 'I could not find the answer in the provided documents.' Do not guess.\n"
            "2. Always include inline citations referring to the exact source file name provided in the context.\n"
            "3. Do not use any outside knowledge."
        )

        user_prompt = f"Context documents:\n{context_text}\n\nQuestion: {query}"

        # فراخوانی API با دمای صفر برای دریافت قطعی‌ترین پاسخ
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        return response.choices[0].message.content