import os
from openai import OpenAI
from typing import List
from src.core.models import Chunk

class AnswerGenerator:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        # کلید API به صورت خودکار از متغیرهای محیطی یا فایل .env خوانده می‌شود
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
            
        self.client = OpenAI(api_key=api_key)
        self.model = model_name

    def generate_answer(self, query: str, retrieved_chunks: List[Chunk]) -> str:
        # اگر در مرحله Retrieval هیچ سندی پیدا نشد، اصلاً به LLM درخواست نمی‌دهیم
        if not retrieved_chunks:
            return "I cannot answer this question because I don't have access to relevant documents or no matching information was found."

        # آماده‌سازی Context از روی Chunkهای بازیابی‌شده
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks):
            # نام فایل را صریحاً به مدل می‌دهیم تا بتواند Citation کند
            context_parts.append(f"--- Document {i+1} ---\n{chunk.text}\n")
        
        context_text = "\n".join(context_parts)

        # Prompt مهندسی‌شده برای جلوگیری از توهم
        system_prompt = (
            "You are a highly strictly professional Company Knowledge Assistant. "
            "Your task is to answer the user's question ONLY using the provided context documents. "
            "Follow these rules strictly:\n"
            "1. If the answer cannot be deduced from the provided context, explicitly output exactly: 'I could not find the answer in the provided documents.' Do not guess.\n"
            "2. Always include inline citations referring to the source file name (e.g.,).\n"
            "3. Do not use outside knowledge."
        )

        user_prompt = f"Context documents:\n{context_text}\n\nQuestion: {query}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0 # روی صفر تنظیم می‌کنیم تا کمترین توهم را داشته باشد
        )
        
        return response.choices[0].message.content