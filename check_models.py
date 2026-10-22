# این اسکریپت به سرور Groq متصل می‌شود و لیست تمام مدل‌های فعال را می‌گیرد
from openai import OpenAI
from src.core.settings import settings

def list_active_models():
    client = OpenAI(
        api_key=settings.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )
    
    print("Fetching available models from Groq...\n")
    models = client.models.list()
    
    for model in models.data:
        print(f"- {model.id}")

if __name__ == "__main__":
    list_active_models()