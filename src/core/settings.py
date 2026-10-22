from pydantic_settings import BaseSettings, SettingsConfigDict

# کلاسی برای تعریف تمام متغیرهای محیطی سیستم
# اگر متغیری در .env نباشد، Pydantic در همان لحظه‌ی اول خطای واضحی می‌دهد
class Settings(BaseSettings):
    GROQ_API_KEY: str
    
    # به Pydantic می‌گوییم فایل .env را از مسیر اصلی پروژه بخواند
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# ساخت یک نمونه‌ی یکتا (Singleton) برای استفاده در تمام بخش‌های پروژه
settings = Settings()