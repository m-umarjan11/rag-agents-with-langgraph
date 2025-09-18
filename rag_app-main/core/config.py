"""
Configuration module for loading environment variables
"""
import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Vector Database - Pinecone only
    PINECONE_API_KEY: str = "pcsk_4mU9YH_Meg7UUUuttUCy3zeKzkGaTp1it6urT7Zfw4CFQKZZMphUZjcoYm75oTHYkknQQP"
    PINECONE_ENVIRONMENT: str = "us-east-1"  # e.g., "us-east-1-aws"
    PINECONE_INDEX_NAME: str = "rag-index"

    # Google Gemini settings
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash-lite"
    EMBEDDING_MODEL: str = "gemini-embedding-001"

    # LangGraph settings
    LANGGRAPH_CHECKPOINTER: str = "memory"  # or "postgres", "redis"

    # File processing settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Validate required settings
def validate_settings():
    """Validate that required environment variables are set"""
    required_vars = []

    if not settings.GOOGLE_API_KEY:
        required_vars.append("GOOGLE_API_KEY")

    if not settings.PINECONE_API_KEY:
        required_vars.append("PINECONE_API_KEY")

    if not settings.PINECONE_ENVIRONMENT:
        required_vars.append("PINECONE_ENVIRONMENT")

    if required_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(required_vars)}")

# Validate settings on import
try:
    validate_settings()
except ValueError as e:
    print(f"Configuration warning: {e}")
    print("Please set the required environment variables in your .env file")
    print("Required variables: GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT")
