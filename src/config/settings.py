from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RCO AI"
    app_description: str = "Predictive healthcare revenue cycle operations intelligence."
    version: str = "0.1.0"
    environment: str = "development"

    data_directory: str = "data"
    model_directory: str = "models"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()