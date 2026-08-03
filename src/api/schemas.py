"""HTTP request and response contracts for the data engineering API."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PresignedFileRequest(BaseModel):
    """One S3 object address supplied by the calling platform."""

    model_config = ConfigDict(extra="ignore")

    key: str = Field(min_length=1)
    presigned_url: str = Field(min_length=1)
    size: int | None = Field(default=None, ge=0)
    etag: str | None = None

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("key must not be blank")
        return value

    @field_validator("presigned_url")
    @classmethod
    def validate_presigned_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("presigned_url must be a valid HTTP(S) URL")
        # Preserve the exact signed value; normalizing it can invalidate the signature.
        return value


class DataEngRequest(BaseModel):
    """Batch of presigned S3 objects to process in one pipeline run."""

    model_config = ConfigDict(extra="ignore")

    bucket: str = Field(min_length=1)
    files: list[PresignedFileRequest] = Field(min_length=1)

    @field_validator("bucket")
    @classmethod
    def validate_bucket(cls, value: str) -> str:
        bucket = value.strip()
        if not bucket:
            raise ValueError("bucket must not be blank")
        return bucket

    def to_inventory(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class DataEngResponse(BaseModel):
    """Common run metadata plus stage-oriented document results."""

    metadata: dict[str, Any]
    documents: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
