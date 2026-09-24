# app/models.py
import hashlib
from enum import Enum
from pydantic import BaseModel
from typing import Optional, List, Literal


class ParserProvenance(BaseModel):
    name: Literal["anydoc"]
    version: str


class ExtractionResult(BaseModel):
    profile: Literal["document-v1"]
    text: str
    format: Literal["markdown"]
    completeness: Literal["complete", "partial"]
    may_omit_content: bool
    pages_needing_ocr: List[int]
    truncated: bool
    parser: ParserProvenance


class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict


class DocumentModel(BaseModel):
    page_content: str
    metadata: Optional[dict] = {}

    def generate_digest(self):
        hash_obj = hashlib.md5(self.page_content.encode())
        return hash_obj.hexdigest()


class StoreDocument(BaseModel):
    filepath: str
    filename: str
    file_content_type: str
    file_id: str


class QueryRequestBody(BaseModel):
    query: str
    file_id: str
    k: int = 4
    entity_id: Optional[str] = None


class CleanupMethod(str, Enum):
    incremental = "incremental"
    full = "full"


class QueryMultipleBody(BaseModel):
    query: str
    file_ids: List[str]
    k: int = 4
    entity_id: Optional[str] = None
