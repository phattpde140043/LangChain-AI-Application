from __future__ import annotations
"""GET/DELETE /knowledge-base endpoints."""

from fastapi import APIRouter, HTTPException, Request

from ..schemas import KnowledgeBaseStats
from ...monitoring.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/knowledge-base", response_model=KnowledgeBaseStats)
async def get_knowledge_base(request: Request) -> KnowledgeBaseStats:
    """Get statistics about the knowledge base."""
    try:
        from ...ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        docs = pipeline.get_indexed_documents()
        
        return KnowledgeBaseStats(
            documents=docs,
            total_documents=len(docs),
        )
    
    except Exception as e:
        logger.error(f"Knowledge base stats error: {e}")
        return KnowledgeBaseStats(documents=[], total_documents=0)


@router.delete("/knowledge-base/{document_name}")
async def delete_document(document_name: str) -> dict:
    """Delete a document from the knowledge base."""
    try:
        from ...ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline()
        success = pipeline.delete_document(document_name)
        
        if success:
            return {"status": "deleted", "document_name": document_name}
        else:
            raise HTTPException(status_code=404, detail=f"Document not found: {document_name}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete document error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
