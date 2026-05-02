"""
FastAPI Application - Alternative to Streamlit for production deployment
RESTful API for NER pipeline with OpenAPI documentation
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from ml_models.ner_pipeline import HybridNERPipeline
from utils import batch_extract, export_results
import json
from pathlib import Path


# Pydantic models for request/response
class TextRequest(BaseModel):
    """Request body for single text extraction"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to extract entities from")
    enable_crf: bool = Field(default=False, description="Use CRF model if available")


class BatchRequest(BaseModel):
    """Request body for batch extraction"""
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of texts")
    batch_size: int = Field(default=32, ge=1, le=100)


class Entity(BaseModel):
    """Single extracted entity"""
    text: str
    label: str
    start: int


class ExtractionResponse(BaseModel):
    """Response with extracted entities"""
    entities: List[Entity]
    total_entities: int
    by_label: Dict[str, int]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str = "1.0"
    models_available: List[str]


# Initialize FastAPI app
app = FastAPI(
    title="NER Pipeline API",
    description="Hybrid Named Entity Recognition API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline (global, reused across requests)
_ner_pipeline: Optional[HybridNERPipeline] = None
_crf_model_path: str = "./models/crf_ner_model.pkl"


def get_pipeline(enable_crf: bool = False) -> HybridNERPipeline:
    """Get or initialize NER pipeline"""
    global _ner_pipeline
    
    if _ner_pipeline is None:
        model_path = _crf_model_path if enable_crf and Path(_crf_model_path).exists() else None
        _ner_pipeline = HybridNERPipeline(
            crf_model_path=model_path,
            enable_crf=bool(model_path)
        )
    
    return _ner_pipeline


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    models = []
    if Path(_crf_model_path).exists():
        models.append("CRF")
    models.extend(["Regex", "spaCy"])
    
    return HealthResponse(
        status="healthy",
        models_available=models
    )


@app.post("/extract", response_model=ExtractionResponse, tags=["Extraction"])
async def extract_entities(request: TextRequest):
    """
    Extract entities from text
    
    **Parameters:**
    - `text`: Input text to analyze (1-10000 characters)
    - `enable_crf`: Whether to use CRF model (default: false)
    
    **Returns:**
    - `entities`: List of extracted entities with text, label, and position
    - `total_entities`: Count of extracted entities
    - `by_label`: Count breakdown by entity type
    """
    try:
        pipeline = get_pipeline(enable_crf=request.enable_crf)
        result = pipeline.extract_with_metadata(request.text)
        
        return ExtractionResponse(
            entities=result["entities"],
            total_entities=result["stats"]["total_entities"],
            by_label=result["stats"]["by_label"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-batch", tags=["Extraction"])
async def extract_batch(request: BatchRequest):
    """
    Extract entities from multiple texts
    
    **Parameters:**
    - `texts`: List of texts (1-100 items)
    - `batch_size`: Processing batch size (default: 32)
    
    **Returns:**
    - List of extraction results with entities and statistics
    """
    try:
        pipeline = get_pipeline(enable_crf=False)  # Faster for batch
        results = batch_extract(pipeline, request.texts, batch_size=request.batch_size)
        return {"results": results, "total_processed": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-file", tags=["Extraction"])
async def extract_from_file(file: UploadFile = File(...)):
    """
    Extract entities from uploaded text file
    
    **Parameters:**
    - `file`: Text file to upload (.txt)
    
    **Returns:**
    - Extraction results with entities
    """
    try:
        content = await file.read()
        text = content.decode("utf-8")
        
        pipeline = get_pipeline(enable_crf=False)
        result = pipeline.extract_with_metadata(text)
        
        return ExtractionResponse(
            entities=result["entities"],
            total_entities=result["stats"]["total_entities"],
            by_label=result["stats"]["by_label"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", tags=["System"])
async def list_models():
    """List available NER models"""
    return {
        "extractors": ["Regex", "spaCy"],
        "crf_available": Path(_crf_model_path).exists(),
        "crf_path": _crf_model_path if Path(_crf_model_path).exists() else "not found"
    }


@app.post("/config", tags=["System"])
async def configure_model(crf_path: str):
    """Configure CRF model path"""
    global _crf_model_path, _ner_pipeline
    
    if not Path(crf_path).exists():
        raise HTTPException(status_code=400, detail=f"Model path not found: {crf_path}")
    
    _crf_model_path = crf_path
    _ner_pipeline = None  # Reset pipeline
    
    return {"status": "configured", "crf_path": crf_path}


# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "name": "NER Pipeline API",
        "version": "1.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "extract": "/extract",
            "extract_batch": "/extract-batch",
            "extract_file": "/extract-file",
            "models": "/models"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
