from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Tuple, Dict
import os
import uuid
import httpx
from text_handler.TextExtractor import TextExtractor
from text_handler.TextProcessor import TextProcessor
from text_handler.TextService import TextHandler

app = FastAPI(
    title="EvilFlowers Text Service",
    description="Service for extracting and processing text from documents",
    version="1.0.0"
)

ELASTICSEARCH_SERVICE_URL = "http://elasticsearch-module:9200"

async def save_text_in_elasticsearch(file: UploadFile, content: str, extracted_text: str) -> Tuple[str, Dict]:
    # Prepare document for Elasticsearch
    document_id = str(uuid.uuid4())
    document = {
        "content": extracted_text,
        "metadata": {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
            "extraction_type": "text"
        },
        "document_id": document_id
    }
    
    # Store in Elasticsearch through API
    async with httpx.AsyncClient() as client:
        es_response = await client.post(
            f"{ELASTICSEARCH_SERVICE_URL}/index_text/documents",
            json=document
        )
        es_response.raise_for_status()
        es_result = es_response.json()
        return document_id, es_result


@app.post("/process_text")
async def process_text(file: UploadFile = File(...)):
    temp_path = None
    try:
        # Save uploaded file temporarily
        temp_path = f"temp/{file.filename}"
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text
        text_handler = TextHandler(temp_path)
        extracted_text = text_handler.extract_text()
        
        # Save text in Elasticsearch
        document_id, es_result = await save_text_in_elasticsearch(file, content, extracted_text)
        
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return {
            "processed_text": extracted_text,
            "document_id": document_id,
            "elasticsearch_result": es_result
        }
        
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


