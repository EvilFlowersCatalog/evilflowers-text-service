# Testing TextHandler + Milvus Local Integration

## Overview

You now have complete integration between your TextHandler (TextExtractor) and Milvus local database for semantic search.

## Changes Made

### 1. Updated SemanticIndexer
**File:** [src/semantic/indexer.py](src/semantic/indexer.py)

- Updated `index_document()` method to accept the new dict-based format from TextExtractor
- Changed signature from `index_document(document_id, pages, paragraphs)` to `index_document(document_id, pages)`
- Updated `_prepare_chunks()` to work with the new format where `pages` is a list of dicts:
  ```python
  {
      'page_num': int,
      'text': str,              # Full page text
      'paragraphs': List[str],  # List of paragraphs
      'sentences': List[str]    # List of sentences
  }
  ```

## How to Use

### Basic Usage in Your Code

```python
from text_handler.TextService import TextHandler
from semantic.indexer import SemanticIndexer

# 1. Extract text from PDF
text_handler = TextHandler("path/to/document.pdf")
pages, toc = text_handler.extract_text(found_toc=False)

# 2. Index for semantic search
indexer = SemanticIndexer()
result = indexer.index_document(
    document_id="unique_doc_id",
    pages=pages  # Just pass pages - it contains paragraphs inside!
)

# 3. Search
from semantic.search import SemanticSearch
search = SemanticSearch()
results = search.search("your query here", top_k=5)
```

## Running Tests

### Test 1: With Sample Data (No PDF needed)
This is the fastest way to test if everything works:

```bash
python test_text_processor_milvus.py --sample
```

### Test 2: With Your Own PDF
```bash
python test_text_processor_milvus.py /path/to/your/document.pdf
```

### Test 3: Interactive Mode
```bash
python test_text_processor_milvus.py
```
Then choose option 1 (sample) or 2 (PDF).

### Test 4: Updated Dictionary Test
This tests with sample Slovak algebra text:

```bash
python test_milvus_local.py
```

## Environment Variables

The tests automatically set these variables, but for production use:

```bash
# Use embedded Milvus (no server needed)
export USE_MILVUS_LITE=true
export MILVUS_LITE_DB=./milvus_lite.db

# Semantic search settings
export SEMANTIC_SEARCH_ENABLED=true
export CHUNK_LEVEL=paragraph  # or 'page'
export EMBEDDING_DEVICE=cpu    # or 'cuda' for GPU
export EMBEDDING_BATCH_SIZE=8

# Collection name
export MILVUS_COLLECTION_NAME=document_embeddings
```

## What Each Test Does

### test_text_processor_milvus.py
**Complete integration test** that:
1. ✓ Extracts text from PDF using TextHandler
2. ✓ Generates embeddings using sentence-transformers
3. ✓ Stores vectors in Milvus Local (embedded database)
4. ✓ Tests semantic search with queries
5. ✓ Shows collection statistics

### test_milvus_local.py
**Quick test** with hardcoded Slovak algebra text - good for:
- Testing without PDF files
- Verifying semantic search works
- Cross-lingual search testing (Slovak ↔ English)

## Expected Output

When you run the tests successfully, you should see:

```
================================================================================
TEXT HANDLER + MILVUS LOCAL INTEGRATION TEST
================================================================================

STEP 1: Extracting text from PDF using TextHandler
✓ TextHandler initialized
✓ Text extraction complete!
   Total pages: 150
   Total paragraphs: 450

STEP 2: Initializing Semantic Indexer with Milvus Local
✓ SemanticIndexer initialized
   Using Milvus Lite (embedded mode)

STEP 3: Indexing document for semantic search
✓ Indexing successful!
   Chunks indexed: 450
   Chunk level: paragraph

STEP 4: Testing Semantic Search
✓ SemanticSearch initialized
🔍 Query: 'your search term'
   Found 3 results:
   Result 1:
      Score: 0.8523
      Page: 42
      Text: [matching paragraph text]...

✓ INTEGRATION TEST COMPLETED SUCCESSFULLY!
```

## Troubleshooting

### Error: "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### Error: "No module named 'pymilvus'"
```bash
pip install pymilvus
```

### Error: PDF file not found
Make sure you provide the full path to your PDF file.

### Slow on first run
The first run downloads the multilingual embedding model (~500MB). Subsequent runs are much faster.

## Integration with Your API

To add semantic search to your FastAPI endpoint:

```python
from semantic.indexer import SemanticIndexer
from semantic.search import SemanticSearch

@app.post("/process_text")
async def process_text(file: UploadFile = File(...)):
    # ... existing code to save file ...

    # Extract text
    text_handler = TextHandler(temp_path)
    pages, toc = text_handler.extract_text(found_toc=False)

    # Index for semantic search
    indexer = SemanticIndexer()
    document_id = str(uuid.uuid4())

    result = indexer.index_document(
        document_id=document_id,
        pages=pages
    )

    return {
        "document_id": document_id,
        "pages": len(pages),
        "chunks_indexed": result['chunks_indexed']
    }

@app.get("/search")
async def search(query: str, top_k: int = 5):
    search = SemanticSearch()
    results = search.search(query, top_k=top_k)
    return {"results": results}
```

## Next Steps

1. ✓ **Test locally** - Run the tests to verify everything works
2. **Integrate into API** - Add semantic indexing to your FastAPI endpoints
3. **Connect to Kafka** - Index documents as they come through Kafka
4. **Production deployment** - Switch to Milvus server for scalability

## Files Modified/Created

- ✓ `src/semantic/indexer.py` - Updated to accept new dict format
- ✓ `test_text_processor_milvus.py` - New comprehensive test
- ✓ `test_milvus_local.py` - Updated to use new format
- ✓ `example_semantic_indexing.py` - Simple usage example
