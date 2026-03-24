# Search Test Script Summary

This script simulates **real library users** searching across all 5 indexed documents.

## What It Does

### **Test 1: Random User Queries** (Semantic Search)
- Picks 5 random queries from realistic search terms
- Examples: 
  - "neurónové siete"
  - "ako nakonfigurovať smerovač"
  - "LaTeX tabuľky"
- Searches across **ALL documents** (not filtered by document_id)
- Shows top 3 results with:
  - Document name
  - Page number
  - Relevance score
  - Text preview

### **Test 2: Highlighted Text Snippets**
- Simulates users **copy-pasting text** from PDFs to search
- Examples: 
  - "konvolučné neurónové siete sú typ hlbokého učenia"
  - "IP adresa jednoznačne identifikuje zariadenie v sieti"
- Tests if semantic search can find similar content even with exact text
- Shows which documents contain similar content

### **Test 3: Semantic vs Elasticsearch Comparison**
- Runs **same query** through both search methods
- Compares results side-by-side for 3 queries
- Shows timing difference (which is faster)
- Helps you see when semantic search works better vs keyword search

### **Test 4: Performance Test**
- Runs 20 searches with same query
- Measures speed for both:
  - Milvus (semantic search)
  - Elasticsearch (keyword search)
- Reports metrics:
  - Average response time
  - Min response time
  - Max response time
- Helps identify performance issues

### **Test 5: Cross-Language Search**
- Tests if **Slovak and English queries** find the same documents
- Examples:
  - "neurónové siete" vs "neural networks"
  - "sieťové zariadenia" vs "network devices"
  - "matematická logika" vs "mathematical logic"
- Shows if multilingual embeddings work correctly
- Displays overlap between Slovak/English results

## Sample Output
```
Found 12 results (showing top 3):

[1] Document: umela-inteligencia-kognitivna-veda | Page: 45 | Score: 0.8523
    Neurónové siete sú základným stavebným prvkom hlbokého učenia...

[2] Document: matematicka-logika | Page: 12 | Score: 0.7234
    Logické operátory umožňujú kombinovať jednoduché výroky...

[3] Document: sietove-zariadenia | Page: 89 | Score: 0.6891
    TCP/IP protokol definuje komunikáciu medzi zariadeniami...
```

## Performance Metrics Example
```
Semantic Search (Milvus):
  Average: 45.32ms
  Min: 38.12ms
  Max: 67.89ms

Elasticsearch Search:
  Average: 23.45ms
  Min: 18.34ms
  Max: 42.11ms
```

## Key Difference

**Unlike document-specific search:** This searches the **entire library** at once, showing results from multiple books - exactly how a real search engine works!

## Usage
```bash
docker exec text-service python src/search_test.py
```

## Test Categories

| Test | Purpose | What It Validates |
|------|---------|-------------------|
| Random Queries | Realistic user searches | Search quality across documents |
| Highlighted Text | Copy-paste search | Exact match + semantic similarity |
| Method Comparison | Semantic vs Keyword | When to use which search method |
| Performance | Speed testing | Response times under load |
| Cross-Language | Multilingual search | Slovak ↔ English search equivalence |