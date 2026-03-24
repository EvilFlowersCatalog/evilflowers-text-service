# search_test.py
import requests
import time
import random

SEARCH_SERVICE_URL = "http://search-service:8001"

# Realistic user queries - what actual students/researchers would search for
user_queries = [
    # General academic queries
    "neurónové siete",
    "machine learning algoritmy",
    "TCP/IP protokol",
    "matematické vzorce",
    "LaTeX tabuľky",
    
    # Specific technical queries
    "ako nakonfigurovať smerovač",
    "logické operátory",
    "eye tracking metódy",
    "kognitívne procesy",
    "predikátová logika",
    
    # Cross-domain queries
    "artificial intelligence",
    "network topology",
    "statistical analysis",
    "bibliography formatting",
    "visual attention mechanisms",
    
    # Longer natural queries (how users actually search)
    "ako vytvoriť tabuľku v LaTeXu",
    "čo je výroková logika",
    "princípy strojového učenia",
    "konfigurácia sieťových zariadení",
    "metódy sledovania pohybu očí",
    
    # Mixed language queries
    "neural network learning",
    "sieťové protokoly",
    "mathematical logic examples",
    "LaTeX equation syntax",
    "cognitive science overview",
]

# Simulated highlighted text snippets (random excerpts users might copy)
highlighted_snippets = [
    "konvolučné neurónové siete sú typ hlbokého učenia",
    "IP adresa jednoznačne identifikuje zariadenie v sieti",
    "výroková logika pracuje s výrokmi ktoré môžu byť pravdivé alebo nepravdivé",
    "LaTeX poskytuje príkazy pre sadzbu matematických vzorcov",
    "eye tracking zariadenia merajú smer a trvanie pohľadu",
    "kognitívna veda skúma mentálne procesy",
    "strojové učenie je podoblasť umelej inteligencie",
    "smerovač prepája rôzne siete a riadi tok dát",
    "predikátová logika rozširuje výrokovú logiku o kvantifikátory",
    "visual attention determines what information is processed",
]

def test_semantic_search(query, top_k=5):
    """Test semantic search across ALL documents"""
    payload = {
        "query": query,
        "top_k": top_k
    }
    
    try:
        response = requests.post(
            f"{SEARCH_SERVICE_URL}/search/semantic",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def test_elasticsearch_search(query, top_k=5):
    """Test Elasticsearch search across ALL documents"""
    payload = {
        "query": query,
        "top_k": top_k
    }
    
    try:
        response = requests.post(
            f"{SEARCH_SERVICE_URL}/search/elasticsearch",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_health():
    """Get search service health stats"""
    try:
        response = requests.get(f"{SEARCH_SERVICE_URL}/health")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def print_search_results(result, query, search_type, max_results=3):
    """Pretty print search results"""
    print(f"\n  [{search_type}] Query: '{query}'")
    print(f"  {'-'*76}")
    
    if 'error' in result:
        print(f"    ERROR: {result['error']}")
        return
    
    results = result.get('results', [])
    total = result.get('total_results', len(results))
    
    if not results:
        print("    No results found")
        return
    
    print(f"    Found {total} results (showing top {min(max_results, len(results))}):\n")
    
    # Group results by document
    docs_found = {}
    for hit in results:
        doc_id = hit.get('document_id', 'unknown')
        if doc_id not in docs_found:
            docs_found[doc_id] = []
        docs_found[doc_id].append(hit)
    
    # Show results
    shown = 0
    for doc_id, hits in docs_found.items():
        if shown >= max_results:
            break
        
        for hit in hits:
            if shown >= max_results:
                break
            
            score = hit.get('score', hit.get('distance', 0))
            page = hit.get('source_page', hit.get('page', 'N/A'))
            text = hit.get('text', hit.get('source', {}).get('text', ''))
            
            # Get highlight if available (Elasticsearch)
            highlight = hit.get('highlight', {})
            if highlight:
                text_hl = highlight.get('text', highlight.get('content', []))
                if text_hl:
                    text = text_hl[0] if isinstance(text_hl, list) else text_hl
            
            # Truncate text
            max_len = 200
            if len(text) > max_len:
                text = text[:max_len] + "..."
            
            print(f"    [{shown+1}] Document: {doc_id} | Page: {page} | Score: {score:.4f}")
            print(f"        {text}\n")
            
            shown += 1

def compare_search_methods(query, top_k=5):
    """Compare semantic vs keyword search for the same query"""
    print(f"\n{'='*80}")
    print(f"COMPARING SEARCH METHODS")
    print(f"{'='*80}")
    
    # Semantic search
    start = time.time()
    semantic_result = test_semantic_search(query, top_k)
    semantic_time = time.time() - start
    
    # Elasticsearch search
    start = time.time()
    es_result = test_elasticsearch_search(query, top_k)
    es_time = time.time() - start
    
    print_search_results(semantic_result, query, "SEMANTIC/MILVUS", max_results=3)
    print(f"    Search time: {semantic_time*1000:.2f}ms")
    
    print_search_results(es_result, query, "ELASTICSEARCH", max_results=3)
    print(f"    Search time: {es_time*1000:.2f}ms")

print("="*80)
print("LIBRARY SEARCH SERVICE TEST - USER SIMULATION")
print("="*80)

# Check service health
print("\nChecking service health...")
health = get_health()
if 'error' in health:
    print(f"ERROR: Cannot connect to search service: {health['error']}")
    exit(1)

print(f"  Status: {health.get('status')}")
print(f"  Milvus entities: {health.get('milvus', {}).get('total_entities', 0)}")
print(f"  Elasticsearch chunks: {health.get('elasticsearch', {}).get('total_chunks', 0)}")

# Test 1: Random user queries
print("\n" + "="*80)
print("TEST 1: RANDOM USER QUERIES (Semantic Search)")
print("="*80)

sample_queries = random.sample(user_queries, min(5, len(user_queries)))

for query in sample_queries:
    result = test_semantic_search(query, top_k=5)
    print_search_results(result, query, "SEMANTIC", max_results=3)
    time.sleep(0.3)

# Test 2: Highlighted text search
print("\n" + "="*80)
print("TEST 2: HIGHLIGHTED TEXT SNIPPETS (Simulating copy-paste search)")
print("="*80)

sample_snippets = random.sample(highlighted_snippets, min(3, len(highlighted_snippets)))

for snippet in sample_snippets:
    result = test_semantic_search(snippet, top_k=5)
    print_search_results(result, snippet, "SEMANTIC", max_results=3)
    time.sleep(0.3)

# Test 3: Comparison between search methods
print("\n" + "="*80)
print("TEST 3: SEMANTIC vs ELASTICSEARCH COMPARISON")
print("="*80)

comparison_queries = [
    "neurónové siete",
    "network configuration",
    "LaTeX tables",
]

for query in comparison_queries:
    compare_search_methods(query, top_k=5)
    time.sleep(0.3)

# Test 4: Performance test
print("\n" + "="*80)
print("TEST 4: PERFORMANCE TEST")
print("="*80)

test_query = random.choice(user_queries)
iterations = 20

print(f"\nRunning {iterations} searches for: '{test_query}'")

# Semantic search timing
semantic_times = []
for _ in range(iterations):
    start = time.time()
    test_semantic_search(test_query, top_k=10)
    semantic_times.append(time.time() - start)

# Elasticsearch timing
es_times = []
for _ in range(iterations):
    start = time.time()
    test_elasticsearch_search(test_query, top_k=10)
    es_times.append(time.time() - start)

print(f"\nSemantic Search (Milvus):")
print(f"  Average: {sum(semantic_times)/len(semantic_times)*1000:.2f}ms")
print(f"  Min: {min(semantic_times)*1000:.2f}ms")
print(f"  Max: {max(semantic_times)*1000:.2f}ms")

print(f"\nElasticsearch Search:")
print(f"  Average: {sum(es_times)/len(es_times)*1000:.2f}ms")
print(f"  Min: {min(es_times)*1000:.2f}ms")
print(f"  Max: {max(es_times)*1000:.2f}ms")

# Test 5: Cross-language search
print("\n" + "="*80)
print("TEST 5: CROSS-LANGUAGE SEARCH (Slovak + English)")
print("="*80)

cross_lang_queries = [
    ("neurónové siete", "neural networks"),
    ("sieťové zariadenia", "network devices"),
    ("matematická logika", "mathematical logic"),
]

for sk_query, en_query in cross_lang_queries:
    print(f"\n  Testing: '{sk_query}' vs '{en_query}'")
    print(f"  {'-'*76}")
    
    sk_result = test_semantic_search(sk_query, top_k=3)
    en_result = test_semantic_search(en_query, top_k=3)
    
    sk_docs = set(r.get('document_id') for r in sk_result.get('results', []))
    en_docs = set(r.get('document_id') for r in en_result.get('results', []))
    
    overlap = sk_docs & en_docs
    
    print(f"    Slovak query found in: {sk_docs}")
    print(f"    English query found in: {en_docs}")
    print(f"    Overlap: {overlap if overlap else 'None'}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)