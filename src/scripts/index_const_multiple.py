from config.Config import Config
from celery import Celery
import time
import requests

app = Celery(
    'text_service',
    broker='redis://redis:6379/0',
    backend=Config.REDIS_URL
)

# Test documents - add your real document paths here
test_documents = [
    {
        'path': '0b52959f-1c2a-4e1f-b44f-4e2c9bd1403e/42ca0c8c-243b-4a49-8465-73ada285a802.pdf',
        'doc_id': 'eye-tracking'
    },
    {
        'path': '0b87443f-4f60-49a9-9aa5-c27fd4acb837/fb672719-9bbe-45a9-9a4b-d1b3188eb28f.pdf',
        'doc_id': 'matematicka-logika'
    },
    {
        'path': '0d54e152-eac9-4bb0-801c-60779cc6dc25/9bab2e37-4aa4-4115-81ab-9a80c6324638.pdf',
        'doc_id': 'uvod-do-latex'
    },
    {
        'path': '0f5fde23-498b-49fe-96f3-33de2929f4fe/854e7935-8ff4-499d-a573-78415e73f091.pdf',
        'doc_id': 'umela-inteligencia-kognitivna-veda'
    },
    {
        'path': '1dfc532f-2a4a-4efc-9c99-2c5306f89efc/b61b3ca1-09dc-45e2-a708-876b94b65446.pdf',
        'doc_id': 'sietove-zariadenia'
    },
]

def get_search_service_stats():
    """Get stats from search service"""
    try:
        response = requests.get('http://search-service:8001/health')
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Failed to get stats: {e}")
        return None

print("="*80)
print("STARTING BATCH INDEXING TEST")
print("="*80)

# Get initial stats
print("\nInitial Stats:")
initial_stats = get_search_service_stats()
if initial_stats:
    print(f"  Milvus: {initial_stats.get('milvus', {}).get('total_entities', 0)} entities")
    print(f"  Elasticsearch: {initial_stats.get('elasticsearch', {}).get('total_chunks', 0)} chunks")

# Process all documents
results = []
start_time = time.time()

print(f"\nProcessing {len(test_documents)} documents...")
print("-"*80)

for i, doc in enumerate(test_documents, 1):
    print(f"\n[{i}/{len(test_documents)}] Processing: {doc['doc_id']}")
    print(f"  Path: {doc['path']}")
    
    try:
        # Send task
        result = app.send_task(
            'evilflowers_text_worker.process_pdf',
            args=[doc['path'], doc['doc_id']],
            queue='evilflowers_text_worker'
        )
        
        print(f"  Task ID: {result.id}")
        print(f"  Waiting for result...")
        
        # Get result with timeout
        task_result = result.get(timeout=500)
        
        # Store result
        results.append({
            'doc_id': doc['doc_id'],
            'success': True,
            'result': task_result,
            'task_id': result.id
        })
        
        print(f"  SUCCESS")
        print(f"     Elasticsearch: {task_result.get('elasticsearch', {}).get('indexed', 0)} chunks")
        print(f"     Milvus: {task_result.get('milvus', {}).get('chunks_indexed', 0)} chunks")
        
    except Exception as e:
        print(f"  FAILED: {e}")
        results.append({
            'doc_id': doc['doc_id'],
            'success': False,
            'error': str(e)
        })

total_time = time.time() - start_time

# Print summary
print("\n" + "="*80)
print("INDEXING SUMMARY")
print("="*80)

successful = sum(1 for r in results if r['success'])
failed = len(results) - successful

print(f"\nResults:")
print(f"  Total documents: {len(test_documents)}")
print(f"  Successful: {successful}")
print(f"  Failed: {failed}")
print(f"  Total time: {total_time:.2f}s")
print(f"  Average time per document: {total_time/len(test_documents):.2f}s")

# Detailed results
print(f"\nDetailed Results:")
for result in results:
    if result['success']:
        es_count = result['result'].get('elasticsearch', {}).get('indexed', 0)
        mv_count = result['result'].get('milvus', {}).get('chunks_indexed', 0)
        match_status = "MATCH" if es_count == mv_count else "MISMATCH"
        print(f"  [{match_status}] {result['doc_id']}: ES={es_count}, Milvus={mv_count}")
    else:
        print(f"  [FAILED] {result['doc_id']}: {result['error']}")

# Get final stats
print(f"\nFinal Stats:")
final_stats = get_search_service_stats()
if final_stats:
    milvus_final = final_stats.get('milvus', {}).get('total_entities', 0)
    es_final = final_stats.get('elasticsearch', {}).get('total_chunks', 0)
    
    print(f"  Milvus: {milvus_final} entities")
    print(f"  Elasticsearch: {es_final} chunks")
    
    if initial_stats:
        milvus_initial = initial_stats.get('milvus', {}).get('total_entities', 0)
        es_initial = initial_stats.get('elasticsearch', {}).get('total_chunks', 0)
        
        milvus_added = milvus_final - milvus_initial
        es_added = es_final - es_initial
        
        print(f"\n  Added:")
        print(f"     Milvus: +{milvus_added}")
        print(f"     Elasticsearch: +{es_added}")
        
        if milvus_added == es_added:
            print(f"  Status: COUNTS MATCH")
        else:
            print(f"  Status: MISMATCH - {abs(milvus_added - es_added)} chunks difference")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)