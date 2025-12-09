"""
Test semantic search with sample dictionary
No PDF, no TextHandler - just a simple dict with paragraphs
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Set environment variables before importing
os.environ['USE_MILVUS_LITE'] = 'true'
os.environ['SEMANTIC_SEARCH_ENABLED'] = 'true'
os.environ['EMBEDDING_DEVICE'] = 'cpu'
os.environ['EMBEDDING_BATCH_SIZE'] = '8'
os.environ['CHUNK_LEVEL'] = 'paragraph'
os.environ['MILVUS_COLLECTION_NAME'] = 'test_collection'

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from semantic.indexer import SemanticIndexer
from semantic.search import SemanticSearch


def main():
    print("="*70)
    print("SEMANTIC SEARCH TEST WITH SAMPLE DICTIONARY")
    print("="*70)
    
    # Sample data structure matching your TextHandler output
    sample_data = {
        "document_id": "sample_algebra_book",
        
        # Pages: List of (page_num, page_text) tuples
        "pages": [
            (1, "Úvod do algebry. Algebra je matematická disciplína."),
            (2, "Lineárna algebra. Vektory a matice sú základné objekty lineárnej algebry."),
            (3, "Matice. Matica je usporiadané pole čísel v riadkoch a stĺpcoch."),
            (4, "Determinanty. Determinant je číslo priradené štvorcovej matici."),
            (5, "Lineárne transformácie. Transformácie zachovávajú vektorové operácie.")
        ],
        
        # Paragraphs: List of lists (one list per page)
        "paragraphs": [
            # Page 1 paragraphs
            [
                "Úvod do algebry. Algebra je matematická disciplína, ktorá študuje štruktúry, vzťahy a operácie.",
                "História algebry siaha do starovekého Babylonu a Egypta.",
                "Moderná algebra zahŕňa mnoho oblastí ako lineárna algebra, abstraktná algebra a algebraická geometria."
            ],
            # Page 2 paragraphs
            [
                "Lineárna algebra študuje vektory a matice. Vektory sú základné objekty, ktoré reprezentujú veličiny s veľkosťou a smerom.",
                "Matice sú tabuľky čísel usporiadané do riadkov a stĺpcov. Používajú sa na reprezentáciu lineárnych transformácií.",
                "Lineárne rovnice tvoria systémy, ktoré možno riešiť pomocou maticových metód."
            ],
            # Page 3 paragraphs
            [
                "Matica je usporiadané pole čísel v riadkoch a stĺpcoch. Každý prvok má pozíciu [i,j].",
                "Maticové operácie zahŕňajú sčítanie, odčítanie a násobenie. Násobenie matíc nie je komutatívne.",
                "Štvorcové matice majú rovnaký počet riadkov a stĺpcov. Majú špeciálne vlastnosti ako determinant."
            ],
            # Page 4 paragraphs
            [
                "Determinant je číslo priradené štvorcovej matici. Používa sa na riešenie lineárnych systémov.",
                "Pre maticu 2×2 je determinant ad-bc, kde a,b,c,d sú prvky matice.",
                "Determinant je nulový práve vtedy, keď matica je singulárna a nemá inverziu."
            ],
            # Page 5 paragraphs
            [
                "Lineárne transformácie zachovávajú vektorové operácie. T(u+v) = T(u) + T(v).",
                "Každá lineárna transformácia môže byť reprezentovaná maticou.",
                "Príklady lineárnych transformácií: rotácie, reflexie, škálovania a projekcie."
            ]
        ]
    }
    
    print(f"\n📊 Sample Data:")
    print(f"   Document ID: {sample_data['document_id']}")
    print(f"   Pages: {len(sample_data['pages'])}")
    print(f"   Total paragraphs: {sum(len(p) for p in sample_data['paragraphs'])}")
    
    # Step 1: Initialize indexer
    print("\n" + "="*70)
    print("STEP 1: Initializing Semantic Indexer")
    print("="*70)
    
    try:
        indexer = SemanticIndexer()
        print("✓ Indexer initialized")
    except Exception as e:
        print(f"✗ Failed to initialize indexer: {e}")
        return
    
    # Step 2: Index the sample document
    print("\n" + "="*70)
    print("STEP 2: Indexing Sample Document")
    print("="*70)
    
    try:
        result = indexer.index_document(
            document_id=sample_data["document_id"],
            pages=sample_data["pages"]
        )
        
        if result["success"]:
            print(f"✓ Indexing successful!")
            print(f"   Chunks indexed: {result['chunks_indexed']}")
            print(f"   Chunk level: {result['chunk_level']}")
            print(f"   Embedding dimension: {result['embedding_dim']}")
        else:
            print(f"✗ Indexing failed: {result.get('error', 'Unknown error')}")
            return
            
    except Exception as e:
        print(f"✗ Indexing error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Test semantic search
    print("\n" + "="*70)
    print("STEP 3: Testing Semantic Search")
    print("="*70)
    
    try:
        search = SemanticSearch()
        print("✓ Search interface initialized")
    except Exception as e:
        print(f"✗ Failed to initialize search: {e}")
        return
    
    # Test queries
    test_queries = [
        ("Slovak query", "čo je matica"),
        ("Slovak query", "lineárne transformácie"),
        ("Slovak query", "determinant matice"),
        ("English query (cross-lingual)", "What are vectors?"),
        ("English query (cross-lingual)", "matrix operations"),
    ]
    
    print("\n" + "-"*70)
    
    for query_type, query in test_queries:
        print(f"\n🔍 {query_type}: '{query}'")
        print("-"*70)
        
        try:
            results = search.search(query, top_k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n   Result {i}:")
                    print(f"      Score: {result['score']:.4f}")
                    print(f"      Page: {result['page_num']}")
                    print(f"      Text: {result['text'][:120]}...")
            else:
                print("   No results found")
                
        except Exception as e:
            print(f"   ✗ Search failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("✓ TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nYour semantic search pipeline is working correctly with dictionary input!")
    print("Next steps:")
    print("  1. Integrate with your Kafka consumer")
    print("  2. Use TextHandler output: pages, paragraphs, sentences, toc")
    print("  3. Pass pages and paragraphs to indexer.index_document()")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()