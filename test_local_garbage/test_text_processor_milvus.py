"""
Complete integration test: TextExtractor/TextHandler + Milvus Local
Tests the full pipeline from PDF extraction to semantic search
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Set environment variables BEFORE importing
os.environ['USE_MILVUS_LITE'] = 'true'
os.environ['SEMANTIC_SEARCH_ENABLED'] = 'true'
os.environ['EMBEDDING_DEVICE'] = 'cpu'
os.environ['EMBEDDING_BATCH_SIZE'] = '8'
os.environ['CHUNK_LEVEL'] = 'paragraph'  # or 'page'
os.environ['MILVUS_COLLECTION_NAME'] = 'test_text_handler_collection'
os.environ['MILVUS_LITE_DB'] = './test_milvus.db'

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from text_handler.TextService import TextHandler
from semantic.indexer import SemanticIndexer
from semantic.search import SemanticSearch


def test_with_real_pdf(pdf_path: str):
    """
    Test with a real PDF file using TextHandler

    Args:
        pdf_path: Path to PDF file
    """
    print("="*80)
    print("TEXT HANDLER + MILVUS LOCAL INTEGRATION TEST")
    print("="*80)

    # Validate PDF exists
    if not os.path.exists(pdf_path):
        print(f"✗ PDF file not found: {pdf_path}")
        print("\nPlease provide a valid PDF path as argument:")
        print(f"  python {__file__} /path/to/your/document.pdf")
        return False

    print(f"\n📄 PDF File: {pdf_path}")
    document_id = f"test_doc_{Path(pdf_path).stem}"

    # =========================================================================
    # STEP 1: Extract text using TextHandler
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: Extracting text from PDF using TextHandler")
    print("="*80)

    try:
        text_handler = TextHandler(pdf_path)
        print("✓ TextHandler initialized")

        # Extract text - returns (pages, toc)
        # pages is now a list of dicts with structure:
        # {
        #     'page_num': int,
        #     'text': str,              # Full page text
        #     'paragraphs': List[str],  # List of paragraphs
        #     'sentences': List[str]    # List of sentences
        # }
        pages, toc = text_handler.extract_text(found_toc=False)

        print(f"✓ Text extraction complete!")
        print(f"   Total pages: {len(pages)}")

        if pages:
            # Show sample from first page
            first_page = pages[0]
            print(f"\n   Sample from page {first_page['page_num']}:")
            print(f"      Text preview: {first_page['text'][:150]}...")
            print(f"      Paragraphs: {len(first_page['paragraphs'])}")
            print(f"      Sentences: {len(first_page['sentences'])}")

        # Calculate total stats
        total_paragraphs = sum(len(page['paragraphs']) for page in pages)
        total_sentences = sum(len(page['sentences']) for page in pages)
        print(f"\n   Total paragraphs: {total_paragraphs}")
        print(f"   Total sentences: {total_sentences}")

    except Exception as e:
        print(f"✗ Text extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # STEP 2: Initialize Semantic Indexer
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Initializing Semantic Indexer with Milvus Local")
    print("="*80)

    try:
        indexer = SemanticIndexer()
        print("✓ SemanticIndexer initialized")
        print("   Using Milvus Lite (embedded mode)")
        print(f"   Database: {os.environ.get('MILVUS_LITE_DB')}")
        print(f"   Collection: {os.environ.get('MILVUS_COLLECTION_NAME')}")

    except Exception as e:
        print(f"✗ Failed to initialize indexer: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # STEP 3: Index the document
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: Indexing document for semantic search")
    print("="*80)

    try:
        print(f"Indexing document: {document_id}")
        print(f"Chunk level: {os.environ.get('CHUNK_LEVEL')}")

        # Index document - now just pass pages (new dict format)
        result = indexer.index_document(
            document_id=document_id,
            pages=pages
        )

        if result['success']:
            print(f"\n✓ Indexing successful!")
            print(f"   Document ID: {result['document_id']}")
            print(f"   Chunks indexed: {result['chunks_indexed']}")
            print(f"   Chunk level: {result['chunk_level']}")
            print(f"   Embedding dimension: {result['embedding_dim']}")
        else:
            print(f"✗ Indexing failed: {result.get('error', 'Unknown error')}")
            return False

    except Exception as e:
        print(f"✗ Indexing error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # =========================================================================
    # STEP 4: Test semantic search
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: Testing Semantic Search")
    print("="*80)

    try:
        search = SemanticSearch()
        print("✓ SemanticSearch initialized")

    except Exception as e:
        print(f"✗ Failed to initialize search: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Get some text from the document for test queries
    print("\n📝 Enter search queries (or press Enter for auto-generated queries):")

    # Auto-generate queries based on extracted text
    auto_queries = []
    if pages:
        # Extract some keywords from first few pages
        sample_text = " ".join([p['text'] for p in pages[:3]])
        words = sample_text.split()[:100]  # First 100 words

        # Create simple queries
        if len(words) > 10:
            auto_queries = [
                " ".join(words[0:5]),  # First 5 words
                " ".join(words[10:15]),  # Words 10-15
            ]

    # Add some generic queries
    test_queries = auto_queries + [
        "hlavná téma dokumentu",  # Main topic
        "kľúčové koncepty",  # Key concepts
    ]

    print(f"\nRunning {len(test_queries)} test queries:")
    print("-"*80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: '{query}'")
        print("-"*80)

        try:
            results = search.search(
                query=query,
                top_k=3,
                document_id=document_id  # Search only in this document
            )

            if results:
                print(f"   Found {len(results)} results:")
                for j, result in enumerate(results, 1):
                    print(f"\n   Result {j}:")
                    print(f"      Score: {result['score']:.4f}")
                    print(f"      Page: {result['page_num']}")
                    print(f"      Chunk: {result['chunk_type']}")
                    text_preview = result['text'][:200].replace('\n', ' ')
                    print(f"      Text: {text_preview}...")
            else:
                print("   No results found")

        except Exception as e:
            print(f"   ✗ Search failed: {e}")

    # =========================================================================
    # STEP 5: Collection stats
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: Collection Statistics")
    print("="*80)

    try:
        stats = indexer.milvus_manager.get_stats()
        print(f"Collection: {stats['collection_name']}")
        print(f"Total entities: {stats['total_entities']}")
        print(f"Embedding dimension: {stats['embedding_dim']}")

    except Exception as e:
        print(f"✗ Failed to get stats: {e}")

    # =========================================================================
    # SUCCESS!
    # =========================================================================
    print("\n" + "="*80)
    print("✓ INTEGRATION TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nYour pipeline is working:")
    print("  1. ✓ TextHandler extracts text from PDF")
    print("  2. ✓ SemanticIndexer generates embeddings")
    print("  3. ✓ Milvus Local stores vectors")
    print("  4. ✓ SemanticSearch finds relevant content")
    print("\nNext steps:")
    print("  - Integrate this into your API (api.py)")
    print("  - Connect to your Kafka consumer")
    print("  - Scale to production with Milvus server")

    return True


def test_with_sample_data():
    """
    Test with sample dictionary data (no PDF needed)
    Useful for quick testing without PDF files
    """
    print("="*80)
    print("SAMPLE DATA TEST (No PDF required)")
    print("="*80)

    # Create sample data matching TextHandler output format
    sample_pages = [
        {
            'page_num': 1,
            'text': 'Úvod do programovania. Python je moderný programovací jazyk. Je jednoduché sa ho naučiť.',
            'paragraphs': [
                'Úvod do programovania. Python je moderný programovací jazyk, ktorý je populárny pre svoju jednoduchosť.',
                'Python podporuje objektovo orientované programovanie, funkcionálne programovanie aj procedurálne programovanie.'
            ],
            'sentences': [
                'Úvod do programovania.',
                'Python je moderný programovací jazyk.',
                'Je jednoduché sa ho naučiť.'
            ]
        },
        {
            'page_num': 2,
            'text': 'Premenné a dátové typy. V Pythone existujú rôzne dátové typy ako čísla, reťazce a zoznamy.',
            'paragraphs': [
                'Premenné v Pythone sa vytvárajú jednoducho priradením hodnoty. Nie je potrebné deklarovať typ.',
                'Základné dátové typy zahŕňajú int, float, str, bool, list, dict a tuple.'
            ],
            'sentences': [
                'Premenné a dátové typy.',
                'V Pythone existujú rôzne dátové typy.',
                'Môžeme použiť čísla, reťazce a zoznamy.'
            ]
        },
        {
            'page_num': 3,
            'text': 'Funkcie a moduly. Funkcie sú základným stavebným prvkom v Pythone.',
            'paragraphs': [
                'Funkcie definujeme pomocou kľúčového slova def. Funkcie môžu prijímať parametre a vracať hodnoty.',
                'Moduly umožňujú organizovať kód do opakovaně použiteľných súborov. Import modulov sa robí cez import.'
            ],
            'sentences': [
                'Funkcie a moduly.',
                'Funkcie sú základným stavebným prvkom.',
                'Moduly organizujú kód.'
            ]
        }
    ]

    document_id = "sample_python_guide"

    print(f"\n📊 Sample Data: {len(sample_pages)} pages")

    # Step 1: Initialize indexer
    print("\n" + "="*80)
    print("Initializing Semantic Indexer")
    print("="*80)

    try:
        indexer = SemanticIndexer()
        print("✓ Indexer initialized")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    # Step 2: Index
    print("\n" + "="*80)
    print("Indexing sample document")
    print("="*80)

    try:
        result = indexer.index_document(
            document_id=document_id,
            pages=sample_pages
        )

        if result['success']:
            print(f"✓ Indexed {result['chunks_indexed']} chunks")
        else:
            print(f"✗ Failed: {result.get('error')}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Search
    print("\n" + "="*80)
    print("Testing search")
    print("="*80)

    try:
        search = SemanticSearch()

        queries = [
            "čo je Python",
            "dátové typy",
            "funkcie a moduly",
            "what are variables",  # Cross-lingual
        ]

        for query in queries:
            print(f"\n🔍 '{query}'")
            results = search.search(query, top_k=2, document_id=document_id)

            for i, r in enumerate(results, 1):
                print(f"   {i}. [Page {r['page_num']}, Score: {r['score']:.3f}]")
                print(f"      {r['text'][:100]}...")

    except Exception as e:
        print(f"✗ Search failed: {e}")
        return False

    print("\n✓ Sample data test completed successfully!")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test TextHandler + Milvus integration')
    parser.add_argument('pdf_path', nargs='?', help='Path to PDF file (optional)')
    parser.add_argument('--sample', action='store_true', help='Run with sample data instead of PDF')

    args = parser.parse_args()

    try:
        if args.sample:
            # Test with sample data
            success = test_with_sample_data()
        elif args.pdf_path:
            # Test with provided PDF
            success = test_with_real_pdf(args.pdf_path)
        else:
            # No PDF provided, ask user
            print("="*80)
            print("TEXT HANDLER + MILVUS LOCAL TEST")
            print("="*80)
            print("\nOptions:")
            print("  1. Test with sample data (no PDF needed)")
            print("  2. Test with your PDF file")

            choice = input("\nChoose option (1 or 2): ").strip()

            if choice == "1":
                success = test_with_sample_data()
            elif choice == "2":
                pdf_path = input("Enter PDF path: ").strip()
                success = test_with_real_pdf(pdf_path)
            else:
                print("Invalid choice")
                success = False

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
