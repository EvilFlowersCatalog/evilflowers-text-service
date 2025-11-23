#!/usr/bin/env python3
"""
Example Usage: Semantic Search for Slovak Academic Library System

This script demonstrates how to use the semantic search system for indexing
and searching academic documents in Slovak and other languages.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from semantic_embeddings import SemanticEmbeddingsService
from milvus_client import MilvusVectorStore
from semantic_search_service import SemanticSearchService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_basic_embeddings():
    """Example 1: Generate embeddings for Slovak and English text"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Embedding Generation")
    print("="*80)
    
    service = SemanticEmbeddingsService()
    
    # Slovak academic texts
    slovak_texts = [
        "Algebra a diskrétna matematika sú základné oblasti matematiky.",
        "Lineárna algebra študuje vektory, matice a lineárne transformácie.",
        "Diskrétna matematika zahŕňa teóriu grafov, kombinatoriku a logiku."
    ]
    
    # English equivalent texts
    english_texts = [
        "Algebra and discrete mathematics are fundamental areas of mathematics.",
        "Linear algebra studies vectors, matrices, and linear transformations.",
        "Discrete mathematics includes graph theory, combinatorics, and logic."
    ]
    
    # Generate embeddings
    slovak_embeddings = service.encode_text(slovak_texts)
    english_embeddings = service.encode_text(english_texts)
    
    print(f"\nGenerated {len(slovak_embeddings)} Slovak embeddings")
    print(f"Embedding dimension: {slovak_embeddings.shape[1]}")
    
    # Compute cross-lingual similarities
    print("\nCross-lingual similarity (Slovak → English):")
    for i, sk_text in enumerate(slovak_texts):
        similarities = service.compute_similarity(slovak_embeddings[i], english_embeddings)
        best_match_idx = similarities.argmax()
        print(f"\n  SK: {sk_text[:60]}...")
        print(f"  EN: {english_texts[best_match_idx][:60]}...")
        print(f"  Similarity: {similarities[best_match_idx]:.4f}")


def example_2_document_indexing():
    """Example 2: Index a simulated academic document"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Document Indexing")
    print("="*80)
    
    service = SemanticSearchService(
        milvus_host="localhost",
        milvus_port="19530"
    )
    
    # Simulated document: Algebra textbook
    document_id = "algebra-textbook-sk-001"
    
    pages = [
        (1, "Úvod do algebry. Algebra je jednou z najdôležitejších oblastí matematiky. Študuje štruktúry, vzťahy a operácie."),
        (2, "Lineárna algebra. Vektory sú základné objekty lineárnej algebry. Matice reprezentujú lineárne transformácie."),
        (3, "Abstraktná algebra. Grupy, okruhy a polia sú základné algebraické štruktúry."),
        (4, "Aplikácie algebry. Algebra má široké využitie v informatike, fyzike a ekonomike."),
    ]
    
    paragraphs = [
        ["Úvod do algebry. Algebra je jednou z najdôležitejších oblastí matematiky.", "Študuje štruktúry, vzťahy a operácie."],
        ["Lineárna algebra. Vektory sú základné objekty lineárnej algebry.", "Matice reprezentujú lineárne transformácie."],
        ["Abstraktná algebra. Grupy, okruhy a polia sú základné algebraické štruktúry."],
        ["Aplikácie algebry. Algebra má široké využitie v informatike, fyzike a ekonomike."],
    ]
    
    # Index document
    result = service.index_document(
        document_id=document_id,
        pages=pages,
        paragraphs=paragraphs,
        chunk_level="paragraph"
    )
    
    print(f"\nIndexing result:")
    print(f"  Document ID: {result['document_id']}")
    print(f"  Chunks indexed: {result['chunks_indexed']}")
    print(f"  Chunk level: {result['chunk_level']}")
    print(f"  Success: {result['success']}")
    
    return service, document_id


def example_3_semantic_search(service, document_id):
    """Example 3: Perform semantic search queries"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Semantic Search")
    print("="*80)
    
    # Slovak queries
    queries = [
        "Čo sú vektory?",
        "Kde sa používa algebra?",
        "Základné algebraické štruktúry",
        "What are vectors?",  # English query on Slovak document
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        results = service.search_similar(query, top_k=2, document_id=document_id)
        
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Score: {result['score']:.4f}")
            print(f"    Page: {result['page_num']}")
            print(f"    Text: {result['text_preview'][:100]}...")


def example_4_find_similar_content(service):
    """Example 4: Find similar content based on text selection"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Find Similar Content (Selection-based)")
    print("="*80)
    
    # User selects this text
    selected_text = "Matice reprezentujú lineárne transformácie"
    
    print(f"\n🔍 Selected text: {selected_text}")
    print("\nFinding similar content across all documents...\n")
    
    results = service.search_by_selection(selected_text, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"Similar content {i}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Document: {result['document_id']}")
        print(f"  Page: {result['page_num']}")
        print(f"  Text: {result['text_preview'][:100]}...")
        print()


def example_5_multilingual_corpus():
    """Example 5: Search across multilingual document corpus"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Multilingual Document Corpus")
    print("="*80)
    
    service = SemanticSearchService(
        milvus_host="localhost",
        milvus_port="19530"
    )
    
    # Index multiple documents in different languages
    documents = [
        {
            "id": "math-sk-001",
            "language": "Slovak",
            "pages": [
                (1, "Diferenciálne rovnice sú základným nástrojom v matematickom modelovaní."),
                (2, "Parciálne diferenciálne rovnice popisujú mnoho prírodných javov."),
            ],
            "paragraphs": [
                ["Diferenciálne rovnice sú základným nástrojom v matematickom modelovaní."],
                ["Parciálne diferenciálne rovnice popisujú mnoho prírodných javov."],
            ]
        },
        {
            "id": "math-en-001",
            "language": "English",
            "pages": [
                (1, "Differential equations are a fundamental tool in mathematical modeling."),
                (2, "Partial differential equations describe many natural phenomena."),
            ],
            "paragraphs": [
                ["Differential equations are a fundamental tool in mathematical modeling."],
                ["Partial differential equations describe many natural phenomena."],
            ]
        },
        {
            "id": "math-de-001",
            "language": "German",
            "pages": [
                (1, "Differentialgleichungen sind ein grundlegendes Werkzeug in der mathematischen Modellierung."),
            ],
            "paragraphs": [
                ["Differentialgleichungen sind ein grundlegendes Werkzeug in der mathematischen Modellierung."],
            ]
        }
    ]
    
    # Index all documents
    for doc in documents:
        result = service.index_document(
            document_id=doc["id"],
            pages=doc["pages"],
            paragraphs=doc["paragraphs"],
            chunk_level="paragraph"
        )
        print(f"✓ Indexed {doc['language']} document: {doc['id']} ({result['chunks_indexed']} chunks)")
    
    # Cross-lingual search
    print("\n🌍 Cross-lingual search: Query in Czech, find Slovak/English/German content")
    query = "Co jsou diferenciální rovnice?"  # Czech query
    
    results = service.search_similar(query, top_k=5)
    
    print(f"\nResults for query: '{query}'\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Document: {result['document_id']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Text: {result['text_preview'][:80]}...")
        print()


def example_6_service_info():
    """Example 6: Get service information"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Service Information")
    print("="*80)
    
    service = SemanticSearchService(
        milvus_host="localhost",
        milvus_port="19530"
    )
    
    info = service.get_service_info()
    
    print("\n📊 Embeddings Model:")
    for key, value in info['embeddings'].items():
        print(f"  {key}: {value}")
    
    print("\n📦 Vector Store:")
    for key, value in info['vector_store'].items():
        print(f"  {key}: {value}")
    
    print("\n⚙️ Capabilities:")
    for key, value in info['capabilities'].items():
        print(f"  {key}: {value}")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("SEMANTIC SEARCH DEMONSTRATION")
    print("Slovak Academic Library System")
    print("="*80)
    
    try:
        # Example 1: Basic embeddings
        example_1_basic_embeddings()
        
        # Example 2-4: Require Milvus connection
        # Check if Milvus is available
        print("\n\nChecking Milvus connection...")
        try:
            from pymilvus import connections
            connections.connect(host="localhost", port="19530", timeout=5)
            print("✓ Milvus connection successful")
            connections.disconnect("default")
            
            # Example 2: Index document
            service, doc_id = example_2_document_indexing()
            
            # Example 3: Search
            example_3_semantic_search(service, doc_id)
            
            # Example 4: Find similar
            example_4_find_similar_content(service)
            
            # Example 5: Multilingual
            example_5_multilingual_corpus()
            
            # Example 6: Service info
            example_6_service_info()
            
            # Cleanup
            service.close()
            
        except Exception as e:
            print(f"⚠️  Milvus not available: {e}")
            print("   Skipping examples that require Milvus")
            print("   Start Milvus with: docker-compose -f docker-compose-semantic.yml up -d")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
