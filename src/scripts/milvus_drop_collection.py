from pymilvus import connections, utility
import os

# Connect to Milvus with fallback values
host = os.getenv('MILVUS_HOST') or 'milvus-standalone'
port = int(os.getenv('MILVUS_PORT') or '19530')
collection_name = os.getenv('MILVUS_COLLECTION_NAME') or 'document_embeddings'

print(f"Connecting to Milvus at {host}:{port}")
connections.connect(
    alias="default",
    host=host,
    port=port
)

if utility.has_collection(collection_name):
    print(f"Dropping collection: {collection_name}")
    utility.drop_collection(collection_name)
    print(f"✓ Collection {collection_name} dropped")
else:
    print(f"Collection {collection_name} does not exist")

connections.disconnect("default")