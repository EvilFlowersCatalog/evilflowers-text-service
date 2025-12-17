from text_handler.TextService import TextService
from semantic.SemanticService import SemanticService

def main():
    document_path = "src/test_data/algebra-a-diskretna-matematika.pdf"
    text_service = TextService(document_path)
    semantic_service = SemanticService()

    # This will come from analyzer service
    found_toc = False

    text_results = text_service.extract_text(found_toc)
    tables = text_service.extract_tables()

    pages, toc = text_results

    chunks = text_service.process_text(pages)

    result = semantic_service.index_document(chunks)

    print('-' * 100)
    print("example page: ", pages[9]['text'])
    print('-' * 100)
    # print('example chunk', chunks['chunks'][50])
    print('-' * 100)
    # print('embeddings example', embeddings[50])
    print(result)


if __name__ == "__main__":
    main()
