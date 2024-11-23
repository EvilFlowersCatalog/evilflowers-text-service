from text_handler.TextService import TextHandler


def main():
    document_path = "test_data/doc1.pdf"
    text_handler = TextHandler(document_path)
    return text_handler.extract_text(), text_handler.extract_tables()

if __name__ == "__main__":
    text, tables = main()
    print(text)
    print(tables)
