from text_handler.TextService import TextHandler


def main():
    document_path = "src/test_data/algebra-a-diskretna-matematika.pdf"
    text_handler = TextHandler(document_path)

    # This will come from analyzer service
    found_toc = False

    return text_handler.extract_text(found_toc), text_handler.extract_tables()

if __name__ == "__main__":
    text, tables = main()
   
    # print("whole text object: ", text)
    
    # print("example page: ", text[0][9])
    print('-' * 100)
    # print("example page: ", text[2][9])
    print('-' * 100)
    print("example page: ", text[1][9])
    # print("toc: ", text[3])
    
    # print("all paragraphs: ", text[1])
    # print("example all paragraph for specific page: \n\n", text[1][10])
    # print("example all paragraph for specific page: \n\n", text[1][10][1])

    # print("all page sentences: ", text[2])
    # print("all paragraph sentences: ", text[2][0])
    # print("example sentence: ", text[2][0][1])

    # print(tables)

