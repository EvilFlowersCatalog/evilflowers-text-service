import os
import pandas as pd
import pdfplumber

from domain.base.ExtractorInterface import ExtractorInterface

class PdfPlumberExtractor(ExtractorInterface):

    def extract_tables(self, dataframe: bool = True, csv: bool = False, output_path: str = ".") -> dict[str, pd.DataFrame]:
        self._load_document()
        dfs = {}

        for page_num, page in enumerate(self._doc.pages):
            tables = page.extract_tables()

            # Get tables as dataframe / csv
            if tables:
                # print(f"{len(tables)} table(s) on page {page_num + 1}...")
                for i, table in enumerate(tables):
                    table_name = f"{os.path.splitext(os.path.basename(self._document_path))[0]}_page_{str(page_num + 1).zfill(3)}_table_{str(i + 1).zfill(2)}"
                    df = pd.DataFrame(table[1:], columns=table[0])

                    if dataframe:
                        dfs[table_name] = df

                    if csv:
                        df.to_csv(f"{output_path}/{table_name}.csv", index=False)

        self._doc.close()
        return dfs
    
    # Private methods
    def _load_document(self):
        self._doc = pdfplumber.open(self._document_path)
        return