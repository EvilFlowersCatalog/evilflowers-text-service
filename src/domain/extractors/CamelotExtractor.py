import os
import pandas as pd
import camelot

from domain.base.ExtractorInterface import ExtractorInterface

class CamelotExtractor(ExtractorInterface):
    
    def extract_tables(self, dataframe: bool = True, csv: bool = False, output_path: str = ".") -> dict[str, pd.DataFrame]:
        self._load_document()
        dfs = {}
        for i, table in enumerate(self._doc):
            table_name = f"{os.path.splitext(os.path.basename(self._document_path))[0]}_lattice_table_{str(i + 1).zfill(3)}.csv"
            
            if dataframe:
                dfs[table_name] = table.df

            if csv:
                table.to_csv(f"{output_path}/{table_name}.csv", index=False) 

        return dfs

    # Private methods
    def _load_document(self):
        if self._mode is None:
            print("No mode provided, using default mode: lattice")
            self._mode = "lattice"
        self._doc = camelot.read_pdf(self._document_path, pages="all", flavor=self._mode)
        return