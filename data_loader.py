from external_libs import logging, pd, os
from config import *


class DataLoader:
    def __init__(self, path):
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataLoader")

        file_extension = os.path.splitext(path)[-1].lower()  # grab file extension 
        self.data = None  

        if file_extension == ".csv":
            self.logger.info(f"import data from CSV {path}")
            self.data = self.load_csv_in_chunks(path)

        elif file_extension == ".json" or file_extension == ".jsonl":
            self.logger.info(f"import data from JSON {path}")
            self.data = self.load_json_in_chunks(path, lines=(file_extension == ".jsonl"))

        elif file_extension == ".xlsx":
            self.logger.info(f"import data from Excel {path}")
            self.data = pd.read_excel(path)

        elif file_extension == ".parquet":
            self.logger.info(f"import data from Parquet {path}")
            self.data = pd.read_parquet(path)

        elif file_extension == ".feather":
            self.logger.info(f"import data from Feather {path}")
            self.data = pd.read_feather(path)

        elif file_extension == ".hdf":
            self.logger.info(f"import data from HDF5 {path}")
            self.data = pd.read_hdf(path)

        elif file_extension == ".pkl":
            self.logger.info(f"import data from Pickle {path}")
            self.data = pd.read_pickle(path)

        elif file_extension == ".txt":
            self.logger.info(f"import data from texte {path}")
            self.data = pd.read_csv(path, delimiter='\t')

        else:
            self.logger.critical(f"unsupported file format : {file_extension}")
            self.data = None

        if self.data is not None:
            self.logger.info("converting text columns to lowercase")
            text_columns = self.data.select_dtypes(include=['object']).columns
            for col in text_columns:
                self.data[col] = self.data[col].str.lower()

    def load_csv_in_chunks(self, path, chunksize=1000):
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunk = chunk.sample(frac=1).reset_index(drop=True)  
            chunks.append(chunk)
        concatenated_data = pd.concat(chunks, ignore_index=True)
        return concatenated_data.sample(frac=1).reset_index(drop=True)  

    def load_json_in_chunks(self, path, lines=True, chunksize=1000):
        chunks = []
        with open(path, 'r') as file:
            for chunk in pd.read_json(file, lines=lines, chunksize=chunksize):
                chunk = chunk.sample(frac=1).reset_index(drop=True)  
                chunks.append(chunk)
        concatenated_data = pd.concat(chunks, ignore_index=True)
        return concatenated_data.sample(frac=1).reset_index(drop=True)  

    def get_df(self):
        return self.data
