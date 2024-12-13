from external_libs import logging, pd, warnings, np
from config import *



logger = logging.getLogger(__name__)


def columns_renamer(columns, data): # rename every columns in the list by the new name also in the list, ex: ['titi', 'toto'], toto will replace titi
    if len(columns) % 2 != 0:
        logger.critical("the list must countain an even number of elements")
    rename_dict = {columns[i]: columns[i + 1] for i in range(0, len(columns), 2)}
    logger.info(f"columns renaming mapping: {rename_dict}")
    renamed_data = data.rename(columns=rename_dict)
    logger.info("columns renamed successfully")
    return renamed_data


def columns_text2code(columns, data, batch_size=1000): # assigns unique code to text, female = 1, male = 0, female = , etc 
    mappings = {}
    for column in columns:
        unique_values = data[column].dropna().unique()
        mappings[column] = {value: idx for idx, value in enumerate(unique_values)}
        for value, code in mappings[column].items():
            logging.info(f'{code} -> {value}')
    total_rows = data.shape[0]
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch_data = data.iloc[start:end].copy()
        for column in columns:
            if column in batch_data.columns:
                batch_data[column] = batch_data[column].map(mappings[column])
        data.iloc[start:end] = batch_data
    return data


def date_converter(data, columns, expected_format, batch_size=1000): # convert dates to the expected format  
    logger.info(f"convert dates for columns: {columns}, to the expected format: {expected_format}")
    date_patterns = [
        (r'(\d{2})-(\d{2})-(\d{2})', r'\1/\2/20\3'),               # Format dd-mm-yy
        (r'(\d{2})/(\d{2})/(\d{4})', r'\1/\2/\3'),                 # Format dd/mm/yyyy
        (r'(\d{4})-(\d{2})-(\d{2})', r'\2/\3/\1'),                 # Format yyyy-mm-dd
        (r'(\d{2})\.(\d{2})\.(\d{4})', r'\1/\2/\3'),               # Format dd.mm.yyyy
        (r'(\d{4})\.(\d{2})\.(\d{2})', r'\2/\3/\1'),               # Format yyyy.mm.dd
        (r'(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2})', r'\1/\2/\3 \4:\5'), # dd/mm/yyyy hh:mm
        (r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})', r'\2/\3/\1 \4:\5'), # yyyy-mm-dd hh:mm
        (r'(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2}):(\d{2})', r'\1/\2/\3 \4:\5:\6'), # dd/mm/yyyy hh:mm:ss
        (r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})', r'\2/\3/\1 \4:\5:\6'), # yyyy-mm-dd hh:mm:ss
        (r'(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2}):(\d{2})\.(\d{3})', r'\1/\2/\3 \4:\5:\6.\7'), # dd/mm/yyyy hh:mm:ss.ms
        (r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d{3})', r'\2/\3/\1 \4:\5:\6.\7'), # yyyy-mm-dd hh:mm:ss.ms
    ]
    for column in columns:
        if column in data.columns:
            total_rows = data.shape[0]
            for start in range(0, total_rows, batch_size):
                end = min(start + batch_size, total_rows)
                batch_data = data.loc[start:end, column].copy()
                for pattern, replacement in date_patterns:
                    batch_data = batch_data.str.replace(pattern, replacement, regex=True)
                batch_data = pd.to_datetime(batch_data, format=expected_format, errors='coerce')
                data.loc[start:end, column] = batch_data.dt.strftime('%m/%d/%Y')
    return data


def date_converter_to_float(data, columns, batch_size=1000): # convert date yyyy-mm-dd to float
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for column in columns:
            if column in data.columns:
                total_rows = data.shape[0]
                for start in range(0, total_rows, batch_size):
                    end = min(start + batch_size, total_rows)
                    batch_data = data.loc[start:end, column].copy()
                    batch_data = pd.to_datetime(batch_data, errors='coerce', format='%Y-%m-%d')
                    data.loc[start:end, column] = batch_data.map(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    return data







