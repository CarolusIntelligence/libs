from external_libs import logging
from config import *



logging.basicConfig(
    filename=LOGS_PATH, 
    filemode="a", # append mode
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # messages format
    level=logging.DEBUG) # log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)