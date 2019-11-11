import os


PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
DATA_RAW_PATH = os.path.join(DATA_PATH, 'raw')
DATA_CLEAN_PATH = os.path.join(DATA_PATH, 'clean')
