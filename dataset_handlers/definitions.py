import os
from os.path import join as pjoin
import warnings
# TODO: Test if this removes the warnings!
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

#  SETUP PARAMETERS
BASE_DIRECTORY = os.getcwd()
STUDIES_DATABASE = 'data/db'
TCR_DATABASES = {'tcrdb': 'tcrdb', 'tcrdb2': 'tcrdb2'}
OBJECTS_DATABASE = 'data/objects'
OBJECTS_TYPES = ['studies']

#  REQUESTS CONSTANTS
TIMEOUT = 10.0
WAIT_TIME = 1.0
RETRIES = 10
RETRY_STATUS_LIST = [429, 500, 502, 503, 504]
DEFAULT_HEADER = "https://"

#  DIRECTORIES
TCR_DB_PATH = pjoin(STUDIES_DATABASE, TCR_DATABASES['tcrdb'])
TCR_DB2_PATH = pjoin(STUDIES_DATABASE, TCR_DATABASES['tcrdb2'])
STUDY_SAVE_DIR = pjoin(OBJECTS_DATABASE, 'studies')
INDEX = 'index.json'
