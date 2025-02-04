import os
from os.path import join as pjoin
import warnings
import warnings
# TODO: Test if this removes the warnings!
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

#  SETUP PARAMETERS
BASE_DIRECTORY = os.getcwd()
STUDIES_DATABASE = 'db'
TCR_DATABASES = {'tcrdb': 'tcrdb'}
OBJECTS_DATABASE = 'objects'
OBJECTS_TYPES = ['studies']
INIT_STUDIES = ['PRJNA393498', 'immunoSEQ47']

#  REQUESTS CONSTANTS

TIMEOUT = 10.0
WAIT_TIME = 1.0
RETRIES = 10
RETRY_STATUS_LIST = [429, 500, 502, 503, 504]
DEFAULT_HEADER = "https://"

#URLs

TCRDB_DOWNLOAD_URL = "https://guolab.wchscu.cn/TCRdb/Download/"

#  DIRECTORIES
TCR_DB_PATH = pjoin(STUDIES_DATABASE, TCR_DATABASES['tcrdb'])
STUDY_SAVE_DIR = pjoin(OBJECTS_DATABASE, 'studies')
INDEX = 'index.json'
