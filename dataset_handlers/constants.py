import os
from os.path import join as pjoin
import warnings
# TODO: Test if this removes the warnings!
from Bio import BiopythonDeprecationWarning
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Study IDs
STUDY_ID = 'PRJNA393498'  # Ankylosing Spondylitis study
STUDY_ID2 = 'immunoSEQ47'  # Hepatitis B virus study
STUDY_ID3 = 'immunoSEQ77'  # Rheumatoid arthritis study (plus healthy)
STUDY_ID4 = 'PRJNA258001'  # HIV study (plus healthy)
STUDY_ID5 = 'PRJNA390125'  # Only healthy study
STUDY_ID6 = 'PRJNA495603'  # Multiple sclerosis study (plus healthy)
STUDY_ID7 = 'PRJNA579190'  #  Multiple sclerosis study (plus healthy)
STUDY_ID8 = 'PRJNA280417'  #  Multiple sclerosis study
STUDY_ID9 = 'PRJNA427746'  #  Cytomegalovirus (plus healthy)
STUDY_ID10 = 'PRJNA318421'  #  Cytomegalovirus
STUDY_ID11 = 'PRJNA473147'  #  Cytomegalovirus
STUDY_ID12 = 'PRJNA273698'  # Healthy
STUDY_ID13 = 'immunoSEQ139'  # Cancer and Healthy
STUDY_ID14 = 'immunoSEQ21'  # Healthy
STUDY_ID15 = 'immunoSEQ54'  # Alopecia Areata and Healthy
STUDY_ID16 = 'immunoSEQ68'  # Cytomegalovirus
HEALTHY_STUDY_ID = STUDY_ID3  # ONLY CD8
HEALTHY_STUDY_ID2 = STUDY_ID4  # Both CD8 and CD4
HEALTHY_STUDY_ID3 = STUDY_ID5  # Larger both CD8 and CD4 (But fewer patients!)
HEALTHY_STUDY_ID4 = STUDY_ID6  # Other healthy study
HEALTHY_STUDY_ID5 = STUDY_ID7  # Other healthy study
HEALTHY_STUDY_ID6 = STUDY_ID12
HEALTHY_STUDY_ID7 = STUDY_ID13
HEALTHY_STUDY_ID8 = STUDY_ID14
HEALTHY_STUDY_ID9 = STUDY_ID15
STUDIES = [STUDY_ID, STUDY_ID2, STUDY_ID3, STUDY_ID4, STUDY_ID5, STUDY_ID6, STUDY_ID7]
TCRDB2_PATH = 'data/db/tcrdb2'


#  SETUP PARAMETERS
BASE_DIRECTORY = os.path.dirname(os.getcwd())
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
