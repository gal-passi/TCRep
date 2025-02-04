import os.path

from definitions import *
import wget
from Curation import build_study
import pandas as pd


def create_directories():
    print("Creating directories...")
    os.makedirs(STUDIES_DATABASE, exist_ok=True)
    os.chdir(pjoin(BASE_DIRECTORY, STUDIES_DATABASE))
    for db in TCR_DATABASES.values():
        os.makedirs(db, exist_ok=True)
        with open(pjoin(db, INDEX), 'w') as _:
            pass
    os.chdir(BASE_DIRECTORY)
    os.makedirs(OBJECTS_DATABASE, exist_ok=True)
    os.chdir(pjoin(BASE_DIRECTORY, OBJECTS_DATABASE))
    for db in OBJECTS_TYPES:
        os.makedirs(db, exist_ok=True)
    os.chdir(BASE_DIRECTORY)
    print('done')


def download_studies():
    print('downloading studies...')
    for study_id in INIT_STUDIES:
        study_file = study_id + '.tsv'
        url = TCRDB_DOWNLOAD_URL + study_file
        if os.path.exists(pjoin(TCR_DB_PATH, study_file)):
            print(f"Study {study_file} already exists")
            continue
        print(f"Downloading: {url}")
        try:
            wget.download(url, out=TCR_DB_PATH)
        except Exception as e:
            print(f"Failed to download {study_file}: {e}\n"
                  f"Please download the file manually from {url} and place it in {TCR_DB_PATH}")
    print('done')


def get_study_df(study_id, columns):
    # Read the file (e.x. "db/tcrdb/PRJNA393498.txt")
    with open(os.path.join(STUDIES_DATABASE, TCR_DATABASES['tcrdb'], f"{study_id}.txt"), "r") as f:
        lines = [line.strip() for line in f if line.strip()]  # Remove empty lines

    # Parse the data in chunks of len(lines)+1 lines (1 line per attribute)
    # Note that the first line is useless, because it is just the column number
    data = []
    for i in range(0, len(lines), len(columns)+1):
        entry = lines[i:i + len(columns)+1]
        for j in range(1, len(entry)):
            if columns[j-1] == "Read Length":
                entry[j] = int(entry[j])
            elif columns[j-1] == "Bases (M)":
                entry[j] = float(entry[j])
        data.append(entry[1:])

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    return df


if __name__ == '__main__':
    # Create directories and downloading studies
    create_directories()
    download_studies()

    study_id = "PRJNA393498"
    columns = ["Sample ID", "Cell Source", "Cell Type", "Condition",
               "Comment", "Read Length", "Bases (M)", "LibraryLayout"]
    study_df = get_study_df(study_id, columns)

    # Build AS study:
    # Building the dataset has different strategies for each study (according to the Study ID).
    build_study(study_id, study_df,
                "This study was aimed to search for AS-specific T cell receptor (TCR) variants, to determine the phenotype and involvement of corresponding T-cells in joint inflammation",
                [['SRR5812617'], ['SRR5812618'], ['SRR5812627'],
                 ['SRR5812653'], ['SRR5812656'], ['SRR5812663'],
                 ['SRR5812665'], ['SRR5812666'], ['SRR5812676']],
                [],
                [['SRR5812612'], ['SRR5812623'], ['SRR5812669'], ['SRR5812671'],
                 ['SRR5812637'], ['SRR5812668'], ['SRR5812657'], ['SRR5812640'],
                 ['SRR5812672'], ['SRR5812616'], ['SRR5812687'], ['SRR5812648'],
                 ['SRR5812651'], ['SRR5812677'], ['SRR5812626'], ['SRR5812643'],
                 ['SRR5812624'], ['SRR5812644'], ['SRR5812645'], ['SRR5812678'],
                 ['SRR5812655'], ['SRR5812683'], ['SRR5812682'], ['SRR5812613'],
                 ['SRR5812667'], ['SRR5812625'], ['SRR5812649'], ['SRR5812674'],
                 ['SRR5812611'], ['SRR5812610'], ['SRR5812684'], ['SRR5812622'],
                 ['SRR5812654'], ['SRR5812658'], ['SRR5812686'], ['SRR5812662'],
                 ['SRR5812636'], ['SRR5812660'], ['SRR5812633'], ['SRR5812679'],
                 ['SRR5812634'], ['SRR5812646'], ['SRR5812635'], ['SRR5812620'],
                 ['SRR5812681'], ['SRR5812652'], ['SRR5812685'], ['SRR5812675'],
                 ['SRR5812614'], ['SRR5812680'], ['SRR5812642'], ['SRR5812621'],
                 ['SRR5812630'], ['SRR5812650'], ['SRR5812664'], ['SRR5812639'],
                 ['SRR5812670'], ['SRR5812659'], ['SRR5812638'], ['SRR5812629'],
                 ['SRR5812641'], ['SRR5812661'], ['SRR5812673'], ['SRR5812631'],
                 ['SRR5812647'], ['SRR5812619'], ['SRR5812632'], ['SRR5812628'],
                 ['SRR5812615']])


    study_id = "immunoSEQ47"
    columns = ["Sample ID", "Cell Source", "Cell Type", "Condition", "Comment"]
    study_df = get_study_df(study_id, columns)
    build_study(study_id, study_df, "Multifactorial Heterogeneity of Virus-specific T Cells and Association with the Progression of Human Chronic Hepatitis B Infection",
                [], [], [])
