import pandas as pd
from pathlib import Path

def __get_path(filename):
    return Path(__file__).parent.parent.parent/'datasets'/(filename+'.csv')

def load_dataset(dataset_name):
    name = dataset_name.lower().strip()

    try:
        return pd.read_csv(__get_path(name))
    except FileNotFoundError:
        raise ValueError(f"Dataset '{dataset_name}' not found.")