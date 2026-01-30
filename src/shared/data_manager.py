import pandas as pd
from pathlib import Path

DATASETS_DIR = Path(__file__).parents[2] / 'datasets'

def load_dataset(dataset_name: str) -> pd.DataFrame:
    name = dataset_name.lower().strip()
    file_path = DATASETS_DIR / f"{name}.csv"

    try:
        return pd.read_csv(file_path)

    except FileNotFoundError:
        
        all_files = [f.stem for f in DATASETS_DIR.glob('*.csv')]
        
        similar = [f for f in all_files if name in f or f in name]
        suggestions = similar if similar else all_files

        raise FileNotFoundError(f"Dataset '{name}' not found. Did you mean: {suggestions}?")