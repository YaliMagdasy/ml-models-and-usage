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
    

# Standardizes DataFrame columns and string values to lowercase snake_case for consistency across datasets.
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    new_columns = []
    for col in df.columns:
        res = ""
        for i, char in enumerate(col.strip()):
            if i > 0 and char.isupper() and not col[i-1].isupper() and col[i-1] != '_' and col[i-1] != '-':
                res += "_" + char.lower()
            else:
                res += char.lower()
        clean_name = res.replace(' ', '_').replace('.', '_').replace('-', '_').replace('__', '_')
        new_columns.append(clean_name)
    df.columns = new_columns

    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].apply(
            lambda x: "".join(
                ["_" + c.lower() if i > 0 and c.isupper() and not str(x)[i-1].isupper() and str(x)[i-1] not in ['_', '-'] 
                 else c.lower() for i, c in enumerate(str(x).strip())]
            ).replace(' ', '_').replace('.', '_').replace('-', '_').replace('__', '_') if pd.notnull(x) else x
        )
    return df