import pandas as pd
from pathlib import Path


def load_dataset(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty")

    return df
