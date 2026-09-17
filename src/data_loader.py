from pathlib import Path
import pandas as pd


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Loads raw retail dataset from a given CSV or Excel file path."""
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at specified location: '{path.resolve()}'")

    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format '{path.suffix}'. Expected .csv or .xlsx")