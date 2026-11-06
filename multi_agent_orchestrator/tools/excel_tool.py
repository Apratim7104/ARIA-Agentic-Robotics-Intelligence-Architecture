import pandas as pd

def read_excel(filepath: str) -> str:
    df = pd.read_excel(filepath)
    return df.to_string()
