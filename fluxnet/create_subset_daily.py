import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SITES = [
    "AU-ASM",
    "BR-Npw",
    "CA-ARB",
    "CH-Dav",
    "CN-Cha",
    "DK-Sor",
    "IT-Ro1",
    "RU-Fyo",
    "US-NR1",
    "ZM-Mon",
]


if __name__ == "__main__":
    folder_path = os.path.join(BASE_DIR, "data_cleaned")
    data_path = os.path.join(folder_path, "daily.csv")
    data = pd.read_csv(data_path, index_col=0).reset_index(drop=True)
    data10 = data[data["site_id"].isin(SITES)]
    data10.to_csv(os.path.join(folder_path, "daily10.csv"))
