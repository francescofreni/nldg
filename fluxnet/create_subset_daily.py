import os
import argparse
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 0
SITES10 = [
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
SITES30 = [
    "AR-SLu",  # Argentina
    "AT-Neu",  # Austria
    "AU-ASM",  # Australia
    "BE-Bra",  # Belgium
    "BR-Npw",  # Brazil
    "CA-ARB",  # Canada
    "CH-Dav",  # Switzerland
    "CN-Cha",  # China
    "CZ-BK1",  # Czech Republic
    "DE-Geb",  # Germany
    "DK-Sor",  # Denmark
    "ES-Abr",  # Spain
    "FI-Hyy",  # Finland
    "FR-LBr",  # France
    "GF-Guy",  # French Guiana
    "GH-Ank",  # Ghana
    "GL-ZaF",  # Greenland
    "IL-Yat",  # Israel
    "IT-Ro1",  # Italy
    "JP-SMF",  # Japan
    "MX-Tes",  # Mexico
    "MY-PSO",  # Malaysia
    "NL-Loo",  # Netherlands
    "PA-SPn",  # Panama
    "RU-Fyo",  # Russia
    "SD-Dem",  # Sudan
    "SE-Nor",  # Sweden
    "SJ-Adv",  # Svalbard
    "US-NR1",  # United States
    "ZM-Mon",  # Zambia
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nsites",
        type=int,
        default=10,
        help="Number of sites in the subset.",
    )
    args = parser.parse_args()
    nsites = args.nsites

    folder_path = os.path.join(BASE_DIR, "data_cleaned")
    data_path = os.path.join(folder_path, "daily.csv")
    data = pd.read_csv(data_path, index_col=0).reset_index(drop=True)

    if nsites == 10:
        subset = data[data["site_id"].isin(SITES10)]
    elif nsites == 30:
        subset = data[data["site_id"].isin(SITES30)]
    else:
        unique_sites = data["site_id"].unique()
        if nsites > len(unique_sites):
            raise ValueError(
                f"Requested nsites={args.nsites} but only {len(unique_sites)} unique sites available."
            )
        rng = np.random.default_rng(SEED)
        sampled_sites = rng.choice(unique_sites, size=nsites, replace=False)
        subset = data[data["site_id"].isin(sampled_sites)]

    subset.to_csv(os.path.join(folder_path, f"daily{nsites}.csv"))
