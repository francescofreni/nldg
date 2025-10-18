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
        default=None,
        help="Number of sites in the subset. If None, use all sites.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Filter dataset by year (e.g. 2020). If None, use all years.",
    )
    parser.add_argument(
        "--country_code",
        type=str,
        default=None,
        help="First two letters of the site ID (e.g. 'US'). If None, use all countries.",
    )
    parser.add_argument(
        "--agg",
        type=str,
        choices=["seasonal", "daily", "raw"],
        default="daily",
        help="Data aggregation level",
    )
    args = parser.parse_args()
    nsites = args.nsites
    year = args.year
    country_code = args.country_code
    agg = args.agg

    folder_path = os.path.join(BASE_DIR, "data_cleaned")
    data_path = os.path.join(folder_path, agg + ".csv")
    data = pd.read_csv(data_path, index_col=0).reset_index(drop=True)

    if year is not None:
        data = data[data["year"] == year]

    if country_code is not None:
        data = data[data["site_id"].str.startswith(f"{country_code}-")]

    if nsites is not None:
        if year is None and nsites == 10:
            data = data[data["site_id"].isin(SITES10)]
        elif year is None and nsites == 30:
            data = data[data["site_id"].isin(SITES30)]
        else:
            if agg == "daily":
                unique_sites = data["site_id"].unique()
                if nsites > len(unique_sites):
                    raise ValueError(
                        f"Requested nsites={args.nsites} but only {len(unique_sites)} unique sites available."
                    )
                rng = np.random.default_rng(SEED)
                sampled_sites = rng.choice(
                    unique_sites, size=nsites, replace=False
                )
                data = data[data["site_id"].isin(sampled_sites)]
            else:
                daily_path = "daily"
                if nsites is not None:
                    daily_path += f"-{nsites}"
                if year is not None:
                    daily_path += f"-{year}"
                daily_path += ".csv"
                # if daily_path does not exist, raise error
                if not os.path.exists(os.path.join(folder_path, daily_path)):
                    raise ValueError(
                        f"File {daily_path} does not exist. Please create the subset for daily data first."
                    )
                daily_data = pd.read_csv(
                    os.path.join(folder_path, daily_path), index_col=0
                ).reset_index(drop=True)
                unique_sites = daily_data["site_id"].unique()
                data = data[data["site_id"].isin(unique_sites)]

    if year is None and nsites is None and country_code is None:
        print(
            "No country code, sites or year selected, the dataset is unchanged."
        )
    elif year is not None and nsites is None and country_code is None:
        data.to_csv(os.path.join(folder_path, f"{agg}-{year}.csv"))
    elif year is None and nsites is not None and country_code is None:
        data.to_csv(os.path.join(folder_path, f"{agg}-{nsites}.csv"))
    elif year is None and nsites is None and country_code is not None:
        data.to_csv(os.path.join(folder_path, f"{agg}-{country_code}.csv"))
    elif year is not None and nsites is not None and country_code is None:
        data.to_csv(os.path.join(folder_path, f"{agg}-{nsites}-{year}.csv"))
    elif year is not None and nsites is None and country_code is not None:
        data.to_csv(
            os.path.join(folder_path, f"{agg}-{country_code}-{year}.csv")
        )
    elif year is None and nsites is not None and country_code is not None:
        data.to_csv(
            os.path.join(folder_path, f"{agg}-{country_code}-{nsites}.csv")
        )
    elif year is not None and nsites is not None and country_code is not None:
        data.to_csv(
            os.path.join(
                folder_path, f"{agg}-{country_code}-{nsites}-{year}.csv"
            )
        )
