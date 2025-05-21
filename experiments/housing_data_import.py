import argparse
import os
from sklearn.datasets import fetch_california_housing

# Define constants for default paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
os.makedirs(data_dir, exist_ok=True)
DEFAULT_PATH = os.path.join(data_dir, "housing-sklearn.csv")


def save_california_housing_to_csv(output_path: str):
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X["MedHouseVal"] = y
    X.to_csv(path_or_buf=output_path, index=False)
    print(f"Saved California housing data to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save California housing data to a CSV file."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_PATH,
        help="Path to where the CSV will be saved.",
    )
    args = parser.parse_args()

    save_california_housing_to_csv(args.output_path)
