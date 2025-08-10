<h1>Nonlinear Distribution Generalization</h1>
<p>
<img src="https://img.shields.io/badge/python-≥3.10-blue" alt="Python >= 3.10">
<a href="https://github.com/python/mypy"><img src="https://img.shields.io/badge/mypy-checked-2b507e" alt="Checked with mypy"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Linted with ruff"></a>
</p>


This repository contains the code for the Master's Thesis written by Francesco Freni under the supervision of Prof. Dr. Jonas Peters at ETH Zurich during Spring 2025.


## ⚙️ Installation

First, create a Python environment:
```bash
python -m venv venv_nldg
source venv_nldg/bin/activate  # On Windows use `venv_nldg\Scripts\activate`
```

This project depends on a **modified version** of the library [adaXT](https://github.com/NiklasPfister/adaXT) by [NiklasPfister](https://github.com/NiklasPfister), which is licensed under the BSD 3-Clause License.

We maintain a fork with necessary changes [here](https://github.com/francescofreni/adaXT). This version must be installed for this project to work as intended.

The code is organized as a Python package, and can be installed using `pip`:
```bash
git clone https://github.com/francescofreni/nldg.git
cd nldg
pip install .
cd ..
git clone https://github.com/francescofreni/adaXT.git
cd adaXT
pip install .
cd ../nldg  # On Windows use cd ..\nldg
```
To install it in editable mode (for modifying the code and seeing the changes immediately) and with developer dependencies (for testing and code formatting), use

```bash
pip install -e ".[dev]"
```


## 🚀 Usage
The code in `notebooks/demo_rf.ipynb` and `notebooks/demo_ss.ipynb` demonstrates the core functionalities of the main functions provided in this repository.


## 📁 Directory Structure
```plaintext
.
├── data                             # Processed data
|
├── experiments 
│   ├── bcd_test_runtime.py          # Comparison between BCD and default variants
│   ├── ca_housing_analysis.py       # California housing experiments
│   ├── ca_housing_data_import.py    # Script to import data
│   ├── parallel_test_runtime.py     # Experiment to check the effect of parallelization
│   ├── sim_diff_methods.py          # Comparing different variants of MaxRM Random Forest
│   ├── sim_gen_gap.py               # Verifying generalization guarantees
│   ├── sim_mse_degeneration.py      # Guarantee for MSE fails with heteroskedastic noise
│   ├── sim_smoothsplines.py         # Comparing MaxRM Smoothing Splines with standard SS
│   └── utils.py                     # Helper functions
|
├── fluxnet
│   ├── data                         # Raw data
│   ├── data_cleaned                 # Aggregated datasets after preprocessing 
│   ├── results                      # Experimental results
│   ├── create_subset_daily.py       # Script to create a subset of the aggregated daily data
│   ├── dataloader.py                # Creates train and test splits
│   ├── eval.py                      # Evaluation metrics
│   ├── preprocessing.py             # Preprocessing raw data
│   └── run_experiment.py            # Run the fluxnet experiment
|
├── nldg           
│   ├── nn.py                        # Neural Network class
│   ├── rf.py                        # Magging Random Forest class
│   ├── ss.py                        # MaxRM Smoothing Spline class
│   ├── train_nn.py                  # GDRO
│   └── utils.py                     # Helper functions: data generation, plotting, metrics
|
├── notebooks
│   ├── all_methods.ipynb            # Different variants of MaxRM Random Forest        
│   ├── demo_rf.ipynb                # Demo for MaxRM Random Forest
│   ├── demo_ss.ipynb                # Demo for MaxRM Smoothing Splines
│   └── miscellanea.ipynb            # Additional plots and settings
|
└── results
    ├── figures                      # Saved figures
    ├── output_ca_housing            # Saved results real-world data experiment
    └── output_simulation
        ├── sim_diff_methods         # Saved results variants comparison
        ├── sim_mse_degeneration     # Saved results MSE degeneration
        ├── sim_gen_gap              # Saved results generalization guarantees
        └── sim_smoothsplines        # Saved splines simulation
```


## 🧪 Running experiments

### Simulated data

#### 1) Comparing different MaxRM Random Forest implementations
```bash
python experiments/sim_diff_methods.py
```

#### 2) Comparing MaxRM Smoothing Splines against standard Smoothing Splines
```bash
python experiments/sim_smoothsplines.py
```

#### 3) Generalization Guarantees
```bash
python experiments/sim_gen_gap.py
```
With covariate shift:
```bash
python experiments/sim_gen_gap.py --covariate_shift "different"
```
Or replace `"different"` with `"mixture"`.

#### 4) Generalization Guarantee fails with MSE objective and heteroskedastic noise
```bash
python experiments/sim_mse_degeneration.py
```

#### 5) Runtime parallelization experiment (only if $50$+ cores are available!)
```bash
python experiments/parallel_test_runtime.py
```

### California Housing

#### 1) Runtime BCD algorithm
```bash
python experiments/bcd_test_runtime.py
```

#### 2) Main experiment
```bash
python experiments/ca_housing_data_import.py
python experiments/ca_housing_analysis.py
```

### FLUXNET
See https://github.com/anyafries/fluxnet_bench for more details.

The data needs to be copied in the folder `fluxnet/data`. To obtain the cleaned raw, daily and seasonal datasets, run the following:
```bash
python fluxnet/preprocessing.py
```
The aggregated datasets will be available in `fluxnet/data_cleaned`. To create a subset of the daily data with $10$ sites, run the following:
```bash
python fluxnet/create_subset_daily.py --nsites 10
```
Alternatively, you can also create a subset with $30$ sites. The resulting dataset will show up in `fluxnet/data_cleaned`.

To run the experiment with $10$ sites:
```bash
python fluxnet/run_experiment.py --agg "daily10" --setting "loso" --model_name "rf"
python fluxnet/run_experiment.py --agg "daily10" --setting "loso" --model_name "rf" --method "maxrm" --risk "mse"
python fluxnet/run_experiment.py --agg "daily10" --setting "loso" --model_name "rf" --method "maxrm" --risk "reward"
python fluxnet/run_experiment.py --agg "daily10" --setting "loso" --model_name "rf" --method "maxrm" --risk "regret"
```
To run the experiment with $30$ sites, replace `"daily10"` with `"daily30"`. The results will be available at `fluxnet/results`. 

To get a LaTeX table summarizing the results, run:
```bash
python fluxnet/eval.py --agg "daily10" --setting "loso" --metric "rmse"
```
If, instead of the RMSE, you would like to report the $R^2$, replace `"rmse"` with `"r2_score"`.

[//]: # (## 📚 Documentation)

[//]: # ()
[//]: # (As of now, the code does not have explicit documentation, but the code is heavily commented and should be easy to understand. )

[//]: # (The code is also automatically formatted using `black`, linted with `ruff`, and type-checked with `mypy`.)