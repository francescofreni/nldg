<h1>MaxRM Random Forest</h1>
<p>
<img src="https://img.shields.io/badge/python-≥3.10-blue" alt="Python >= 3.10">
<a href="https://github.com/python/mypy"><img src="https://img.shields.io/badge/mypy-checked-2b507e" alt="Checked with mypy"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Linted with ruff"></a>
</p>


This repository contains the code for the Master's Thesis written by Francesco Freni under the supervision of Prof. Dr. Jonas Peters and Anya Fries at ETH Zurich during Spring 2025.


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
├── data                                # Processed data (for California housing)
|
├── experiments 
│   ├── additional
│   |   ├── comparison_drol.py          # Comparison with DRoL (Appendix F.2)
│   |   ├── comparison_gdro.py          # Comparison with group DRO (Section 6.1.2)
│   |   ├── comparison_intro.py         # Produce the plot in the introduction (Figure 1)
│   |   ├── comparison_magging.ipynb    # Comparison with magging (Appendix B)
│   |   ├── datasets_comparison.py      # Comparison between MaxRM-RF and RF on different datasets (Appendix F.4)
│   |   └── tree_rf_comparison.py       # Comparison between MaxRM-RF and MaxRM-RT (Appendix F.3)
│   ├── bcd_test_runtime.py             # Comparison between BCD and default variants (Appendix F.5.2)
│   ├── ca_housing_analysis.py          # California housing experiment (Section 6.2.1)
│   ├── ca_housing_data_import.py       # Script to import California housing data
│   ├── ca_housing_hyperparameters.py   # California housing - varying hyperparameters (Appendix F.6)
│   ├── parallel_test_runtime.py        # Experiment to check the effect of parallelization
│   ├── sim_diff_methods.py             # Comparing different variants of MaxRM Random Forest (Section 6.1.1)
│   ├── sim_gen_gap.py                  # Verifying generalization guarantees
│   ├── sim_mse_degeneration.py         # MSE fails with heteroskedastic noise
│   ├── sim_smoothsplines.py            # Comparing MaxRM Smoothing Splines with standard SS (Appendix F.1)
│   └── utils.py                        # Helper functions
|
├── fluxnet
│   ├── cv
│   |   ├── dataloader.py               # Creates train and test splits
│   |   └── run_experiment.py           # Run the fluxnet experiment with CV (final version) (Section 6.2.2)
│   ├── data                            # Raw data
│   ├── data_cleaned                    # Aggregated datasets after preprocessing 
│   ├── results                         # Experimental results
│   ├── create_subset_daily.py          # Script to create a subset of the aggregated daily data
│   ├── dataloader.py                   # Creates train and test splits
│   ├── eval.py                         # Evaluation metrics
│   ├── preprocessing.py                # Preprocessing raw data
│   ├── run_experiment.py               # Run the fluxnet experiment
│   └── run_experiment_distance.py      # Run the fluxnet experiment with distance to convex hull
|
├── nldg           
│   ├── nn.py                           # Neural Network class
│   ├── rf.py                           # Magging Random Forest class
│   ├── ss.py                           # MaxRM Smoothing Spline class
│   ├── train_nn.py                     # GDRO
│   └── utils.py                        # Helper functions: data generation, plotting, metrics
|
├── notebooks
│   ├── all_methods.ipynb               # Different variants of MaxRM Random Forest        
│   ├── demo_rf.ipynb                   # Demo for MaxRM Random Forest
│   ├── demo_ss.ipynb                   # Demo for MaxRM Smoothing Splines
│   └── miscellanea.ipynb               # Additional plots and settings
|
└── results
    ├── figures                         # Saved figures
    ├── output_additional               # Saved results additional experiment
    |   ├── comparison_drol             # Saved results - comparison with DRoL
    |   └── comparison_gdro             # Saved results - comparison with group DRO
    ├── output_ca_housing               # Saved results California Housing experiment
    |   └── hyperparameters             # Saved results varying hyperparameters experiment
    └── output_simulation
        ├── sim_diff_methods            # Saved results variants comparison
        ├── sim_mse_degeneration        # Saved results MSE degeneration
        ├── sim_gen_gap                 # Saved results generalization guarantees
        └── sim_smoothsplines           # Saved splines simulation
```


## 🧪 Running experiments

### Section 1
```bash
python experiments/additional/comparison_intro.py
```
The results are saved in `results/figures/`.

### Section 6.1.1 + Appendix F.5.1
```bash
python experiments/sim_diff_methods.py
```
The results are saved in `results/output_simulation/sim_diff_methods/`.

### Section 6.1.2
```bash
python experiments/additional/comparison_gdro.py --risk "reward"
```
You can also set `n_jobs`. Add `--change_X_distr` if you want the test covariate distribution to be different from that of the training environments

The results are saved in `results/output_additional/comparison_gdro/`.

### Section 6.2.1
```bash
python experiments/ca_housing_data_import.py
python experiments/ca_housing_analysis.py
```
The results are saved in `results/output_ca_housing/`.

### Section 6.2.2
See https://pad.gwdg.de/s/yuCtk9fj5 for more details.

The data needs to be copied in the folder `fluxnet/data`. To obtain the cleaned raw, daily and seasonal datasets, run the following:
```bash
python fluxnet/preprocessing.py
```
The aggregated datasets will be available in `fluxnet/data_cleaned`. To create the subset of the daily data used in the experiments, run the following:
```bash
python fluxnet/create_subset_daily.py --nsites 50 --year 2017
```
The resulting dataset will show up in `fluxnet/data_cleaned`.

To run the experiments under the L5SO setting (only if $20$+ cores are available!):
```bash
python fluxnet/run_experiment.py --agg "daily-50-2017" --setting "l5so" --model_name "rf"
python fluxnet/run_experiment.py --agg "daily-50-2017" --setting "l5so" --model_name "rf" --method "maxrm" --risk "mse"
python fluxnet/run_experiment.py --agg "daily-50-2017" --setting "l5so" --model_name "rf" --method "maxrm" --risk "reward"
python fluxnet/run_experiment.py --agg "daily-50-2017" --setting "l5so" --model_name "rf" --method "maxrm" --risk "regret"
```
To use the LOGO strategy, replace `"l5so"` with `"logo"`. 

To use linear regression, run:
```bash
python fluxnet/run_experiment.py --agg "daily-50-2017" --setting "l5so" --model_name "lr"
```
The results will be available at `fluxnet/results`. 


### Appendix B
See `experiment/additional/comparison_magging.ipynb`.


### Appendix F.1
```bash
python experiments/sim_smoothsplines.py
```
The results are saved in `results/output_simulation/sim_smoothsplines/`.


### Appendix F.2
```bash
python experiments/additional/comparison_drol.py --risk "reward"
```
You can also set `n_jobs`. Add `--change_X_distr` if you want the test covariate distribution to be different from that of the training environments

The results are saved in `results/output_additional/comparison_drol/`.


### Appendix F.3
```bash
python experiments/additional/tree_rf_comparison.py
```
The results are saved in `results/output_additional/`.


### Appendix F.4
```bash
python experiments/additional/datasets_comparison.py
```
The results are saved in `results/output_additional/`.


### Appendix F.5.2
```bash
python experiments/bcd_test_runtime.py
```
The results are saved in `results/figures/`.


### Appendix F.6
```bash
python experiments/ca_housing_data_import.py
python experiments/ca_housing_hyperparameters.py
```
The results are saved in `results/output_ca_housing/hyperparameters/`.


### Miscellanea

#### Generalization Guarantees
```bash
python experiments/sim_gen_gap.py
```
With change in covariate distribution:
```bash
python experiments/sim_gen_gap.py --change_X_distr "different"
```
You can also replace `"different"` with `"mixture"`.

The results are saved in `results/output_simulation/sim_gen_gap`.

#### Verifying that the guarantee does not hold for the MSE with heteroskedastic noise
```bash
python experiments/sim_mse_degeneration.py
```
The results are saved in `results/output_simulation/sim_mse_degeneration`.

#### Runtime parallelization experiment (only if $50$+ cores are available!)
```bash
python experiments/parallel_test_runtime.py
```
The results are saved in `results/figures/`.
