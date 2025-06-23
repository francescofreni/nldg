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
The code in **notebooks/demo_rf.ipynb** and **notebooks/demo_ss.ipynb** demonstrates the core functionalities of the main functions provided in this repository.


## 📁 Directory Structure
```plaintext
.
├── data                             # Processed data
|
├── experiments 
│   ├── bcd_test_runtime             # Comparison between BCD and default variants
│   ├── housing_data_analysis        # California housing experiments
│   ├── housing_data_import          # Script to import data
│   ├── parallel_test_runtime        # Experiment to check the effect of parallelization
│   ├── results_rf                   # Plots for simulation and real-world data example
│   ├── simulation_rf                # Simulation experiments with Minimax Random Forest
│   └── utils                        # Helper functions: plotting
|
├── nldg           
│   ├── nn                           # Neural Network class
│   ├── rf                           # Magging Random Forest class
│   ├── ss                           # Minimax Smoothing Spline class
│   ├── train_nn                     # GDRO
│   └── utils                        # Helper functions: data generation, plotting, metrics
|
├── notebooks           
│   ├── demo_rf                      # Demo for Minimax Random Forest
│   ├── demo_ss                      # Demo for Minimax Smoothing Splines
│   └── minimax_rf                   # Different solutions to the Minimax Random Forest problem
|
└── results
    ├── figures                      # Saved figures
    ├── output_data_housing_rf       # Saved results real-world data experiment
    └── output_data_simulation_rf    # Saved results simulation experiment
```


## 🧪 Running experiments

### Simulated data

#### 1) Minimizing the max MSE over training environments
```bash
python experiments/simulation_rf.py
```

#### 2) Runtime parallelization experiment
```bash
python experiments/parallel_test_runtime.py
```

### California Housing

#### 1) Runtime BCD algorithm
```bash
python experiments/bcd_test_runtime.py
```

#### 2) RF and MinimaxRF comparison with different $m_\text{try}$ values
```bash
python experiments/housing_data_import.py
python experiments/housing_data_analysis.py --version "train_mtry_resample"
```

#### 3) Experiment with held-out data
```bash
python experiments/housing_data_import.py
python experiments/housing_data_analysis.py
```


[//]: # (## 📚 Documentation)

[//]: # ()
[//]: # (As of now, the code does not have explicit documentation, but the code is heavily commented and should be easy to understand. )

[//]: # (The code is also automatically formatted using `black`, linted with `ruff`, and type-checked with `mypy`.)