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
│   ├── ca_housing_analysis          # California housing experiments
│   ├── ca_housing_data_import       # Script to import data
│   ├── parallel_test_runtime        # Experiment to check the effect of parallelization
│   ├── sim_diff_methods             # Comparing different variants of WORME Forest
│   ├── sim_gen_gap                  # Verifying generalization guarantees
│   ├── sim_mse_degeneration         # Guarantee for MSE fails with heteroskedastic noise
│   └── utils                        # Helper functions
|
├── nldg           
│   ├── nn                           # Neural Network class
│   ├── rf                           # Magging Random Forest class
│   ├── ss                           # WORME Smoothing Spline class
│   ├── train_nn                     # GDRO
│   └── utils                        # Helper functions: data generation, plotting, metrics
|
├── notebooks
│   ├── all_methods                  # Different variants of WORME Forest        
│   ├── demo_rf                      # Demo for WORME Forest
│   ├── demo_ss                      # Demo for WORME Smoothing Splines
│   └── miscellanea                  # Additional plots and settings
|
└── results
    ├── figures                      # Saved figures
    ├── output_ca_housing            # Saved results real-world data experiment
    └── output_simulation
        ├── sim_diff_methods         # Saved results variants comparison
        ├── sim_mse_degeneration     # Saved results MSE degeneration
        └── sim_gen_gap              # Saved results generalization guarantees
```


## 🧪 Running experiments

### Simulated data

#### 1) Comparing different WORME Forest implementations
```bash
python experiments/sim_diff_methods.py
```

#### 2) Generalization Guarantees
```bash
python experiments/sim_gen_gap.py
```

#### 3) Generalization Guarantee fails with MSE objective and heteroskedastic noise
```bash
python experiments/sim_mse_degeneration.py
```

#### 4) Runtime parallelization experiment (only if $50$+ cores are available!)
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

[//]: # (## 📚 Documentation)

[//]: # ()
[//]: # (As of now, the code does not have explicit documentation, but the code is heavily commented and should be easy to understand. )

[//]: # (The code is also automatically formatted using `black`, linted with `ruff`, and type-checked with `mypy`.)