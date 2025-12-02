<h1>MaxRM-RF: Maximum Risk Minimization with Random Forests</h1>
<p>
<img src="https://img.shields.io/badge/python-≥3.10-blue" alt="Python >= 3.10">
<a href="https://github.com/python/mypy"><img src="https://img.shields.io/badge/mypy-checked-2b507e" alt="Checked with mypy"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Linted with ruff"></a>
</p>

## ⚙️ Installation

First, create a Python environment:
```bash
python -m venv venv_nldg
source venv_nldg/bin/activate  # On Windows use `venv_nldg\Scripts\activate`
```

This project depends on a **modified version** of the library [adaXT](https://github.com/NiklasPfister/adaXT) by [NiklasPfister](https://github.com/NiklasPfister), which is licensed under the BSD 3-Clause License.
We maintain a fork with necessary changes [here](https://github.com/francescofreni/adaXT). This version must be installed for this project to work as intended, as it contains the implementation of MaxRM Random Forest.

This repository also contains the code for **MaxRM Smoothing Splines** (MaxRM-SS) and **MaxRM Additive Models** (MaxRM-AM).

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
The code in `notebooks/demo_rf.ipynb`, `notebooks/demo_ss.ipynb` and `notebooks/demo_gam.ipynb` demonstrates the core functionalities of the main functions provided in this repository.


## 🧪 Running experiments

### Section 1
To reproduce Fig. 1:
```bash
python experiments/additional/comparison_intro.py
```
The results are saved in `results/output_additional/`.


### Section 5.1
To reproduce Tab. 1:
```bash
python experiments/sim_diff_methods.py
```
The results are saved in `results/output_simulation/sim_diff_methods/`.


### Section 5.2
To reproduce Fig. 3: 
```bash
python experiments/comparison_gdro_magging.py
```
To reproduce Fig. 4: 
```bash
python experiments/comparison_gdro_magging.py --change_X_distr
```
The results are saved in `results/output_simulation/comparison_gdro_magging/`.

To reproduce Fig. 5: 
```bash
python experiments/comparison_equal_envs.py
```
The results are saved in `results/output_simulation/comparison_equal_envs/`.


### Section 6
**TBD**


### Appendix B
See `experiment/additional/comparison_magging.ipynb` (to see Fig. 7).


### Appendix D.1
To reproduce Tab. 3:
```bash
python experiments/additional/tree_rf_comparison.py
```
The results are saved in `results/output_additional/`.


### Appendix D.2
To reproduce Tab. 4 and Fig. 8:
```bash
python experiments/additional/sim_eg_bcd.py
```
The results are saved in `results/output_additional/sim_eg_bcd/`.


### Appendix D.3
To reproduce Fig. 9:
```bash
python experiments/additional/comparison_indeterminate_leaves.py
```
The results are saved in `results/output_additional/comparison_indeterminate_leaves`.


### Appendix D.4
To reproduce Fig. 10:
```bash
python experiments/additional/comparison_hyperparameters.py --change_X_distr
```
The results are saved in `results/output_additional/comparison_hyperparameters`.


### Appendix D.5
To reproduce Fig. 11: 
```bash
python experiments/comparison_gdro_magging.py --risk "reward"
python experiments/comparison_gdro_magging.py --risk "regret"
```
To reproduce Fig. 12: 
```bash
python experiments/comparison_gdro_magging.py --risk "reward" --change_X_distr
python experiments/comparison_gdro_magging.py --risk "regret" --change_X_distr
```
The results are saved in `results/output_simulation/comparison_gdro_magging/`.
