<h1>Nonlinear Distribution Generalization</h1>
<p>
<img src="https://img.shields.io/badge/python-≥3.10-blue" alt="Python >= 3.10">
<a href="https://github.com/python/mypy"><img src="https://img.shields.io/badge/mypy-checked-2b507e" alt="Checked with mypy"></a>
<a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Linted with ruff"></a>
</p>


This repository contains the code for the Master's Thesis written by Francesco Freni under the supervision of Prof. Dr. Jonas Peters at ETH Zurich during Spring 2025.


## 🚀 Installation

The code is organized as a Python package, and can be installed using `pip`:
```bash
git clone git@github.com:francescofreni/nldg.git
cd nldg
pip install -e .
```
To install it in editable mode (for modifying the code and seeing the changes immediately) and with developer dependencies (for testing and code formatting), replace the last line with:

```bash
pip install -e ".[dev]"
```


## 🔧 Usage
The code in **notebooks/demo.ipynb** demonstrates the core functionality of the main functions provided in this repository.


## 🧪 Running experiments

Running the experiments is as simple as:
```bash
python experiments/name_experiments_file.py
```


## 🚨 Tests

The code base comes with a set of unit tests, which can be run using `pytest`:

```bash
pytest tests
```


## 📚 Documentation

As of now, the code does not have explicit documentation, but the code is heavily commented and should be easy to understand. 
The code is also automatically formatted using `black`, linted with `ruff`, and type-checked with `mypy`.