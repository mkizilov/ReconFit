# Reconvolution Curve Fit

This repository contains Python scripts for performing reconvolution curve fitting using both custom NNLS-based and lmfit-based methods. These scripts are designed to analyze time-resolved fluorescence decay data by fitting exponential decay models convolved with an instrument response function (IRF).

## Features

- **Custom NNLS-based fitting**: Uses non-negative least squares (NNLS) for fitting decay curves.
- **lmfit-based fitting**: Utilizes the `lmfit` library for flexible model fitting with parameter constraints.
- **Grid fitting**: Splits the parameter space into equal intervals and performs fitting for each combination.
- **Data smoothing**: Optional data smoothing using Savitzky-Golay filter.

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/mkizilov/TCSCP-Reconvolution
cd reconvolution-curve-fit
pip install -r requirements.txt
```

# Usage
## Preparing Data

You can prepare and smooth your data using the prepare_data function:

```python
from utils import prepare_data

# Load your data into pandas DataFrames
df_signal = pd.read_csv('signal.csv')
df_irf = pd.read_csv('irf.csv')

# Prepare and smooth the data
signal, irf = prepare_data(df_signal, df_irf, window_length=5, polyorder=3)

```
## NNLS-based Reconvolution Fit
Use the reconvolution_curve_fit function for NNLS-based reconvolution fitting:

```python
from fit_functions import reconvolution_curve_fit

# Perform reconvolution curve fitting
popt, pcov = reconvolution_curve_fit(df_signal, df_irf, exp_num=3, tau_bounds=(0, 0.8), num_splits=10, plot_title='Exponential Decay Reconvolution', filename='fit_result')
```

## lmfit-based Reconvolution Fit
Use the reconvolution_lmfit function for lmfit-based reconvolution fitting:

```python
from fit_functions import reconvolution_lmfit

# Perform lmfit-based reconvolution fitting
best_result, best_values = reconvolution_lmfit(df_signal, df_irf, model_type='double', num_splits=10, plot_title='Exponential Decay Reconvolution')
```

# Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

# License

This project is licensed under the MIT License. See the LICENSE file for details.

This README snippet includes an introduction, installation instructions, usage examples, and details on how to contribute and the project's license. Adjust the content as needed based on the specific details of your project and repository.
