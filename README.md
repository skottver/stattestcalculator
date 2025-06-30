# StatTestCalculator

A Python class for statistical hypothesis testing in signal/background models using both Monte Carlo and asymptotic methods. Designed for statistical analysis of high energy physics experimental data.
Quick tutorial can be found in stc_tutorial.ipynb
## Features
- **Significance testing**: Calculate p-values and significance (Z-score)
- **Upper limits**: Compute upper bounds on signal strength at given confidence level
- **Systematic uncertainties**: Support for log-normal & Gaussian priors (correlated/uncorrelated)
- **Visualization**: Plot test statistic distributions and data models
- **Methods**: Hybrid (Bayesian-Frequentist) and frequentist approaches

## Dependencies
- NumPy
- SciPy
- Matplotlib

## Installation
```bash
pip install numpy scipy matplotlib
```
## Class initialization
```python
from stat_test_calculator import StatTestCalculator

StatTestCalculator(
    signal_hist=None,       # Expected signal counts per bin
    background_hist=None,   # Expected background counts per bin
    data=None,              # Observed data counts per bin
    delta=None,             # Fractional background uncertainty (e.g., 0.1 for 10%)
    delta_s=None,           # Fractional signal uncertainty
    test_stat_type='q',     # Test stat: 'q' (q0/qμ) or 'clsb' (CLs+b)
    mode='significance',    # 'significance' or 'upper_limits'
    prior='lognormal',      # 'lognormal' or 'gauss' for systematics
    ntoys=10000,            # Number of Monte Carlo toys
    ul_grid=None,           # Grid for upper limit scan (e.g., np.linspace(0,5,11))
    cl=0.95,                # Confidence level
    verbose=True,           # Print progress
    method='hybrid',        # 'hybrid' (default) or 'frequent'
    syst_type='uncorrelated' # 'correlated' or 'uncorrelated' systematics
)
```
## Key Methods
```python
import_data(data, signal_hist, background_hist)
```
Update datasets after initialization
Parameters:

    data: Observed counts array

    signal_hist: Expected signal counts

    background_hist: Expected background counts

```python
monte_carlo_hypotest(test_stat_type=None)
```
Run Monte Carlo hypothesis test
Returns:

    For significance mode: (p-value, Z-score)

    For upper limits: Signal strength upper limit
    Updates internal state: Test statistic distributions
```python
asymptotic_hypotest(...)
```
Calculate results using asymptotic approximations

Parameters (optional overrides):

    data, signal_hist, background_hist, delta, cl, mode, prior
    Returns:

    Significance (σ) or upper limit (μ)
```python
plot_hypo_test(path=None)
```
Visualize test results

    Significance mode: Distribution of test statistic under null hypothesis

    Upper limits mode: p-value vs signal strength curve
    Parameters:

    path: Save figure to file if provided
```python
plot_data(...)
```
Visualize signal/background models and data

Parameters (optional overrides):

    data, signal_hist, background_hist, delta, delta_s

    plot_signal/plot_background/plot_data: Toggle components

    path: Save figure to file
