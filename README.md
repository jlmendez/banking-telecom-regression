# Regression Diagnostics for Banking & Telecom

Applied regression and numerical-linear-algebra toolkit for multicollinearity, Ridge stabilization and lag relationships in banking and telecommunications data.

## Highlights

- Ordinary least squares in matrix form
- Variance Inflation Factor (VIF)
- Singular Value Decomposition and condition number
- Closed-form Ridge regression
- Cross-correlation for lagged operational relationships
- Synthetic banking and telecom case studies

## Tech stack

Python · NumPy · pandas · SciPy · scikit-learn

## Repository structure

- `src/regression_diagnostics.py` — reusable OLS/VIF/SVD/Ridge utilities
- `src/case_studies.py` — reproducible banking and telecom demonstrations
- `requirements.txt`

## Run

```bash
pip install -r requirements.txt
python src/case_studies.py
```

The examples use synthetic data so the numerical behavior can be reproduced without proprietary banking or network datasets.
