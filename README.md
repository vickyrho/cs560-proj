# Formal Verification and Auto-Correction of Regression Models

This repository contains the implementation of a project integrating formal verification techniques with regression models to improve prediction reliability and robustness. The project uses the Z3 solver to enforce domain-specific constraints and includes datasets for housing prices and used car values.

## Features
- **Regression Models**: Implementation for predicting housing and used car prices.
- **Z3 Solver Integration**: Enforces constraints on inputs and predictions to ensure logical consistency.
- **Synthetic Data Generation**: Improves robustness for edge cases.
- **Preprocessed Datasets**: Includes both housing and used car datasets.
- **Dockerized Setup**: A Docker image for consistent and reproducible environments.

## Files in Repository
- `housing_price_dataset.csv` and `used_cars_india.csv`: Datasets for housing and used car predictions.
- `housing_z3_autocorrect.ipynb` and `used_cars_notebook.ipynb`: Jupyter notebooks demonstrating preprocessing, modeling, and Z3 integration for both datasets.
- `existing_sol_cars.csv` and `sampled_points_cars.csv`: Example outputs and sampled data points.
- `server.py`: Backend setup (if applicable).
- `Dockerfile`: Docker configuration for the project.
- `requirements.txt`: Python dependencies.

