# vmo2_mlops_tech_test

VMO2 MLOPs tech test.

## Introduction

This project is an MLOPs exercise. It uses XGBoost model to predict which animals would be adopted.

The project contains two main components:
  - feature engineering, model composition and training in [src/train.py](src/train.py)
  - inference in [src/predict.py](src/predict.py)

An exploratory notebook is in [notebooks/mlops_tech_test.ipynb](notebooks/mlops_tech_test.ipynb)


You can see the full list of available make commands by typing:

```
make
```

## Install

1. Install [poetry](https://python-poetry.org/docs/#installation)

2. Install requirements
```
make update_dependencies
```

3. Ensure environment is setup by running tests
```
make tests
```

## Run

1. Run train
```
poetry run python src/train.py
```

2. Run predict
```
poetry run python src/predict.py
```
