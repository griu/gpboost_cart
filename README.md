# gpboost_cart

testint of hydra, optuna, mlflow, lightgbm amd gpboost with instacart data

## Environment creation

```{shell}
conda env create -f conda.yaml
python -m ipykernel install --user --name gpboost_env --display-name "gpboost_env 3.9"
conda install -c conda-forge pdpbox
```

## Git App

download from https://git-scm.com/download/win

cmd prompt


## mlflow activate server

from folder gpboost_cart

```{shell}
mlflow ui
# -> http://localhost:5000
```

More examples in : https://github.com/Toumash/mlflow-docker

## hydra_mlflow

```{shell}
python code/transaction/hydra_mlflow3.py model.ROUNDS=5

```

## tutorial lightgbm, optuna, mlflow, gpboost, hydra with transaction data 

code/transaction/lighgbm_transacion.ipynb

## CART example:

Look at the code in code/cart/cart.py

data from :

- https://www.kaggle.com/c/instacart-market-basket-analysis


## Quantile regression:

https://towardsdatascience.com/lightgbm-for-quantile-regression-4288d0bb23fd

## lightgbm

- https://lightgbm.readthedocs.io/en/latest/Parameters.html


## gpboost

- https://github.com/fabsig/GPBoost
- https://towardsdatascience.com/tree-boosted-mixed-effects-models-4df610b624cb
- panel data example: https://github.com/fabsig/GPBoost/blob/master/examples/python-guide/panel_data_example.py


## Optuna

- fast tunner for lightgbm:  https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.lightgbm.LightGBMTuner.html
- tutorial: https://www.kaggle.com/code/bjoernholzhauer/lightgbm-tuning-with-optuna/notebook
- Optuna Sweeper plugin | Hydra https://hydra.cc/docs/next/plugins/optuna_sweeper/


