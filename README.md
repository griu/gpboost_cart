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


## Comand line Calling

```{shell}
python train.py 

python train.py model.QUANT_PARAM=0.5

python train.py --multirun model.QUANT_PARAM=0.5,0.75 model.ROUNDS=100,1000
```

## mlflow example

from folder gpboost_cart

```{shell}
mlflow ui
# -> http://localhost:5000
```

Execute ml_flow_example.py.


## hydra_mlflow

```{shell}
python code/transaction/hydra_mlflow3.py model.ROUNDS=5

```



