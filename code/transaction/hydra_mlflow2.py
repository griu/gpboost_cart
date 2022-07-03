import gc
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import os
from omegaconf import DictConfig, OmegaConf, ListConfig, open_dict
import hydra
import mlflow

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)


def RMSE_func(real, estim):
    return np.sqrt(np.mean((real - estim)*(real - estim)))

def MQ_func(real, estim,q):
    return np.mean(np.where(real>=estim,q*(real - estim),(1-q)*(estim - real)))

def PQ_func(real, estim,q):
    return np.mean(np.where(real>=estim,1,0))


@hydra.main(version_base=None, config_path="../../config/", config_name="params")
def my_app(cfg):
    with mlflow.start_run(experiment_id='1'):
        mlflow.log_param("beta", 3)
        mlflow.log_metric("rmse_train" , 33)

if __name__ == "__main__":
    my_app()
