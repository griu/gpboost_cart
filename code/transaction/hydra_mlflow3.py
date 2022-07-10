# conda activate gpboost_env
# python code/transaction/hydra_mlflow3.py model.ROUNDS=5
# https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.lightgbm.LightGBMTuner.html
# https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_tuner_simple.py
# https://github.com/optuna/optuna-examples/blob/main/lightgbm/lightgbm_simple.py
# https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5
# https://www.kaggle.com/code/bjoernholzhauer/lightgbm-tuning-with-optuna/notebook
# git clone --recursive https://github.com/fabsig/GPBoost
# cd GPBoost
# mkdir build
# cd build
# cmake ..
# make -j4
# cd ..
# cd python-package
# python setup.py sdist
# cd dist
# python -m pip install gpboost-0.7.7.tar.gz
 
import gc
import time
import numpy as np
import pandas as pd
#import lightgbm as lgb
import optuna.integration.lightgbm as lgb
import optuna
 
from lightgbm import early_stopping
from lightgbm import log_evaluation
 
import os
 
from omegaconf import DictConfig, OmegaConf, ListConfig, open_dict
import hydra
 
import json
import mlflow
from mlflow.models.signature import infer_signature
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
 
 
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
 
# lightgbm help functions
 
def lgb_feature_importance(model, sort=True, output="df"):
    """
    alternatively output could be 'dict'
    """
    df = pd.DataFrame({'Feature':model.feature_name(), 'Importance':100*model.feature_importance()/np.sum(model.feature_importance()) })
    if sort:
        df = df.sort_values("Importance", ascending=False)
   
    res = {}
    if output=="dict":
        for i,row in df.iterrows():
            res['imp_'+row.Feature] = row.Importance
    else:
        res = df
    return res
 
 
# metricas
 
def RMSE_func(real, estim):
    return np.sqrt(np.mean((real - estim)*(real - estim)))
 
def MQ_func(real, estim,q):
    return np.mean(np.where(real>=estim,q*(real - estim),(1-q)*(estim - real)))
 
def PQ_func(real, estim,q):
    return np.mean(np.where(real>=estim,1,0))
 
 
@hydra.main(version_base=None, config_path="../../config/", config_name="params")
def my_app(cfg):
    data = pd.read_csv("data/output/transaction/pan.csv")
    #data = pd.read_pickle("data/output/transaction/pan.csv")
   

    params = {
        'task': 'train'
        ,'boosting_type': 'gbdt'
        ,'objective': 'quantile'
        ,'metric': 'quantile'
        ,'seed': 42
        #,'early_stopping_round ': 10
        ,'alpha' : cfg.model.QUANT_PARAM
        #,'num_threads':8
        , 'verbosity': -1
    }
 
 
    study_tuner = optuna.create_study(direction='minimize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    
    X_train = data[(data.yearmonth < 7)&(data.yearmonth > 4)].drop(['amount'], axis=1)
    Y_train = data[(data.yearmonth < 7)&(data.yearmonth > 4)]['amount']
    X_valid = data[data.yearmonth == 7].drop(['amount'], axis=1)
    Y_valid = data[data.yearmonth == 7]['amount']
    X_test = data[data.yearmonth == 8].drop(['amount'], axis=1)
 
    d_train = lgb.Dataset(X_train,
                      label=Y_train,
                      categorical_feature=["user_id","transac_code"])
 
    d_valid = lgb.Dataset(X_valid,
                      label=Y_valid,
                      categorical_feature=["user_id","transac_code"])
 
    d_test = lgb.Dataset(X_test,
                      categorical_feature=["user_id","transac_code"])
 
 
    client = mlflow.tracking.MlflowClient()
    #mlflow.set_tracking_uri(cfg.mlflow.url)
    mlflow.set_experiment(cfg.mlflow.exp_name)
    #experiment_id = client.create_experiment(cfg.mlflow.exp_name)
 
    # Fetch experiment metadata information
    #experiment = client.get_experiment(experiment_id)

    with mlflow.start_run() as run:
        mlflow.lightgbm.autolog(log_models=False,log_input_examples=False)
        #evals={}
        bst = lgb.train(params,d_train, num_boost_round=cfg.model.ROUNDS,valid_sets= [d_train,d_valid],
                            study=study_tuner,
                            time_budget=cfg.model.budget_time,
#                            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
                            optuna_seed=cfg.model.optuna_seed,
                            model_dir=cfg.model.dir)
        #log_params_from_omegaconf_dict(params)
        #mlflow.log_param("ROUNDS", "100")
       
        preds_train = bst.predict(X_train,num_iteration=bst.best_iteration)
        preds_valid = bst.predict(X_valid,num_iteration=bst.best_iteration)
        #for i,v in enumerate(evals["train1"]["quantile"]):
        #    mlflow.log_metric("mq_train_step",v, step=i+1)
        #for i,v in enumerate(evals["valid1"]["quantile"]):
        #    mlflow.log_metric("mq_valid_step",v, step=i+1)
        #
        #fig = lgb.plot_metric(evals);
 
        #mlflow.log_figure(fig, 'importance_plor.png')
 
 
        mlflow.set_tag("mode", "train")
 
        #mlflow.log_metric("rmse_train" , RMSE_func(Y_train,preds_train))
        #mlflow.log_metric("rmse_valid" , RMSE_func(Y_valid,preds_valid))
 
        mlflow.log_metric("mq_train",  MQ_func(Y_train,preds_train,cfg.model.QUANT_PARAM))
        mlflow.log_metric("mq_valid",  MQ_func(Y_valid,preds_valid,cfg.model.QUANT_PARAM))
 
        mlflow.log_metric("pq_train",  PQ_func(Y_train,preds_train,cfg.model.QUANT_PARAM))
        mlflow.log_metric("pq_valid",  PQ_func(Y_valid,preds_valid,cfg.model.QUANT_PARAM))
 
        #dict_importance = lgb_feature_importance(bst, output="dict")
        #mlflow.log_metrics(dict_importance)
       
        
if __name__ == "__main__":
    my_app()
 
   
#    best_params = model.params
#    print("Best params:", best_params)
#    print("  Accuracy = {}".format(accuracy))
#    print("  Params: ")
#    for key, value in best_params.items():
#        print("    {}: {}".format(key, value))   
 
 
 
# com guardar iteracions
# com fer amb funcio objectiu
# qincorporar gpboost
# incporar lectura escriptura
