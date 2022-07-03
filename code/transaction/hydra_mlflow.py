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
    data = pd.read_pickle("../../data/output/transaction/panel_data2.pkl", compression='zip' )
    X_train = data[(data.yearmonth < 7)&(data.yearmonth > 4)].drop(['amount'], axis=1)
    Y_train = data[(data.yearmonth < 7)&(data.yearmonth > 4)]['amount']
    X_valid = data[data.yearmonth == 7].drop(['amount'], axis=1)
    Y_valid = data[data.yearmonth == 7]['amount']
    X_test = data[data.yearmonth == 8].drop(['amount'], axis=1)
    #del data

    d_train = lgb.Dataset(X_train,
                          label=Y_train,
                          categorical_feature=["user_id","transac_code"])
    
    d_valid = lgb.Dataset(X_valid,
                          label=Y_valid,
                          categorical_feature=["user_id","transac_code"])

    params = {
    'task': 'train'
    ,'boosting_type': 'gbdt'
    ,'objective': 'quantile'
    ,'metric': 'quantile'
    ,'num_leaves': 70
    ,'max_depth': 8
    ,'feature_fraction': 0.8
    ,'bagging_fraction': 0.8
    ,'bagging_freq': 5
    ,'learning_rate': 0.01
    #,'min_data_in_leaf': 300
    ,'seed': 42
    #,'early_stopping_round ': 10
    ,'alpha' : 0.5}

    # load model
    #evals={}
    #with open_dict(cfg.params):
    #mlflow.lightgbm.autolog()
    #bst = lgb.train(params, d_train, cfg.model.ROUNDS, [d_train,d_valid], ["train1","valid1"]  ,callbacks = [lgb.record_evaluation(evals)] )
    #bst = lgb.train(params, d_train, ROUNDS, [d_train,d_valid], ["train1","valid1"] )
    #bstCV = lgb.cv(params, d_train, ROUNDS,nfold=5)


    #eval_plot = lgb.plot_metric(evals);

    ## start new run
    #mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    #mlflow.set_experiment(cfg.mlflow.runname)
    with mlflow.start_run(experiment_id='2'):
        mlflow.lightgbm.autolog()
        bst = lgb.train(params, d_train, cfg.model.ROUNDS, [d_train,d_valid], ["train1","valid1"]  )

    #    #log_params_from_omegaconf_dict(cfg)
#
    #    for i,v in enumerate(evals["train1"]["quantile"]):
    #        mlflow.log_metric("mq_train_step",v, step=i+1)
    #    for i,v in enumerate(evals["valid1"]["quantile"]):
    #        mlflow.log_metric("mq_valid_step",v, step=i+1)
#
    #    preds_train = bst.predict(X_train)
    #    preds_valid = bst.predict(X_valid)
    #    
    #    # log metric
    #    mlflow.log_metric("rmse_train" , RMSE_func(Y_train,preds_train))
    #    mlflow.log_metric("rmse_valid" , RMSE_func(Y_valid,preds_valid))
    #    
    #    mlflow.log_metric("mq_train",  MQ_func(Y_train,preds_train,cfg.model.QUANT_PARAM))
    #    mlflow.log_metric("mq_valid",  MQ_func(Y_valid,preds_valid,cfg.model.QUANT_PARAM))
    #    
    #    mlflow.log_metric("pq_train",  PQ_func(Y_train,preds_train,cfg.model.QUANT_PARAM))
    #    mlflow.log_metric("pq_valid",  PQ_func(Y_valid,preds_valid,cfg.model.QUANT_PARAM))

if __name__ == "__main__":
    my_app()
