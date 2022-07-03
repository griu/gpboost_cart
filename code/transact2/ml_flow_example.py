

import mlflow
import os
from omegaconf import DictConfig, OmegaConf, ListConfig, open_dict
#import hydra

# start new run
#mlflow.set_tracking_uri('file://' + os.getcwd() + '/mlruns')
#mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
tracking_uri = mlflow.get_tracking_uri()
print("Current tracking uri: {}".format(tracking_uri))

#file:///D:/griu/repogit/gpboost_cart/code/transact2/mlruns
#print ('file://' + os.getcwd() + '/mlruns')
experiment_name = "Gpboost"
experiment_id = mlflow.get_experiment_by_name(experiment_name)
#experiment_id = mlflow.create_experiment(experiment_name)
with mlflow.start_run(experiment_id=experiment_id):
  
  # log single key-value param
  mlflow.log_param("param1", 5)
  
  # log single key-value metric
  mlflow.log_metric("foo2", 2, step=1)
  mlflow.log_metric("foo2", 4, step=2)
  mlflow.log_metric("foo2", 6, step=3)
  
  with open("output.txt", "w") as f:
    f.write("Hello world!")
    
  # logs local file or directory as artifact,
  mlflow.log_artifact("output.txt")

