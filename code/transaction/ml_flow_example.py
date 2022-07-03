

import mlflow
import os
from omegaconf import DictConfig, OmegaConf, ListConfig, open_dict

# start new run
mlflow.set_tracking_uri('file://' + os.getcwd() + '/mlruns')
print ('file://' + os.getcwd() + '/mlruns')
mlflow.set_experiment(experiment_id='0')
with mlflow.start_run():
  
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

