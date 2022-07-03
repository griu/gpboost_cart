# https://www.mlflow.org/docs/latest/python_api/mlflow.html
import mlflow

# Create nested runs
with mlflow.start_run(run_name='PARENT_RUN') as parent_run:
    mlflow.log_param("parent", "yes")
    with mlflow.start_run(run_name='CHILD_RUN', nested=True) as child_run:
        mlflow.log_param("child", "yes")

print("parent run_id: {}".format(parent_run.info.run_id))
print("child run_id : {}".format(child_run.info.run_id))
print("--")

# Search all child runs with a parent id
query = "tags.mlflow.parentRunId = '{}'".format(parent_run.info.run_id)
results = mlflow.search_runs(filter_string=query)
print(results[["run_id", "params.child", "tags.mlflow.runName"]])