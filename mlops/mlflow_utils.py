import mlflow
from contextlib import contextmanager

@contextmanager
def start_run(run_name=None, experiment_name="default"):
    """Start an MLflow run with an optional experiment name."""
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        yield


def log_metrics(tokens=None, durations=None, model_name=None, prompt=None):
    """Log common parameters and metrics to MLflow."""
    if model_name is not None:
        mlflow.log_param("model_name", model_name)
    if prompt is not None:
        mlflow.log_param("prompt", prompt)

    if tokens:
        for name, value in tokens.items():
            mlflow.log_metric(name, value)

    if durations:
        for name, value in durations.items():
            mlflow.log_metric(name, value)

