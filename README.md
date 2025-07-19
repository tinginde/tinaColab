# TinaColab Experiments

This repository contains a few sample scripts for interacting with different language models. Experiment metrics are logged with [MLflow](https://mlflow.org/).

## Viewing MLflow Results

After running any of the scripts you can launch the MLflow UI to inspect runs:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open the displayed address in your browser to view parameters, metrics and logged responses.
