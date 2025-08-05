# Explore Data and Run Experiments: <br/>Sample Questions and Answers

Costa Rica

[![GitHub](https://img.shields.io/badge/--181717?logo=github&logoColor=ffffff)](https://github.com/)
[brown9804](https://github.com/brown9804)

Last updated: 2025-01-04

----------

> [!NOTE]
> The questions and answers provided in this study guide are for practice purposes only and are not official practice questions.
> They are intended to help you prepare for the [DP-100 Microsoft certification exam](https://learn.microsoft.com/en-us/credentials/certifications/azure-data-scientist-associate/).
> For additional preparation materials and the most up-to-date information, please refer to the [official Microsoft documentation](https://learn.microsoft.com/en-us/training/paths/explore-data-experiment-azure-machine-learning/).

## Topics Covered
- Explore and prepare data
- Create and run experiments
- Track experiments with MLflow
- Use Automated Machine Learning (AutoML)

<details>
<summary><b>List of References </b> (Click to expand)</summary>

- [Work with data in Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-access-data)
- [Track experiments with MLflow](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow)
- [What is automated machine learning?](https://learn.microsoft.com/en-us/azure/machine-learning/concept-automated-ml)
- [Configure AutoML experiments](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train)
- [Command jobs in Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-train-model)

</details>

<details>
<summary><b>List of questions/answers </b> (Click to expand)</summary>

- [Q1: Data Asset Types](#q1-data-asset-types)
- [Q2: MLflow Logging Functions](#q2-mlflow-logging-functions)
- [Q3: AutoML Configuration](#q3-automl-configuration)
- [Q4: Command Job Parameters](#q4-command-job-parameters)
- [Q5: Data Asset Paths](#q5-data-asset-paths)
- [Q6: AutoML Data Guardrails](#q6-automl-data-guardrails)
- [Q7: Experiment Tracking](#q7-experiment-tracking)
- [Q8: Data Access Methods](#q8-data-access-methods)
- [Q9: Environment Management](#q9-environment-management)
- [Q10: AutoML Limits](#q10-automl-limits)

</details>

> [!TIP]
> **Data Asset Types:**

| Asset Type | Use Case | Best For |
|------------|----------|----------|
| **FileDataset** | Raw files, images, documents | Unstructured data, computer vision tasks |
| **TabularDataset** | Structured row-column data | Traditional ML, CSV/Parquet files |
| **mltable** | Standardized dataset definition | v2 SDK, AutoML, reproducible pipelines |
| **uri_folder** | Reference to storage folder | Batch processing, file collections |

> [!TIP]
> **Supported Data Paths:**

| Path Type | Format | Example |
|-----------|--------|---------|
| **Local** | `./<path>` | `./data/train.csv` |
| **Azure Blob** | `wasbs://` | `wasbs://account.blob.core.windows.net/container/file` |
| **Azure Data Lake Gen2** | `abfss://` | `abfss://filesystem@account.dfs.core.windows.net/folder/file` |
| **Datastore** | `azureml://` | `azureml://datastores/datastore_name/paths/folder/file` |

> [!TIP]
> **MLflow Logging Functions:**

| Function | Purpose | Example Usage |
|----------|---------|---------------|
| **mlflow.log_param()** | Log single parameter | `mlflow.log_param("learning_rate", 0.01)` |
| **mlflow.log_metric()** | Log single metric | `mlflow.log_metric("accuracy", 0.95)` |
| **mlflow.log_artifact()** | Log file (plot, model) | `mlflow.log_artifact("confusion_matrix.png")` |
| **mlflow.log_model()** | Log ML model | `mlflow.sklearn.log_model(model, "model")` |

> [!TIP]
> **AutoML Experiment Limits:**

| Parameter | Description | Impact |
|-----------|-------------|---------|
| **timeout_minutes** | Total experiment duration | Prevents runaway experiments |
| **trial_timeout_minutes** | Maximum time per model trial | Controls individual model training time |
| **max_trials** | Maximum number of models to train | Balances exploration vs time |
| **enable_early_termination** | Stop if score isn't improving | Saves compute resources |

> [!TIP]
> **AutoML Data Guardrails:**

| Guardrail | Purpose | States |
|-----------|---------|--------|
| **Class Balancing Detection** | Identifies imbalanced datasets | Passed / Done / Alerted |
| **Missing Feature Values Imputation** | Handles missing data | Passed / Done / Alerted |
| **High Cardinality Feature Detection** | Identifies features with many unique values | Passed / Done / Alerted |

> [!TIP]
> **Command Job Configuration:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| **code** | Folder containing the script | ✅ Yes |
| **command** | Script to execute | ✅ Yes |
| **environment** | Runtime environment | ✅ Yes |
| **compute** | Compute target | ✅ Yes |
| **display_name** | Job display name | ❌ Optional |
| **experiment_name** | Experiment grouping | ❌ Optional |

## Q1: Data Asset Types

> You need to work with a collection of image files for a computer vision project. Which data asset type should you use?

- [ ] **TabularDataset** ❌: `Incorrect. TabularDataset is for structured, row-column data.`
- [ ] **mltable** ❌: `Incorrect. mltable is primarily for structured data with schema.`
- [ ] **FileDataset** ✅: `Correct. FileDataset is designed for working with raw files like images.`
- [ ] **uri_folder** ❌: `Incorrect. While possible, FileDataset provides better functionality for file collections.`

## Q2: MLflow Logging Functions

> You want to log the learning rate hyperparameter and accuracy metric for your experiment. Which MLflow functions should you use?

- [ ] **mlflow.log_artifact() for both** ❌: `Incorrect. log_artifact() is for files, not parameters or metrics.`
- [ ] **mlflow.log_param() for learning rate, mlflow.log_metric() for accuracy** ✅: `Correct. log_param() for hyperparameters, log_metric() for performance metrics.`
- [ ] **mlflow.log_metric() for both** ❌: `Incorrect. Parameters should use log_param().`
- [ ] **mlflow.log_model() for both** ❌: `Incorrect. log_model() is specifically for ML models.`

## Q3: AutoML Configuration

> You want to prevent AutoML experiments from running indefinitely. Which parameter should you configure?

- [ ] **max_trials** ❌: `Incorrect. This limits the number of models, not time.`
- [ ] **trial_timeout_minutes** ❌: `Incorrect. This limits individual trial time, not total experiment time.`
- [ ] **timeout_minutes** ✅: `Correct. This parameter sets the maximum duration for the entire AutoML experiment.`
- [ ] **enable_early_termination** ❌: `Incorrect. This helps stop underperforming trials, but doesn't set a time limit.`

## Q4: Command Job Parameters

> Which parameters are required when configuring a command job in Azure ML? (Select all that apply)

- [ ] **code** ✅: `Correct. The folder containing the script is required.`
- [ ] **command** ✅: `Correct. The command to execute is required.`
- [ ] **environment** ✅: `Correct. The runtime environment must be specified.`
- [ ] **compute** ✅: `Correct. The compute target must be specified.`
- [ ] **display_name** ❌: `Incorrect. Display name is optional.`

## Q5: Data Asset Paths

> Which path format should you use to reference a file in Azure Data Lake Storage Gen2?

- [ ] **`wasbs://account.blob.core.windows.net/container/file`** ❌: `Incorrect. This format is for Azure Blob Storage.`
- [ ] **`abfss://filesystem@account.dfs.core.windows.net/folder/file`** ✅: `Correct. This is the correct format for Azure Data Lake Storage Gen2.`
- [ ] **`azureml://datastores/name/paths/file`** ❌: `Incorrect. This format is for registered datastores.`
- [ ] **`./data/file.csv`** ❌: `Incorrect. This format is for local files.`

## Q6: AutoML Data Guardrails

> What are the three possible states for AutoML data guardrails?

- [ ] **Passed, Failed, Warning** ❌: `Incorrect. These are not the correct state names.`
- [ ] **Passed, Done, Alerted** ✅: `Correct. These are the three states for data guardrails.`
- [ ] **Success, Error, Pending** ❌: `Incorrect. These are not the data guardrail states.`
- [ ] **Complete, Incomplete, Skipped** ❌: `Incorrect. These are not the correct states.`

## Q7: Experiment Tracking

> You want to track multiple related machine learning experiments. What should you do?

- [ ] **Create a separate workspace for each experiment** ❌: `Incorrect. Workspaces are for broader organizational boundaries.`
- [ ] **Use the same experiment name for all related runs** ✅: `Correct. Experiment names group related runs together for comparison.`
- [ ] **Create a new MLClient for each experiment** ❌: `Incorrect. One MLClient can handle multiple experiments.`
- [ ] **Use different compute targets for each experiment** ❌: `Incorrect. Compute targets don't affect experiment organization.`

## Q8: Data Access Methods

> You have a large TabularDataset and want to process it with Spark. Which method should you use?

- [ ] **data.to_pandas_dataframe()** ❌: `Incorrect. Pandas is for in-memory processing, not suitable for large datasets.`
- [ ] **data.to_spark_dataframe()** ✅: `Correct. This converts the dataset to Spark DataFrame for distributed processing.`
- [ ] **data.get()** ❌: `Incorrect. This downloads files locally, not suitable for Spark processing.`
- [ ] **data.sample()** ❌: `Incorrect. This only returns a sample, not the full dataset.`

## Q9: Environment Management

> What is the best practice for managing environments across multiple experiments?

- [ ] **Create a new environment for each experiment** ❌: `Incorrect. This creates unnecessary duplication.`
- [ ] **Create and register an environment, then reuse it** ✅: `Correct. Registered environments ensure reproducibility and reduce maintenance.`
- [ ] **Always use curated environments** ❌: `Incorrect. Curated environments may not have all required packages.`
- [ ] **Share Dockerfile manually between team members** ❌: `Incorrect. This doesn't leverage Azure ML's environment management.`

## Q10: AutoML Limits

> You want to train a maximum of 50 models in your AutoML experiment, with each model taking no more than 15 minutes. How should you configure the limits?

- [ ] **max_trials=50, timeout_minutes=15** ❌: `Incorrect. timeout_minutes is for the entire experiment, not individual trials.`
- [ ] **max_trials=50, trial_timeout_minutes=15** ✅: `Correct. max_trials limits the number of models, trial_timeout_minutes limits each individual trial.`
- [ ] **timeout_minutes=50, trial_timeout_minutes=15** ❌: `Incorrect. timeout_minutes=50 would limit the entire experiment to 50 minutes.`
- [ ] **max_trials=15, timeout_minutes=50** ❌: `Incorrect. This reverses the intended limits.`

## Code Examples

### Creating and Logging to an Experiment

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set experiment
mlflow.set_experiment("customer-churn-prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Make predictions and log metrics
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
```

### Configuring a Command Job

```python
from azure.ai.ml import command
from azure.ai.ml.entities import Environment

# Define environment
env = Environment(
    name="sklearn-env",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest"
)

# Configure command job
job = command(
    code="./src",
    command="python train.py --data-path ${{inputs.training_data}} --model-name ${{inputs.model_name}}",
    inputs={
        "training_data": Input(type="uri_folder", path="azureml://datastores/workspaceblobstore/paths/data/"),
        "model_name": "random-forest"
    },
    environment=env,
    compute="cpu-cluster",
    experiment_name="model-training"
)

# Submit job
ml_client.jobs.create_or_update(job)
```

### AutoML Configuration

```python
from azure.ai.ml import automl

# Configure AutoML classification job
classification_job = automl.classification(
    compute="cpu-cluster",
    experiment_name="automl-classification",
    training_data=training_data,
    target_column_name="target",
    primary_metric="accuracy",
    n_cross_validations=5,
    enable_model_explainability=True,
    timeout_minutes=60,
    trial_timeout_minutes=20,
    max_trials=10,
    enable_early_termination=True
)

# Submit AutoML job
ml_client.jobs.create_or_update(classification_job)
```

## Best Practices

1. **Data Management**:
   - Use versioned datasets for reproducibility
   - Validate data quality before training
   - Monitor for data drift in production

2. **Experiment Tracking**:
   - Use descriptive experiment and run names
   - Log hyperparameters, metrics, and artifacts consistently
   - Tag runs with relevant metadata

3. **AutoML Optimization**:
   - Set appropriate timeout limits to control costs
   - Enable early termination for efficiency
   - Review data guardrails and address issues

4. **Environment Management**:
   - Register and version environments
   - Use minimal base images for faster startup
   - Test environments before production use

<!-- START BADGE -->
<div align="center">
  <img src="https://img.shields.io/badge/Total%20views-1290-limegreen" alt="Total views">
  <p>Refresh Date: 2025-08-04</p>
</div>
<!-- END BADGE -->
