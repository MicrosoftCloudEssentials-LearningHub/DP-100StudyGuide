# Train and Deploy Models: <br/> Sample Questions and Answers

Costa Rica

[![GitHub](https://img.shields.io/badge/--181717?logo=github&logoColor=ffffff)](https://github.com/)
[MicrosoftCloudEssentials-LearningHub](https://github.com/MicrosoftCloudEssentials-LearningHub)

Last updated: 2025-01-04

----------

> [!NOTE]
> The questions and answers provided in this study guide are for practice purposes only and are not official practice questions.
> They are intended to help you prepare for the [DP-100 Microsoft certification exam](https://learn.microsoft.com/en-us/credentials/certifications/azure-data-scientist-associate/).
> For additional preparation materials and the most up-to-date information, please refer to the [official Microsoft documentation](https://learn.microsoft.com/en-us/training/paths/train-deploy-models-azure-machine-learning/).

## Topics Covered

- Train machine learning models
- Optimize model performance with hyperparameter tuning
- Deploy models to endpoints
- Monitor and manage model deployments
- Implement responsible AI practices

<details>
<summary><b>List of References </b> (Click to expand)</summary>

- [Hyperparameter tuning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
- [Deploy models to online endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints)
- [Deploy models to batch endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-use-batch-endpoint)
- [Responsible AI dashboard](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard)
- [Model evaluation metrics](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-understand-automated-ml)

</details>

<details>
<summary><b>List of questions/answers </b> (Click to expand)</summary>

- [Q1: Hyperparameter Sampling Methods](#q1-hyperparameter-sampling-methods)
- [Q2: Early Termination Policies](#q2-early-termination-policies)
- [Q3: Online vs Batch Endpoints](#q3-online-vs-batch-endpoints)
- [Q4: Model Registration Types](#q4-model-registration-types)
- [Q5: Scoring Script Functions](#q5-scoring-script-functions)
- [Q6: Traffic Splitting](#q6-traffic-splitting)
- [Q7: Responsible AI Components](#q7-responsible-ai-components)
- [Q8: Model Evaluation Metrics](#q8-model-evaluation-metrics)
- [Q9: Deployment Configuration](#q9-deployment-configuration)
- [Q10: Batch Scoring Configuration](#q10-batch-scoring-configuration)

</details>

> [!TIP]
> **Hyperparameter Sampling Methods:**

| Method | Description | Parameter Types | Use Case |
|--------|-------------|-----------------|----------|
| **Grid Sampling** | Tries every combination | Discrete only | Small search space, exhaustive search |
| **Random Sampling** | Randomly selects values | Discrete & Continuous | Good balance of coverage and efficiency |
| **Sobol Sampling** | Quasi-random with better coverage | Continuous only | Large search spaces, reproducible results |
| **Bayesian Sampling** | Uses previous results to guide selection | Continuous | Expensive training, iterative improvement |

> [!TIP]
> **Early Termination Policies:**

| Policy | Parameters | Description |
|--------|------------|-------------|
| **Bandit Policy** | slack_factor, slack_amount | Terminates runs performing worse than (best_run - slack) |
| **Median Stopping** | None | Terminates runs performing worse than median of all runs |
| **Truncation Selection** | truncation_percentage | Terminates lowest performing percentage of runs |

> [!TIP]
> **Endpoint Types Comparison:**

| Endpoint Type | Use Case | Response Time | Scalability | Billing |
|---------------|----------|---------------|-------------|---------|
| **Online Endpoint** | Real-time inference | Low latency (ms) | Auto-scale based on load | Pay per hour (always on) |
| **Batch Endpoint** | Bulk processing | Asynchronous | Process large datasets | Pay per job execution |

> [!TIP]
> **Model Registration Types:**

| Type | Description | Best For |
|------|-------------|----------|
| **MLflow** | Models tracked with MLflow | Standard ML workflows, most common |
| **Custom** | Custom model format | Specialized models not supported by MLflow |
| **Triton** | NVIDIA Triton inference server | Deep learning, TensorFlow, PyTorch |

> [!TIP]
> **MLflow Model Signatures:**

| Signature Type | Input Format | Use Case |
|----------------|--------------|----------|
| **Column-based** | pandas.DataFrame | Tabular data, structured datasets |
| **Tensor-based** | numpy.ndarray | Images, text, unstructured data |

> [!TIP]
> **Responsible AI Principles:**

| Principle | Description |
|-----------|-------------|
| **Fairness and Inclusiveness** | Models treat everyone fairly without bias |
| **Reliability and Safety** | Models operate consistently and handle edge cases |
| **Privacy and Security** | Protect individual data and maintain confidentiality |
| **Transparency** | Users understand how decisions are made |
| **Accountability** | Human oversight and responsibility for AI decisions |

> [!TIP]
> **Model Evaluation Metrics:**

| Metric | Type | Formula | Best For |
|--------|------|---------|----------|
| **Accuracy** | Classification | (TP+TN)/(TP+TN+FP+FN) | Balanced datasets |
| **Precision** | Classification | TP/(TP+FP) | Minimizing false positives |
| **Recall** | Classification | TP/(TP+FN) | Minimizing false negatives |
| **F1-Score** | Classification | 2×(Precision×Recall)/(Precision+Recall) | Imbalanced datasets |
| **AUC** | Classification | Area under ROC curve | Ranking ability |
| **R²** | Regression | 1-(SS_res/SS_tot) | Variance explained |
| **RMSE** | Regression | √(MSE) | Error in original units |
| **MAE** | Regression | Σ\|y-ŷ\|/n | Robust to outliers |

## Q1: Hyperparameter Sampling Methods

> You have a large search space with continuous hyperparameters and want to ensure good coverage of the space. Which sampling method should you use?

- [ ] **Grid Sampling** ❌: `Incorrect. Grid sampling only works with discrete parameters.`
- [ ] **Random Sampling** ❌: `Incorrect. While it works, Sobol provides better coverage.`
- [ ] **Sobol Sampling** ✅: `Correct. Sobol sampling provides quasi-random coverage that's more uniform than random sampling.`
- [ ] **Bayesian Sampling** ❌: `Incorrect. Bayesian is better for expensive training scenarios, not necessarily large search spaces.`

## Q2: Early Termination Policies

> You want to terminate hyperparameter tuning runs that perform significantly worse than the best run. Which policy should you use?

- [ ] **Median Stopping Policy** ❌: `Incorrect. This compares against the median, not the best run.`
- [ ] **Truncation Selection Policy** ❌: `Incorrect. This terminates a percentage of runs, not based on comparison to the best.`
- [ ] **Bandit Policy** ✅: `Correct. Bandit policy terminates runs that perform worse than the best run by a specified slack amount.`
- [ ] **No early termination** ❌: `Incorrect. This doesn't address the requirement to terminate poor performers.`

## Q3: Online vs Batch Endpoints

> When should you use a batch endpoint instead of an online endpoint?

- [ ] **When you need real-time predictions with low latency** ❌: `Incorrect. This describes online endpoints.`
- [ ] **When you need to process large datasets asynchronously** ✅: `Correct. Batch endpoints are designed for processing large amounts of data without real-time requirements.`
- [ ] **When you need high availability and auto-scaling** ❌: `Incorrect. Online endpoints provide these features.`
- [ ] **When you need to serve multiple model versions simultaneously** ❌: `Incorrect. Both endpoint types can serve multiple versions.`

## Q4: Model Registration Types

> You have a TensorFlow deep learning model that you want to deploy with NVIDIA Triton Inference Server. Which model type should you register?

- [ ] **MLflow** ❌: `Incorrect. While MLflow supports TensorFlow, Triton provides better performance for deep learning inference.`
- [ ] **Custom** ❌: `Incorrect. Triton is a supported model type in Azure ML.`
- [ ] **Triton** ✅: `Correct. Triton model type is specifically designed for deep learning workloads with TensorFlow and PyTorch.`
- [ ] **Standard** ❌: `Incorrect. This is not a valid model registration type.`

## Q5: Scoring Script Functions

> Which functions are required in a scoring script for model deployment? (Select all that apply)

- [ ] **init()** ✅: `Correct. init() is called when the service starts and is used to load the model.`
- [ ] **run()** ✅: `Correct. run() is called for each inference request and performs the scoring.`
- [ ] **main()** ❌: `Incorrect. main() is not required for Azure ML scoring scripts.`
- [ ] **predict()** ❌: `Incorrect. predict() is not a required function name, though it may be used within run().`

## Q6: Traffic Splitting

> You want to gradually roll out a new model by sending 10% of traffic to the new version and 90% to the current version. How should you configure the traffic split?

- [ ] **{"current": 90, "new": 10}** ✅: `Correct. Traffic values represent percentages and must sum to 100.`
- [ ] **{"current": 0.9, "new": 0.1}** ❌: `Incorrect. Traffic values should be integers representing percentages, not decimals.`
- [ ] **{"current": 9, "new": 1}** ❌: `Incorrect. These values sum to 10, not 100.`
- [ ] **Set new deployment as default** ❌: `Incorrect. This would send all traffic to the new version.`

## Q7: Responsible AI Components

> Which Responsible AI components help you understand feature influence on predictions? (Select all that apply)

- [ ] **Add Explanation to RAI Insights dashboard** ✅: `Correct. Explanations show how features influence predictions.`
- [ ] **Add Causal to RAI Insights dashboard** ❌: `Incorrect. Causal analysis shows causal effects, not direct feature influence.`
- [ ] **Add Counterfactuals to RAI Insights dashboard** ❌: `Incorrect. Counterfactuals show how changes would affect output, not current feature influence.`
- [ ] **Add Error Analysis to RAI Insights dashboard** ❌: `Incorrect. Error analysis identifies error patterns, not feature influence.`

## Q8: Model Evaluation Metrics

> You're building a fraud detection model where missing actual fraud cases is very costly. Which metric should you optimize?

- [ ] **Precision** ❌: `Incorrect. Precision minimizes false positives, not false negatives.`
- [ ] **Recall** ✅: `Correct. Recall minimizes false negatives (missing fraud cases), which is critical for fraud detection.`
- [ ] **Accuracy** ❌: `Incorrect. Accuracy can be misleading with imbalanced fraud datasets.`
- [ ] **F1-Score** ❌: `Incorrect. While F1 balances precision and recall, recall is more important for this use case.`

## Q9: Deployment Configuration

> What information is required to deploy a model to a managed online endpoint? (Select all that apply)

- [ ] **Model assets** ✅: `Correct. The trained model files are required.`
- [ ] **Scoring script** ✅: `Correct. The script that loads the model and performs inference.`
- [ ] **Environment** ✅: `Correct. The runtime environment with necessary packages.`
- [ ] **Compute configuration** ✅: `Correct. VM size and instance count for the deployment.`

## Q10: Batch Scoring Configuration

> You're configuring a batch endpoint to process large datasets in parallel. Which parameters control the parallelization? (Select all that apply)

- [ ] **instance_count** ✅: `Correct. Number of compute nodes for parallel processing.`
- [ ] **max_concurrency_per_instance** ✅: `Correct. Number of parallel processes per compute node.`
- [ ] **mini_batch_size** ✅: `Correct. Number of files processed per scoring script run.`
- [ ] **output_action** ❌: `Incorrect. This controls output format, not parallelization.`

## Code Examples

### Hyperparameter Tuning with Sweep Job

```python
from azure.ai.ml import command
from azure.ai.ml.sweep import Choice, Uniform, BanditPolicy

# Define the command job template
job = command(
    code="./src",
    command="python train.py --learning_rate ${{search_space.learning_rate}} --n_estimators ${{search_space.n_estimators}}",
    environment="sklearn-env",
    compute="cpu-cluster"
)

# Configure sweep
sweep_job = job.sweep(
    compute="cpu-cluster",
    sampling_algorithm="random",
    primary_metric="accuracy",
    goal="Maximize",
    max_total_trials=20,
    max_concurrent_trials=4,
    early_termination=BanditPolicy(slack_factor=0.1, evaluation_interval=1, delay_evaluation=5)
)

# Define search space
sweep_job.search_space = {
    "learning_rate": Uniform(0.01, 0.1),
    "n_estimators": Choice([50, 100, 200])
}

# Submit sweep job
ml_client.jobs.create_or_update(sweep_job)
```

### Deploy Model to Online Endpoint

```python
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model, Environment

# Create endpoint
endpoint = ManagedOnlineEndpoint(
    name="fraud-detection-endpoint",
    auth_mode="key"
)

ml_client.online_endpoints.begin_create_or_update(endpoint)

# Create deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name="fraud-detection-endpoint",
    model=Model(path="./model"),
    code_configuration={
        "code": "./score",
        "scoring_script": "score.py"
    },
    environment=Environment(
        conda_file="./environment.yml",
        image="mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest"
    ),
    instance_type="Standard_DS3_v2",
    instance_count=1
)

ml_client.online_deployments.begin_create_or_update(deployment)

# Set traffic to deployment
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint)
```

### Batch Endpoint Configuration

```python
from azure.ai.ml.entities import BatchEndpoint, BatchDeployment

# Create batch endpoint
batch_endpoint = BatchEndpoint(
    name="batch-scoring-endpoint",
    description="Endpoint for batch scoring"
)

ml_client.batch_endpoints.begin_create_or_update(batch_endpoint)

# Create batch deployment
batch_deployment = BatchDeployment(
    name="batch-deployment",
    endpoint_name="batch-scoring-endpoint",
    model=Model(path="./model"),
    code_configuration={
        "code": "./score",
        "scoring_script": "batch_score.py"
    },
    environment=Environment(
        conda_file="./environment.yml",
        image="mcr.microsoft.com/azureml/minimal-ubuntu20.04-py38-cpu-inference:latest"
    ),
    compute="cpu-cluster",
    instance_count=4,
    max_concurrency_per_instance=2,
    mini_batch_size=10,
    output_action="append_row",
    output_file_name="predictions.csv"
)

ml_client.batch_deployments.begin_create_or_update(batch_deployment)
```

### Responsible AI Dashboard

```python
from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import PipelineJob

@pipeline()
def create_rai_pipeline(target_column_name, test_data, train_data):
    # RAI Insights constructor
    rai_constructor = rai_insights_constructor(
        title="Fraud Detection RAI Dashboard",
        task_type="classification",
        model_info_path=Input(type="uri_file", path="./model_info.json"),
        model_input=Input(type="uri_folder", path="./model"),
        test_dataset=test_data,
        target_column_name=target_column_name,
        categorical_column_names="[]"
    )
    
    # Add explanation component
    explanation = add_explanation(
        rai_insights_dashboard=rai_constructor.outputs.rai_insights_dashboard
    )
    
    # Add error analysis component
    error_analysis = add_error_analysis(
        rai_insights_dashboard=explanation.outputs.rai_insights_dashboard,
        max_depth=3
    )
    
    # Gather insights
    rai_gather = gather_rai_insights(
        constructor=rai_constructor.outputs.rai_insights_dashboard,
        insight_1=explanation.outputs.explanation,
        insight_2=error_analysis.outputs.error_analysis
    )
    
    return {
        "dashboard": rai_gather.outputs.dashboard,
        "ux_json": rai_gather.outputs.ux_json
    }

# Create and submit pipeline
pipeline_job = create_rai_pipeline(
    target_column_name="is_fraud",
    test_data=Input(type="uri_file", path="azureml://datastores/workspaceblobstore/paths/test_data.csv"),
    train_data=Input(type="uri_file", path="azureml://datastores/workspaceblobstore/paths/train_data.csv")
)

ml_client.jobs.create_or_update(pipeline_job)
```

## Best Practices

1. **Hyperparameter Tuning**:
   - Use appropriate sampling methods for your parameter types
   - Implement early termination to save compute costs
   - Start with broader ranges, then narrow down

2. **Model Deployment**:
   - Test deployments in staging before production
   - Use blue-green deployments for safe rollouts. E.g `you want to perform a canary rollout: send 10% of real-time inference requests to blue and 90% to green.` 
        - **“blue”** → the new candidate model  
        - **“green”** → the current production model  
   - Monitor endpoint performance and costs

3. **Responsible AI**:
   - Create RAI dashboards for high-impact models
   - Regular model fairness assessments
   - Document model limitations and assumptions

4. **Performance Optimization**:
   - Choose appropriate endpoint types for your use case
   - Configure parallelization for batch workloads
   - Monitor and adjust resource allocation

<!-- START BADGE -->
<div align="center">
  <img src="https://img.shields.io/badge/Total%20views-1290-limegreen" alt="Total views">
  <p>Refresh Date: 2025-08-04</p>
</div>
<!-- END BADGE -->
