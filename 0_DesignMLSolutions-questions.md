# Design and Prepare Machine Learning Solutions: Sample Questions and Answers

Costa Rica

[![GitHub](https://img.shields.io/badge/--181717?logo=github&logoColor=ffffff)](https://github.com/)
[brown9804](https://github.com/brown9804)

Last updated: 2025-01-04

----------

> [!NOTE]
> The questions and answers provided in this study guide are for practice purposes only and are not official practice questions.
> They are intended to help you prepare for the [DP-100 Microsoft certification exam](https://learn.microsoft.com/en-us/credentials/certifications/azure-data-scientist-associate/).
> For additional preparation materials and the most up-to-date information, please refer to the [official Microsoft documentation](https://learn.microsoft.com/en-us/training/paths/design-machine-learning-solution/).

## Topics Covered
- Design machine learning solutions
- Set up Azure Machine Learning workspace
- Configure compute resources
- Manage workspace security and access

<details>
<summary><b>List of References </b> (Click to expand)</summary>

- [Azure Machine Learning workspace](https://learn.microsoft.com/en-us/azure/machine-learning/concept-workspace)
- [Manage compute instances](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-manage-compute-instance)
- [Set up authentication](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication)
- [Customer-managed keys](https://learn.microsoft.com/en-us/azure/machine-learning/concept-customer-managed-keys)
- [Azure Machine Learning roles](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-assign-roles)

</details>

<details>
<summary><b>List of questions/answers </b> (Click to expand)</summary>

- [Q1: Azure Machine Learning Workspace Components](#q1-azure-machine-learning-workspace-components)
- [Q2: Workspace Creation Methods](#q2-workspace-creation-methods)
- [Q3: Built-in Security Roles](#q3-built-in-security-roles)
- [Q4: Compute Instance vs Compute Cluster](#q4-compute-instance-vs-compute-cluster)
- [Q5: Customer Managed Keys](#q5-customer-managed-keys)
- [Q6: Workspace Assets](#q6-workspace-assets)
- [Q7: Azure CLI vs Python SDK](#q7-azure-cli-vs-python-sdk)
- [Q8: Compute Configuration Parameters](#q8-compute-configuration-parameters)
- [Q9: Environment Management](#q9-environment-management)
- [Q10: Authentication Methods](#q10-authentication-methods)

</details>

> [!TIP]
> **Azure Machine Learning Workspace Architecture:**

When a workspace is provisioned, Azure automatically creates supporting resources:

| Component | Purpose |
|-----------|---------|
| **Azure Storage Account** | Store files, notebooks, job metadata, and models |
| **Azure Key Vault** | Securely manage authentication keys and credentials |
| **Application Insights** | Monitor predictive services and track performance |
| **Azure Container Registry** | Store Docker images for ML environments (created when needed) |

> [!TIP]
> **Workspace Creation Methods:**

| Method | Use Case | Complexity |
|--------|----------|-----------|
| **Azure Portal** | Quick setup, visual interface | Low |  
| **ARM Template** | Infrastructure as Code, repeatable deployments | Medium |
| **Azure CLI** | Automation, scripting, CI/CD pipelines | Medium |
| **Python SDK** | Programmatic control, integration with ML workflows | High |

> [!TIP]
> **Azure ML Security Roles:**

| Role | Permissions | Best For |
|------|-------------|----------|
| **Owner** | Full access to all resources, can grant access to others | Workspace administrators |
| **Contributor** | Full access to all resources, cannot grant access | Senior data scientists |
| **Reader** | View-only access, cannot make changes | Stakeholders, auditors |
| **AzureML Data Scientist** | All actions except compute management and workspace settings | Standard data scientists |
| **AzureML Compute Operator** | Create, change, and manage compute resources | MLOps engineers |

> [!TIP]
> **Compute Types Comparison:**

| Compute Type | Use Case | Scaling | Cost Model |
|--------------|----------|---------|-----------|
| **Compute Instance** | Interactive development, Jupyter notebooks | Fixed size | Pay per hour while running |
| **Compute Cluster** | Training jobs, batch inference | Auto-scale (0-N nodes) | Pay per node per hour |
| **Kubernetes Cluster** | Production deployments, custom container orchestration | Manual configuration | Bring your own cluster |
| **Attached Compute** | Use existing resources (VMs, Databricks) | External management | External billing |
| **Serverless Compute** | On-demand training jobs | Fully managed | Pay per job execution |

> [!TIP]
> **Workspace Assets:**

| Asset Type | Description | Example Use Case |
|------------|-------------|------------------|
| **Models** | Trained ML models with metadata | Store trained models for deployment |
| **Environments** | Runtime configurations (packages, Docker images) | Ensure reproducible training and inference |
| **Data** | Datasets and data references | Manage training and validation datasets |
| **Components** | Reusable pipeline steps | Build modular ML pipelines |

> [!TIP]
> **Compute Cluster Configuration:**

| Parameter | Description | Considerations |
|-----------|-------------|----------------|
| **size** | VM type (CPU/GPU, memory, storage) | Match compute needs to workload |
| **max_instances** | Maximum number of nodes for scaling | Balance performance vs cost |
| **tier** | Priority level (Dedicated vs Low Priority) | Low Priority = 80% cost savings but no availability guarantee |

## Q1: Azure Machine Learning Workspace Components

> When you create an Azure Machine Learning workspace, which additional Azure resources are automatically created?

- [ ] **Azure Storage Account only** ❌: `Incorrect. Multiple resources are created automatically.`
- [ ] **Azure Key Vault and Application Insights only** ❌: `Incorrect. Additional resources are also created.`
- [ ] **Azure Storage Account, Azure Key Vault, Application Insights, and Azure Container Registry** ✅: `Correct. These four resources are automatically provisioned to support the workspace.`
- [ ] **Only the workspace resource itself** ❌: `Incorrect. Supporting resources are automatically created.`

## Q2: Workspace Creation Methods

> Which of the following are valid methods to create an Azure Machine Learning workspace? (Select all that apply)

- [ ] **Azure Portal UI** ✅: `Correct. You can create a workspace through the Azure portal interface.`
- [ ] **Azure Resource Manager (ARM) template** ✅: `Correct. ARM templates enable Infrastructure as Code for workspace creation.`
- [ ] **Azure CLI with ML extension** ✅: `Correct. The Azure CLI with the ML extension supports workspace creation.`
- [ ] **Python SDK** ✅: `Correct. The Azure ML Python SDK can programmatically create workspaces.`

## Q3: Built-in Security Roles

> You need to give a team member the ability to run experiments and manage datasets, but they should not be able to create or delete compute resources. Which role should you assign?

- [ ] **Owner** ❌: `Incorrect. Owner has full access including compute management.`
- [ ] **Contributor** ❌: `Incorrect. Contributor can manage compute resources.`
- [ ] **AzureML Data Scientist** ✅: `Correct. This role allows all workspace actions except compute management and workspace settings.`
- [ ] **AzureML Compute Operator** ❌: `Incorrect. This role is specifically for compute management.`

## Q4: Compute Instance vs Compute Cluster

> When should you use a Compute Instance versus a Compute Cluster?

- [ ] **Use Compute Instance for distributed training, Compute Cluster for development** ❌: `Incorrect. This is backwards.`
- [ ] **Use Compute Instance for interactive development, Compute Cluster for scalable training jobs** ✅: `Correct. Compute Instances are ideal for notebooks and experimentation, while Compute Clusters are for training at scale.`
- [ ] **They are interchangeable and serve the same purpose** ❌: `Incorrect. They have different use cases.`
- [ ] **Use Compute Cluster only for inference, Compute Instance only for training** ❌: `Incorrect. Both can be used for training.`

## Q5: Customer Managed Keys

> When implementing customer-managed keys (CMK) for workspace encryption, what information must you provide?

- [ ] **Only the Key Vault name** ❌: `Incorrect. More information is required.`
- [ ] **Key Vault resource ID and key URI** ✅: `Correct. Both the full Key Vault resource ID and the specific key URI are required.`
- [ ] **Only the key URI** ❌: `Incorrect. The Key Vault resource ID is also needed.`
- [ ] **Key Vault name and subscription ID** ❌: `Incorrect. The full resource ID and key URI are needed.`

## Q6: Workspace Assets

> Which of the following are considered assets in an Azure Machine Learning workspace? (Select all that apply)

- [ ] **Models** ✅: `Correct. Trained models are registered as assets in the workspace.`
- [ ] **Environments** ✅: `Correct. Runtime environments are managed as assets.`
- [ ] **Data** ✅: `Correct. Datasets and data references are workspace assets.`
- [ ] **Components** ✅: `Correct. Reusable pipeline components are assets.`

## Q7: Azure CLI vs Python SDK

> What are the main advantages of using Azure CLI with Azure Machine Learning? (Select all that apply)

- [ ] **Automate creation and configuration for repeatability** ✅: `Correct. CLI enables automation and scripting.`
- [ ] **Ensure consistency across multiple environments** ✅: `Correct. Scripts can be reused across dev, test, and production.`
- [ ] **Integrate with DevOps CI/CD pipelines** ✅: `Correct. CLI commands work well in automated pipelines.`
- [ ] **Provide interactive debugging capabilities** ❌: `Incorrect. Interactive debugging is better suited for SDKs or IDEs.`

## Q8: Compute Configuration Parameters

> When creating a compute cluster, which parameter controls the cost versus availability trade-off?

- [ ] **size** ❌: `Incorrect. Size affects performance and cost but not availability guarantees.`
- [ ] **max_instances** ❌: `Incorrect. This controls scaling but not availability guarantees.`
- [ ] **tier** ✅: `Correct. Setting tier to 'LowPriority' reduces costs by ~80% but provides no availability guarantee.`
- [ ] **location** ❌: `Incorrect. Location affects latency and compliance but not the cost/availability trade-off.`

## Q9: Environment Management

> What is the recommended approach for managing environments across multiple experiments?

- [ ] **Create a new environment for each experiment** ❌: `Incorrect. This creates unnecessary redundancy.`
- [ ] **Create and register an environment, then reuse it across experiments** ✅: `Correct. Registered environments promote reproducibility and reduce duplication.`
- [ ] **Always use the default environment** ❌: `Incorrect. Default environments may not have required packages.`
- [ ] **Share Dockerfiles manually between team members** ❌: `Incorrect. This doesn't leverage Azure ML's environment management capabilities.`

## Q10: Authentication Methods

> What are the three required parameters for authenticating to an Azure Machine Learning workspace using the Python SDK?

- [ ] **workspace_name, resource_group, location** ❌: `Incorrect. Location is not required for authentication.`
- [ ] **subscription_id, resource_group, workspace_name** ✅: `Correct. These three parameters are required to connect to a workspace.`
- [ ] **subscription_id, workspace_name, tenant_id** ❌: `Incorrect. Resource group is required instead of tenant_id.`
- [ ] **resource_group, workspace_name, region** ❌: `Incorrect. Subscription_id is required instead of region.`

## Code Examples

### Creating a Workspace with Python SDK

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace, CustomerManagedKey
from azure.identity import DefaultAzureCredential

# Create workspace with customer-managed key
ws = Workspace(
    name="my-workspace",
    location="eastus",
    display_name="My ML Workspace",
    description="Workspace for DP-100 preparation",
    customer_managed_key=CustomerManagedKey(
        key_vault="/subscriptions/<sub-id>/resourcegroups/<rg>/providers/microsoft.keyvault/vaults/<vault>",
        key_uri="<key-identifier-uri>"
    ),
    tags={"purpose": "DP-100", "environment": "development"}
)

# Create the workspace
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="<subscription-id>",
    resource_group_name="<resource-group>",
    workspace_name="<workspace-name>"
)

ml_client.workspaces.begin_create(ws)
```

### Creating a Compute Cluster

```python
from azure.ai.ml.entities import AmlCompute

compute_cluster = AmlCompute(
    name="cpu-cluster",
    size="Standard_DS3_v2",
    min_instances=0,
    max_instances=4,
    tier="Dedicated",  # or "LowPriority" for cost savings
    idle_time_before_scale_down=120  # seconds
)

ml_client.compute.begin_create_or_update(compute_cluster)
```

## Best Practices

1. **Security**:
   - Use customer-managed keys for sensitive workloads
   - Implement least-privilege access with appropriate roles
   - Enable private endpoints for network isolation

2. **Cost Optimization**:
   - Use Low Priority compute for non-critical workloads
   - Set up auto-shutdown schedules for compute instances
   - Configure appropriate idle time for compute clusters

3. **Organization**:
   - Use consistent naming conventions
   - Apply tags for cost tracking and governance
   - Create separate workspaces for different environments (dev/test/prod)

4. **Automation**:
   - Use ARM templates or CLI scripts for reproducible deployments
   - Integrate workspace creation into CI/CD pipelines
   - Version control your infrastructure code

<!-- START BADGE -->
<div align="center">
  <img src="https://img.shields.io/badge/Total%20views-1290-limegreen" alt="Total views">
  <p>Refresh Date: 2025-08-04</p>
</div>
<!-- END BADGE -->

