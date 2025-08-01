When a workspace is provisioned, Azure automatically creates other Azure resources within the same resource group to support the workspace:

Azure Storage Account: To store files and notebooks used in the workspace, and to store metadata of jobs and models.
Azure Key Vault: To securely manage secrets such as authentication keys and credentials used by the workspace.
Application Insights: To monitor predictive services in the workspace.
Azure Container Registry: Created when needed to store images for Azure Machine Learning environments.

Create the workspace
You can create an Azure Machine Learning workspace in any of the following ways:

Use the user interface in the Azure portal to create an Azure Machine Learning service.
Create an Azure Resource Manager (ARM) template. Learn how to use an ARM template to create a workspace.
Use the Azure Command Line Interface (CLI) with the Azure Machine Learning CLI extension. Learn how to create the workspace with the CLI v2.
Use the Azure Machine Learning Python SDK.

The supported paths you can use when creating a URI file data asset are:

There are three general built-in roles that you can use across resources and resource groups to assign permissions to other users:

Owner: Gets full access to all resources, and can grant access to others using access control.
Contributor: Gets full access to all resources, but can't grant access to others.
Reader: Can only view the resource, but isn't allowed to make any changes.
Additionally, Azure Machine Learning has specific built-in roles you can use:

AzureML Data Scientist: Can perform all actions within the workspace, except for creating or deleting compute resources, or editing the workspace settings.
AzureML Compute Operator: Is allowed to create, change, and manage access the compute resources within a workspace.


As a data scientist, you mostly work with assets in the Azure Machine Learning workspace. Assets are created and used at various stages of a project and include:

Models
Environments
Data
Components

To train models with the Azure Machine Learning workspace, you have several options:

Use Automated Machine Learning.
Run a Jupyter notebook.
Run a script as a job.

The menu shows what you can do in the studio:

Author: Create new jobs to train and track a machine learning model.
Assets: Create and review assets you use when training models.
Manage: Create and manage resources you need to train models.


Connect to the workspace
After the Python SDK is installed, you'll need to connect to the workspace. By connecting, you're authenticating your environment to interact with the workspace to create and manage assets and resources.

To authenticate, you need the values to three necessary parameters:

subscription_id: Your subscription ID.
resource_group: The name of your resource group.
workspace_name: The name of your workspace.


There are many advantages to using the Azure CLI with Azure Machine Learning. The Azure CLI allows you to:

Automate the creation and configuration of assets and resources to make it repeatable.
Ensure consistency for assets and resources that must be replicated in multiple environments (for example, development, test, and production).
Incorporate machine learning asset configuration into developer operations (DevOps) workflows, such as continuous integration and continuous deployment (CI/CD) pipelines.
To interact with the Azure Machine Learning workspace using the Azure CLI, you'll need to install the Azure CLI and the Azure Machine Learning extension.


There are different types of jobs depending on how you want to execute a workload:

Command: Execute a single script.
Sweep: Perform hyperparameter tuning when executing a single script.
Pipeline: Run a pipeline consisting of multiple scripts or components.



Local: ./<path>
Azure Blob Storage: wasbs://<account_name>.blob.core.windows.net/<container_name>/<folder>/<file>
Azure Data Lake Storage (Gen 2): abfss://<file_system>@<account_name>.dfs.core.windows.net/<folder>/<file>
Datastore: azureml://datastores/<datastore_name>/paths/<folder>/<file>


Compute instance: Behaves similarly to a virtual machine and is primarily used to run notebooks. It's ideal for experimentation.
Compute clusters: Multi-node clusters of virtual machines that automatically scale up or down to meet demand. A cost-effective way to run scripts that need to process large volumes of data. Clusters also allow you to use parallel processing to distribute the workload and reduce the time it takes to run a script.
Kubernetes clusters: Cluster based on Kubernetes technology, giving you more control over how the compute is configured and managed. You can attach your self-managed Azure Kubernetes (AKS) cluster for cloud compute, or an Arc Kubernetes cluster for on-premises workloads.
Attached compute: Allows you to attach existing compute like Azure virtual machines or Azure Databricks clusters to your workspace.
Serverless compute: A fully managed, on-demand compute you can use for training jobs.
 
 When you create a compute cluster, there are three main parameters you need to consider:

size: Specifies the virtual machine type of each node within the compute cluster. Based on the sizes for virtual machines in Azure. Next to size, you can also specify whether you want to use CPUs or GPUs.
max_instances: Specifies the maximum number of nodes your compute cluster can scale out to. The number of parallel workloads your compute cluster can handle is analogous to the number of nodes your cluster can scale to.
tier: Specifies whether your virtual machines are low priority or dedicated. Setting to low priority can lower costs as you're not guaranteed availability.


A conda specification file is a YAML file, which lists the packages that need to be installed using conda or pip. Such a YAML file may look like:

When you create an Azure Machine Learning workspace, curated environments are automatically created and made available to you. Alternatively, you can create and manage your own custom environments and register them in the workspace. Creating and registering custom environments makes it possible to define consistent, reusable runtime contexts for your experiments - regardless of where the experiment script is run.

After you've collected the data, you need to create a data asset in Azure Machine Learning. In order for AutoML to understand how to read the data, you need to create a MLTable data asset that includes the schema of the data.


There are several options to set limits to an AutoML experiment:

timeout_minutes: Number of minutes after which the complete AutoML experiment is terminated.
trial_timeout_minutes: Maximum number of minutes one trial can take.
max_trials: Maximum number of trials, or models that will be trained.
enable_early_termination: Whether to end the experiment if the score isn't improving in the short term.

Explore preprocessing steps
When you've enabled featurization for your AutoML experiment, data guardrails will automatically be applied too. The three data guardrails that are supported for classification models are:

Class balancing detection.
Missing feature values imputation.
High cardinality feature detection.
Each of these data guardrails will show one of three possible states:

Passed: No problems were detected and no action is required.
Done: Changes were applied to your data. You should review the changes AutoML has made to your data.
Alerted: An issue was detected but couldn't be fixed. You should review the data to fix the issue.

Common functions used with custom logging are:

mlflow.log_param(): Logs a single key-value parameter. Use this function for an input parameter you want to log.
mlflow.log_metric(): Logs a single key-value metric. Value must be a number. Use this function for any output you want to store with the run.
mlflow.log_artifact(): Logs a file. Use this function for any plot you want to log, save as image file first.
mlflow.log_model(): Logs a model. Use this function to create an MLflow model, which may include a custom signature, environment, and input examples.


Configure and submit a command job
To run a script as a command job, you'll need to configure and submit the job.

To configure a command job with the Python SDK (v2), you'll use the command function. To run a script, you'll need to specify values for the following parameters:

code: The folder that includes the script to run.
command: Specifies which file to run.
environment: The necessary packages to be installed on the compute before running the command.
compute: The compute to use to run the command.
display_name: The name of the individual job.
experiment_name: The name of the experiment the job belongs to.

Discrete hyperparameters
Some hyperparameters require discrete values - in other words, you must select the value from a particular finite set of possibilities. You can define a search space for a discrete parameter using a Choice from a list of explicit values, which you can define as a Python list (Choice(values=[10,20,30])), a range (Choice(values=range(1,10))), or an arbitrary set of comma-separated values (Choice(values=(30,50,100)))

You can also select discrete values from any of the following discrete distributions:

QUniform(min_value, max_value, q): Returns a value like round(Uniform(min_value, max_value) / q) * q
QLogUniform(min_value, max_value, q): Returns a value like round(exp(Uniform(min_value, max_value)) / q) * q
QNormal(mu, sigma, q): Returns a value like round(Normal(mu, sigma) / q) * q
QLogNormal(mu, sigma, q): Returns a value like round(exp(Normal(mu, sigma)) / q) * q


Continuous hyperparameters
Some hyperparameters are continuous - in other words you can use any value along a scale, resulting in an infinite number of possibilities. To define a search space for these kinds of value, you can use any of the following distribution types:

Uniform(min_value, max_value): Returns a value uniformly distributed between min_value and max_value
LogUniform(min_value, max_value): Returns a value drawn according to exp(Uniform(min_value, max_value)) so that the logarithm of the return value is uniformly distributed
Normal(mu, sigma): Returns a real value that's normally distributed with mean mu and standard deviation sigma
LogNormal(mu, sigma): Returns a value drawn according to exp(Normal(mu, sigma)) so that the logarithm of the return value is normally distributed

There are three main sampling methods available in Azure Machine Learning:

Grid sampling: Tries every possible combination.
Random sampling: Randomly chooses values from the search space.
Sobol: Adds a seed to random sampling to make the results reproducible.
Bayesian sampling: Chooses new values based on previous results.



Configure an early termination policy
There are two main parameters when you choose to use an early termination policy:

evaluation_interval: Specifies at which interval you want the policy to be evaluated. Every time the primary metric is logged for a trial counts as an interval.
delay_evaluation: Specifies when to start evaluating the policy. This parameter allows for at least a minimum of trials to complete without an early termination policy affecting them.
New models may continue to perform only slightly better than previous models. To determine the extent to which a model should perform better than previous trials, there are three options for early termination:

Bandit policy: Uses a slack_factor (relative) or slack_amount(absolute). Any new model must perform within the slack range of the best performing model.
Median stopping policy: Uses the median of the averages of the primary metric. Any new model must perform better than the median.
Truncation selection policy: Uses a truncation_percentage, which is the percentage of lowest performing trials. Any new model must perform better than the lowest performing trials.


There are two main reasons why you'd use components:

To build a pipeline.
To share ready-to-go code.


A component consists of three parts:

Metadata: Includes the component's name, version, etc.
Interface: Includes the expected input parameters (like a dataset or hyperparameter) and expected output (like metrics and artifacts).
Command, code and environment: Specifies how to run the code.
To create a component, you need two files:

A script that contains the workflow you want to execute.
A YAML file to define the metadata, interface, and command, code, and environment of the component.

There are various ways to create a schedule. A simple approach is to create a time-based schedule using the RecurrenceTrigger class with the following parameters:

frequency: Unit of time to describe how often the schedule fires. Value can be either minute, hour, day, week, or month.
interval: Number of frequency units to describe how often the schedule fires. Value needs to be an integer.


MLflow allows you to log a model as an artifact, or as a model. When you log a model as an artifact, the model is treated as a file. When you log a model as a model, you're adding information to the registered model that enables you to use the model directly in pipelines or deployments. Learn more about the difference between an artifact and a model

The MLmodel file may include:

artifact_path: During the training job, the model is logged to this path.
flavor: The machine learning library with which the model was created.
model_uuid: The unique identifier of the registered model.
run_id: The unique identifier of job run during which the model was created.
signature: Specifies the schema of the model's inputs and outputs:
inputs: Valid input to the model. For example, a subset of the training dataset.
outputs: Valid model output. For example, model predictions for the input dataset.

The MLmodel file may include:

artifact_path: During the training job, the model is logged to this path.
flavor: The machine learning library with which the model was created.
model_uuid: The unique identifier of the registered model.
run_id: The unique identifier of job run during which the model was created.
signature: Specifies the schema of the model's inputs and outputs:
inputs: Valid input to the model. For example, a subset of the training dataset.
outputs: Valid model output. For example, model predictions for the input dataset.


A flavor is the machine learning library with which the model was created.

There are two types of signatures:

Column-based: used for tabular data with a pandas.Dataframe as inputs.
Tensor-based: used for n-dimensional arrays or tensors (often used for unstructured data like text or images), with numpy.ndarray as inputs.


There are three types of models you can register:

MLflow: Model trained and tracked with MLflow. Recommended for standard use cases.
Custom: Model type with a custom standard not currently supported by Azure Machine Learning.
Triton: Model type for deep learning workloads. Commonly used for TensorFlow and PyTorch model deployments.


Microsoft has listed five Responsible AI principles:

Fairness and inclusiveness: Models should treat everyone fairly and avoid different treatment for similar groups.
Reliability and safety: Models should be reliable, safe, and consistent. You want a model to operate as intended, handle unexpected situations well, and resist harmful manipulation.
Privacy and security: Be transparent about data collection, use, and storage, to empower individuals with control over their data. Treat data with care to ensure an individual's privacy.
Transparency: When models influence important decisions that affect people's lives, people need to understand how those decisions were made and how the model works.
Accountability: Take accountability for decisions that models may influence and maintain human control.


Create a Responsible AI dashboard
To create a Responsible AI (RAI) dashboard, you need to create a pipeline by using the built-in components. The pipeline should:

Start with the RAI Insights dashboard constructor.
Include one of the RAI tool components.
End with Gather RAI Insights dashboard to collect all insights into one dashboard.
Optionally you can also add the Gather RAI Insights score card at the end of your pipeline.
Explore the Responsible AI components
The available tool components and the insights you can use are:

Add Explanation to RAI Insights dashboard: Interpret models by generating explanations. Explanations show how much features influence the prediction.
Add Causal to RAI Insights dashboard: Use historical data to view the causal effects of features on outcomes.
Add Counterfactuals to RAI Insights dashboard: Explore how a change in input would change the model's output.
Add Error Analysis to RAI Insights dashboard: Explore the distribution of your data and identify erroneous subgroups of data.

After you've trained and registered a model in the Azure Machine Learning workspace, you can create the Responsible AI dashboard in three ways:

Using the Command Line Interface (CLI) extension for Azure Machine Learning.
Using the Python Software Development Kit (SDK).
Using the Azure Machine Learning studio for a no-code experience.

Register the training and test datasets as MLtable data assets.
Register the model.
Retrieve the built-in components you want to use.
Build the pipeline.
Run the pipeline.


The output of each component you added to the pipeline is reflected in the dashboard. Depending on the components you selected, you can find the following insights in your Responsible AI dashboard:

Error analysis
Explanations
Counterfactuals
Causal analysis


When you include error analysis, there are two types of visuals you can explore in the Responsible AI dashboard:

Error tree map: Allows you to explore which combination of subgroups results in the model making more false predictions.
Error heat map: Presents a grid overview of a model's errors over the scale of one or two features.


There are various statistical techniques you can use as model explainers. Most commonly, the mimic explainer trains a simple interpretable model on the same data and task. As a result, you can explore two types of feature importance:

Aggregate feature importance: Shows how each feature in the test data influences the model's predictions overall.


Individual feature importance: Shows how each feature impacts an individual prediction.


Managed online endpoint
Within Azure Machine Learning, there are two types of online endpoints:

Managed online endpoints: Azure Machine Learning manages all the underlying infrastructure.
Kubernetes online endpoints: Users manage the Kubernetes cluster which provides the necessary infrastructure.

After you create an endpoint in the Azure Machine Learning workspace, you can deploy a model to that endpoint. To deploy your model to a managed online endpoint, you need to specify four things:

Model assets like the model pickle file, or a registered model in the Azure Machine Learning workspace.
Scoring script that loads the model.
Environment which lists all necessary packages that need to be installed on the compute of the endpoint.
Compute configuration including the needed compute size and scale settings to ensure you can handle the amount of requests the endpoint will receive.


To create an online endpoint, you'll use the ManagedOnlineEndpoint class, which requires the following parameters:

name: Name of the endpoint. Must be unique in the Azure region.
auth_mode: Use key for key-based authentication. Use aml_token for Azure Machine Learning token-based authentication.

Let's take the example of the restaurant recommender model. After experimentation, you select the best performing model. You use the blue deployment for this first version of the model. As new data is collected, the model can be retrained, and a new version is registered in the Azure Machine Learning workspace. To test the new model, you can use the green deployment for the second version of the model.

Next to the model, you also need to specify the compute configuration for the deployment:

instance_type: Virtual machine (VM) size to use. Review the list of supported sizes.
instance_count: Number of instances to use.


Deploy a model to an endpoint
To deploy a model, you must have:

Model files stored on local path or registered model.
A scoring script.
An execution environment.
The model files can be logged and stored when you train a model.

Create the scoring script
The scoring script needs to include two functions:

init(): Called when the service is initialized.
run(): Called when new data is submitted to the service.


Create the deployment
When you have your model files, scoring script, and environment, you can create the deployment.

To deploy a model to an endpoint, you can specify the compute configuration with two parameters:

instance_type: Virtual machine (VM) size to use. Review the list of supported sizes.
instance_count: Number of instances to use.

Deploy an MLflow model to an endpoint
To deploy an MLflow model to a batch endpoint, you'll use the BatchDeployment class.

When you deploy a model, you'll need to specify how you want the batch scoring job to behave. The advantage of using a compute cluster to run the scoring script (which is automatically generated by Azure Machine Learning), is that you can run the scoring script on separate instances in parallel.

When you configure the model deployment, you can specify:

instance_count: Count of compute nodes to use for generating predictions.
max_concurrency_per_instance: Maximum number of parallel scoring script runs per compute node.
mini_batch_size: Number of files passed per scoring script run.
output_action: What to do with the predictions: summary_only or append_row.
output_file_name: File to which predictions will be appended, if you choose append_row for output_action.

Create the scoring script
The scoring script is a file that reads the new data, loads the model, and performs the scoring.

The scoring script must include two functions:

init(): Called once at the beginning of the process, so use for any costly or common preparation like loading the model.
run(): Called for each mini batch to perform the scoring.
The run() method should return a pandas DataFrame or an array/list.


There are some things to note from the example script:

AZUREML_MODEL_DIR is an environment variable that you can use to locate the files associated with the model.
Use global variable to make any assets available that are needed to score the new data, like the loaded model.
The size of the mini_batch is defined in the deployment configuration. If the files in the mini batch are too large to be processed, you need to split the files into smaller files.
By default, the predictions will be written to one single file.


If you want to troubleshoot the scoring script, you can select the child job and review its outputs and logs.

Navigate to the Outputs + logs tab. The logs/user/ folder contains three files that will help you troubleshoot:

job_error.txt: Summarize the errors in your script.
job_progress_overview.txt: Provides high-level information about the number of mini-batches processed so far.
job_result.txt: Shows errors in calling the init() and run() function in the scoring script.


lets make a table

The Azure AI Services resource type includes the following services, making them available from a single endpoint:
Azure AI Speech
Azure AI Language
Azure AI Translator
Azure AI Vision
Azure AI Face
Azure AI Custom Vision
Azure AI Document Intelligence

The Azure AI Foundry resource type includes the following services, and supports working with them through an Azure AI Foundry project*:
Azure OpenAI
Azure AI Speech
Azure AI Language
Azure AI Foundry Content Safety
Azure AI Translator
Azure AI Vision
Azure AI Face
Azure AI Document Intelligence
Azure AI Content Understanding


- Foundry projects are associated with an Azure AI Foundry resource in an Azure subscription. Foundry projects provide support for Azure AI Foundry models (including OpenAI models), Azure AI Foundry Agent Service, Azure AI services, and tools for evaluation and responsible AI development.

An Azure AI Foundry resource supports the most common AI development tasks to develop generative AI chat apps and agents. In most cases, using a Foundry project provides the right level of resource centralization and capabilities with a minimal amount of administrative resource management. You can use Azure AI Foundry portal to work in projects that are based in Azure AI Foundry resources, making it easy to add connected resources and manage model and agent deployments.

Hub-based projects are associated with an Azure AI hub resource in an Azure subscription. Hub-based projects include an Azure AI Foundry resource, as well as managed compute, support for Prompt Flow development, and connected Azure storage and Azure key vault resources for secure data storage.

Azure AI hub resources support advanced AI development scenarios, like developing Prompt Flow based applications or fine-tuning models. You can also use Azure AI hub resources in both Azure AI Foundry portal and Azure Machine learning portal, making it easier to work on collaborative projects that involve data scientists and machine learning specialists as well as developers and AI software engineers

lets make a table 


Programming languages, APIs, and SDKs
You can develop AI applications using many common programming languages and frameworks, including Microsoft C#, Python, Node, TypeScript, Java, and others. When building AI solutions on Azure, some common SDKs you should plan to install and use include:

The Azure AI Foundry SDK, which enables you to write code to connect to Azure AI Foundry projects and access resource connections, which you can then work with using service-specific SDKs.
The Azure AI Foundry Models API, which provides an interface for working with generative AI model endpoints hosted in Azure AI Foundry.
The Azure OpenAI in Azure AI Foundry Models API, which enables you to build chat applications based on OpenAI models hosted in Azure AI Foundry.
Azure AI Services SDKs - AI service-specific libraries for multiple programming languages and frameworks that enable you to consume Azure AI Services resources in your subscription. You can also use Azure AI Services through their REST APIs.
The Azure AI Foundry Agent Service, which is accessed through the Azure AI Foundry SDK and can be integrated with frameworks like Semantic Kernel to build comprehensive AI agent solutions.

The key metrics used for monitoring evaluation in prompt flow each offer unique insight into the performance of LLMs:

Groundedness: Measures alignment of the LLM application's output with the input source or database.
Relevance: Assesses how pertinent the LLM application's output is to the given input.
Coherence: Evaluates the logical flow and readability of the LLM application's text.
Fluency: Assesses the grammatical and linguistic accuracy of the LLM application's output.
Similarity: Quantifies the contextual and semantic match between the LLM application's output and the ground truth.


The project provides multiple endpoints and keys, including:

An endpoint for the project itself; which can be used to access project connections, agents, and models in the Azure AI Foundry resource.
An endpoint for Azure OpenAI Service APIs in the project's Azure AI Foundry resource.
An endpoint for Azure AI services APIs (such as Azure AI Vision and Azure AI Language) in the Azure AI Foundry resource.

The core package for working with projects is the Azure AI Projects library, which enables you to connect to an Azure AI Foundry project and access the resources defined within it. Available language-specific packages the for Azure AI Projects library include:

Azure AI Projects for Python
Azure AI Projects for Microsoft .NET
Azure AI Projects for JavaScript


For example, the AIProjectClient object in Python has a connections property, which you can use to access the resource connections in the project. Methods of the connections object include:

connections.list(): Returns a collection of connection objects, each representing a connection in the project. You can filter the results by specifying an optional connection_type parameter with a valid enumeration, such as ConnectionType.AZURE_OPEN_AI.
connections.get(connection_name, include_credentials): Returns a connection object for the connection with the name specified. If the include_credentials parameter is True (the default value), the credentials required to connect to the connection are returned - for example, in the form of an API key for an Azure AI services resource.

Adding grounding data to an Azure AI project
You can use Azure AI Foundry to build a custom age that uses your own data to ground prompts. Azure AI Foundry supports a range of data connections that you can use to add data to a project, including:

Azure Blob Storage
Azure Data Lake Storage Gen2
Microsoft OneLake


Searching an index
There are several ways that information can be queried in an index:

Keyword search: Identifies relevant documents or passages based on specific keywords or terms provided as input.
Semantic search: Retrieves documents or passages by understanding the meaning of the query and matching it with semantically related content rather than relying solely on exact keyword matches.
Vector search: Uses mathematical representations of text (vectors) to find similar documents or passages based on their semantic meaning or context.
Hybrid search: Combines any or all of the other search techniques. Queries are executed in parallel and are returned in a unified result set.

The Microsoft guidance for responsible generative AI is designed to be practical and actionable. It defines a four stage process to develop and implement a plan for responsible AI when using generative models. The four stages in the process are:

Map potential harms that are relevant to your planned solution.
Measure the presence of these harms in the outputs generated by your solution.
Mitigate the harms at multiple layers in your solution to minimize their presence and impact, and ensure transparent communication about potential risks to users.
Manage the solution responsibly by defining and following a deployment and operational readiness plan.

Red teaming is a strategy that is often used to find security vulnerabilities or other weaknesses that can compromise the integrity of a software solution. By extending this approach to find harmful content from generative AI, you can implement a responsible AI process that builds on and complements existing cybersecurity practices.

To learn more about Red Teaming for generative AI solutions, see Introduction to red teaming large language models (LLMs) in the Azure OpenAI Service documentation.

Model benchmarks
Model benchmarks are publicly available metrics across models and datasets. These benchmarks help you understand how your model performs relative to others. Some commonly used benchmarks include:

Accuracy: Compares model generated text with correct answer according to the dataset. Result is one if generated text matches the answer exactly, and zero otherwise.
Coherence: Measures whether the model output flows smoothly, reads naturally, and resembles human-like language
Fluency: Assesses how well the generated text adheres to grammatical rules, syntactic structures, and appropriate usage of vocabulary, resulting in linguistically correct and natural-sounding responses.
GPT similarity: Quantifies the semantic similarity between a ground truth sentence (or document) and the prediction sentence generated by an AI model.


AI-assisted metrics
AI-assisted metrics use advanced techniques to evaluate model performance. These metrics can include:

Generation quality metrics: These metrics evaluate the overall quality of the generated text, considering factors like creativity, coherence, and adherence to the desired style or tone.

Risk and safety metrics: These metrics assess the potential risks and safety concerns associated with the model's outputs. They help ensure that the model doesn't generate harmful or biased content.'''Natural language processing metrics
Natural language processing (NLP) metrics are also valuable in evaluating model performance. One such metric is the F1-score, which measures the ratio of the number of shared words between the generated and ground truth answers. The F1-score is useful for tasks like text classification and information retrieval, where precision and recall are important. Other common NLP metrics include:

BLEU: Bilingual Evaluation Understudy metric
METEOR: Metric for Evaluation of Translation with Explicit Ordering
ROUGE: Recall-Oriented Understudy for Gisting Evaluation

Evaluation metrics
Automated evaluation enables you to choose which evaluators you want to assess your model's responses, and which metrics those evaluators should calculate. There are evaluators that help you measure:

AI Quality: The quality of your model's responses is measured by using AI models to evaluate them for metrics like coherence and relevance and by using standard NLP metrics like F1 score, BLEU, METEOR, and ROUGE based on ground truth (in the form of expected response text)
Risk and safety: evaluators that assess the responses for content safety issues, including violence, hate, sexual content, and content related to self-harm.


Prompt engineering is a quick and easy way to improve how the model acts, and what the model needs to know. When you want to improve the quality of the model even further, there are two common techniques that are used:

Retrieval Augmented Generation (RAG): Ground your data by first retrieving context from a data source before generating a response.
Fine-tuning: Train a base language model on a dataset before integrating it in your application.


When you fine-tune a language model for chat completion, the data you use to fine-tune a model is a collection of sample conversations. More specifically, the data should contain three components:

The system message
The user message
The assistant's response


The dataset should show the model's ideal behavior. You can create this dataset based on the chat history of a chat application you have. A few things to keep in mind when you use real data is to:

Remove any personal or sensitive information.
Not only focus on creating a large training dataset, but also ensure your dataset includes a diverse set of example

Some considerations you can take into account when deciding on a foundation model before fine-tuning are:

Model capabilities: Evaluate the capabilities of the foundation model and how well they align with your task. For example, a model like BERT is better at understanding short texts.
Pretraining data: Consider the dataset used for pretraining the foundation model. For example, GPT-2 is trained on unfiltered content from the internet that can result in biases.
Limitations and biases: Be aware of any limitations or biases that might be present in the foundation model.
Language support: Explore which models offer the specific language support or multilingual capabilities that you need for your use case.


Name	Description
batch_size	The batch size to use for training. The batch size is the number of training examples used to train a single forward and backward pass. In general, larger batch sizes tend to work better for larger datasets. The default value and the maximum value for this property are specific to a base model. A larger batch size means that model parameters are updated less frequently, but with lower variance.
learning_rate_multiplier	The learning rate multiplier to use for training. The fine-tuning learning rate is the original learning rate used for pretraining multiplied by this value. Larger learning rates tend to perform better with larger batch sizes. We recommend experimenting with values in the range 0.02 to 0.2 to see what produces the best results. A smaller learning rate can be useful to avoid overfitting.
n_epochs	The number of epochs to train the model for. An epoch refers to one full cycle through the training dataset.
seed	The seed controls the reproducibility of the job. Passing in the same seed and job parameters should produce the same results, but can differ in rare cases. If a seed isn't specified, one is generated for you.


make me a table of AI metrics 


Differential Privacy (DP) introduces noise to data or query results to protect individual privacy. The parameter ε (epsilon) controls the privacy-accuracy trade-off.


What does epsilon mean?
Low ε (e.g., 0.1) → Stronger privacy, but more noise → lower accuracy.
High ε (e.g., 10) → Weaker privacy, but less noise → higher accuracy.


In data contexts, high cardinality means that a column (or feature) contains a large number of unique values relative to the total number of rows.



| **Method**        | **What it means** | **Parameter Type** |
|--------------------|--------------------|----------------------|
| **Grid Sampling** | Tries every combination of values in a grid. Good coverage but slow for many parameters. | Discrete |
| **Random Sampling** | Picks values at random. Faster, but results can be uneven. | Discrete or Continuous |
| **Sobol Sampling** | Picks values in a smart way so they cover the space more evenly than random. Good for large search spaces. | Continuous |
| **Bayesian Sampling** | Learns from past runs to pick better values next time. Great when runs are expensive. | Continuous (can handle some discrete) |



| **Option**                                | **When to use it** |
|-------------------------------------------|----------------------|
| **Analyze the selection rate of different cohorts** | Use when you want to check if different groups (e.g., age, gender) are being selected at similar rates. Example: Hiring model fairness. |
| **Analyze the recall of different cohorts** | Use when you want to ensure the model performs equally well for all groups in terms of correctly identifying positive cases. Example: Anomaly detection in purchase orders where age is a feature. |
| **Analyze the global influencers** | Use when you need to understand which features have the most impact on predictions overall. Example: Explaining why a credit risk model makes certain predictions. |
| **Enable Application Insights** | Use when you want to monitor and log model performance and usage in production. Example: Tracking latency and errors in a deployed model. |


| **Command**                                | **What it’s for** |
|-------------------------------------------|----------------------|
| **az ml environment create**             | Creates a new environment in Azure ML using the CLI. |
| **ml_client.environments.create_or_update** | Creates or updates an environment programmatically using the Python SDK (v2). |
| **az ml environment show**               | Displays details of an existing environment in Azure ML. |
| **az ml environment update**             | Updates an existing environment in Azure ML using the CLI. |


| **Option**              | **What it’s for** |
|--------------------------|----------------------|
| **environment**         | Represents the runtime environment for training or inference (e.g., Python packages, Docker image). |
| **command**             | Used to define and run a command job in Azure ML (e.g., training script execution). |
| **MLClient**            | The main Python SDK v2 client for interacting with Azure ML resources (workspaces, jobs, models, etc.). |
| **DefaultAzureCredential** | Handles authentication by automatically picking the best available credential (e.g., managed identity, CLI login). |


| **Option**                                              | **What it’s for** |
|---------------------------------------------------------|----------------------|
| **Move memory-heavy data processing components to high-memory CPU machines** | Use when your pipeline has steps that require a lot of RAM (e.g., large dataset preprocessing). |
| **Select less expensive GPU machines**                 | Use when you need GPUs but want to reduce cost by choosing smaller or cheaper GPU SKUs. |
| **Move all components to CPU machines**                | Use when your workload does not require GPU acceleration (e.g., simple models or data prep). |
| **Move all components to memory optimized machines**    | Use when the entire pipeline is memory-intensive and benefits from high-memory nodes. |


| **Option**                  | **What it’s for** |
|-----------------------------|----------------------|
| **Input**                  | Defines input data for a job or pipeline step (e.g., datasets, files, or folders). |
| **Output**                 | Defines output data for a job or pipeline step (e.g., where to store results or models). |
| **AmlTokenConfiguration**  | Handles authentication for jobs using Azure ML tokens (used for secure access to resources). |
| **MLClient**               | The main Python SDK v2 client for managing Azure ML resources (workspaces, jobs, models, environments). |


| **Option**                                      | **What it’s for** |
|-------------------------------------------------|----------------------|
| **Check the pipeline job link in the submission list** | Use this to view details and status of a submitted pipeline job. |
| **Monitor the status in the authoring page**    | Use this to track the progress of your pipeline directly in the authoring interface. |
| **Publish the pipeline as an endpoint**         | Use this when you want to make the pipeline reusable and callable via REST API for automation or integration. |
| **Refresh the authoring page**                  | Use this to update the page and see the latest status or changes in your pipeline. |


| **Option**              | **What it’s for** |
|--------------------------|----------------------|
| **azure.ai.ml.dsl**     | Provides decorators and functions for defining Azure ML pipelines using the Python SDK v2 (e.g., `@pipeline`). |
| **azure.ai**            | A general namespace for Azure AI services, not specific to Azure ML pipelines. |
| **azure.ai.ml**         | The main Azure ML SDK v2 package for managing resources like jobs, models, environments, and components. |



Here’s how we can turn your example into a **comprehensive wiki-style overview** for Azure ML pipelines (DP-100 context). I’ll structure it like a knowledge base entry:

---

# **Azure ML Pipelines: Passing Data Between Steps**

## **Overview**
Azure Machine Learning pipelines allow you to chain multiple steps together, where the output of one step becomes the input of another. This is essential for modular workflows like **data preparation → training → evaluation**.

---

## **Key Concepts**
- **`OutputFileDatasetConfig`**  
  Defines an output location for a step. Azure ML will store the data in the default datastore and make it available for downstream steps.

- **`PythonScriptStep`**  
  Represents a pipeline step that runs a Python script. You can pass inputs and outputs as arguments.

- **Data Flow**  
  - Step 1 writes data to an **OutputFileDatasetConfig**.
  - Step 2 consumes that output using `.as_input()`.

---

## **Example Workflow**
```python
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep

# Define output for prepared data
prepped_data = OutputFileDatasetConfig(name='prepped')

# Step 1: Data Preparation
step1 = PythonScriptStep(
    name="Prepare Data",
    script_name='data_prep.py',
    arguments=['--out_folder', prepped_data],
    outputs=[prepped_data]
)

# Step 2: Model Training
step2 = PythonScriptStep(
    name="Train Model",
    script_name='train.py',
    arguments=['--training-data', prepped_data.as_input()]
)
```

---

## **How It Works**
- **Step 1** writes processed data to the path provided by `prepped_data`.
- Azure ML automatically tracks this output and stores it in the workspace.
- **Step 2** uses `prepped_data.as_input()` to mount or download the same data for training.

---

## **Why Use This Approach?**
- **Reproducibility**: Outputs are versioned and stored in the workspace.
- **Automation**: No need to hardcode paths; Azure ML manages them.
- **Lineage Tracking**: You can trace which data was used for which run.

---

## **Best Practices**
- Use **OutputFileDatasetConfig** for intermediate data that needs to be reused.
- Always declare outputs in the `outputs` parameter of `PythonScriptStep`.
- For large data, prefer **mounting** over **downloading** for efficiency.

---

✅ Do you want me to **expand this into a full DP-100 Wiki Page** that includes:
- **Pipeline basics**
- **Data passing methods (OutputFileDatasetConfig vs Dataset vs v2 DSL)**
- **Comparison of v1 vs v2 SDK**
- **Common exam scenarios and code snippets**?

I can prepare it as a **structured Markdown document** for your study notes. Should I go ahead and create that?

Here’s a **general overview and formula for cron-based scheduling in Azure ML** (and similar systems):

---

## **General Cron Expression Format**
```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)  ← 24-hour format
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday=0)
│ │ │ │ │
* * * * *
```

### **Example**
```
15 13 * * 3
```
- **15** → Minute (15th minute)
- **13** → Hour (13:00 in 24-hour format → 1 PM)
- **\*** → Every day of the month
- **\*** → Every month
- **3** → Wednesday (0=Sunday, 3=Wednesday)

**Meaning:** Run at **1:15 PM every Wednesday**.

---

## **Azure ML CronTrigger Parameters**
- **expression**: The cron string (e.g., `"15 13 * * 3"`).
- **start_time**: Earliest time the schedule can start (usually `datetime.utcnow()`).
- **time_zone**: Time zone for interpreting the cron expression (e.g., `"Eastern Standard Time"`).

---

## **Full Example**
```python
from datetime import datetime
from azure.ai.ml.entities import CronTrigger

schedule_start_time = datetime.utcnow()

cron_trigger = CronTrigger(
    expression="15 13 * * 3",  # Every Wednesday at 1:15 PM
    start_time=schedule_start_time,
    time_zone="Eastern Standard Time"
)
```

---

### **Key Points**
- **Hours are always in 24-hour format**.
- **Time zone matters**: Azure ML converts your cron expression to UTC internally.
- **start_time** ensures the schedule won’t run before that moment.

---

✅ Do you want me to **expand this into a full “Azure ML Scheduling Overview” wiki section** that includes:
- **Cron vs Recurrence triggers**
- **Common cron patterns (daily, weekly, monthly)**
- **Best practices for DP-100 exam**
- **Full code examples for job scheduling**?

I can prepare it as a **structured Markdown cheat sheet** for your notes. Should I go ahead?


| **Option**                                                   | **When to use it** |
|-------------------------------------------------------------|----------------------|
| **Metadata of the component**                               | Use when you only need to describe the component itself (e.g., name, description, version) without specifying inputs or outputs. |
| **Metadata of the component and inputs**                    | Use when the component requires input data or parameters but does not produce outputs (e.g., a validation step). |
| **Metadata of the component, inputs, outputs, and compute environment** | Use when defining a full component for a pipeline step that needs inputs, produces outputs, and specifies the compute environment (most common scenario for reusable components). |
| **Metadata of the component, inputs, and compute environment** | Use when the component has inputs and requires a specific compute environment but does not produce outputs (e.g., a monitoring or logging step). |


| **Option**                     | **What it’s for** |
|--------------------------------|----------------------|
| **Azure Machine Learning designer** | A drag-and-drop interface for building machine learning pipelines without writing code. Great for quick prototyping and visual workflows. |
| **Responsible AI scorecard**        | Generates a report summarizing model fairness, interpretability, and performance metrics to support responsible AI practices. |
| **Batch endpoints**                  | Used to deploy models for batch inference (processing large datasets asynchronously). |
| **Azure Blob Datastore**             | A storage option in Azure ML for persisting datasets, models, and pipeline outputs using Azure Blob Storage. |

| **Option**                          | **What it’s for** |
|-------------------------------------|----------------------|
| **MLflow**                          | An open-source tool integrated with Azure ML for tracking experiments, logging metrics, and managing model versions. |
| **Azure Machine Learning Endpoint** | A deployment target in Azure ML for serving models as REST APIs (real-time or batch). |
| **Batch Endpoint**                  | A specific type of Azure ML endpoint for running batch inference on large datasets asynchronously. |
| **ONNX Runtime**                    | A high-performance inference engine for models in the ONNX format, often used to optimize and accelerate model scoring. |

| **Flag**                                         | **What it does**                                                                                                                                           | **When to use**                                                                                         |
|--------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `--traffic "version1=90,version2=10"`             | Splits incoming request traffic between deployed model versions. In this example, 90% goes to **version1**, 10% to **version2**.                         | Gradually shift traffic to a new model (canary release) while keeping most users on the stable version. |
| `--traffic "version1=10,version2=90"`             | Same as above but in reverse: 10% → **version1**, 90% → **version2**.                                                                                      | After validating version2’s performance, flip the majority of traffic over to it.                       |
| `--mirror-traffic`                               | Sends a copy of each incoming request to all deployed versions without affecting response routing. Useful for “shadow testing.”                           | Test a new model version in production silently; compare responses without impacting live traffic.     |
| `--name <ENDPOINT_NAME>`                         | Specifies the target endpoint’s name. Required when your CLI context has multiple endpoints or you’re scripting.                                          | Always include this in automation scripts to ensure you’re updating the correct endpoint.               |


| **Command**                       | **What it’s for**                                                                                   |
|-----------------------------------|-----------------------------------------------------------------------------------------------------|
| **az ml compute target create**   | Provision a new compute resource (VM, GPU cluster, or attached-AKS/HDInsight) via the Azure ML CLI. |
| **az ml run submit-script**       | Submit a local Python script or shell command as a training or preprocessing job on a compute target. |
| **az ml model register**          | Take your trained model files and register them in the workspace’s model registry for versioning.    |
| **az ml model deploy**            | Deploy a registered model to an online or batch endpoint, making it available for inference calls.   |


| **Step**                                              | **What it’s for** |
|-------------------------------------------------------|----------------------|
| **Preprocess the data by using a manual process**     | Use when you need full control over feature engineering, cleaning, or transformations (e.g., custom scripts for domain-specific logic). |
| **Preprocess the data by using an automated process** | Use when you want to speed up data prep using AutoML or built-in transformations (e.g., quick prototyping or large-scale automation). |
| **Compare the outputs of the new model to the outputs of the old model** | Use during model validation to ensure the new model performs better or at least meets baseline performance before deployment. |
| **Replace the old model based on a predefined replacement criterion** | Use in CI/CD or MLOps pipelines to automate deployment when the new model meets accuracy, fairness, or latency thresholds. |



| **Option**          | **What it’s for** |
|----------------------|----------------------|
| **traffic**          | Defines how incoming requests are split across multiple deployments (e.g., `blue=90,green=10`). Used for canary releases and gradual rollouts. |
| **kind**             | Specifies the type of endpoint (e.g., `Managed` or `Kubernetes`). Determines the underlying infrastructure for deployment. |
| **tags**             | Key-value metadata for organizing and categorizing endpoints (e.g., `env=prod`, `team=data-science`). Useful for governance and cost tracking. |
| **mirror_traffic**   | Sends a copy of live traffic to another deployment for shadow testing without affecting responses. Great for validating new models in production silently. |


| **Option**                  | **When to use it** |
|-----------------------------|----------------------|
| **Online Endpoint**         | Use when you need **real-time predictions** with low latency (e.g., fraud detection, chatbots). |
| **Batch Endpoint**          | Use when you need to process **large datasets asynchronously** (e.g., scoring millions of records overnight). |
| **Azure Kubernetes Service (AKS)** | Use when you need **high-scale, low-latency deployments** for production workloads or when you require advanced networking and autoscaling. |
| **conda**                   | Use when defining the **Python environment** for your training or inference jobs (e.g., specifying dependencies in `environment.yml`). |


| **Function**    | **Purpose**                                                  | **Azure ML Usage**                                                |
|-----------------|--------------------------------------------------------------|--------------------------------------------------------------------|
| **init**        | Load model files and perform any one-time setup.             | Called once when the container/pod starts up to initialize state. [1](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-batch-scoring-script?view=azureml-api-2) |
| **run**         | Accept input payload, run inference, and return results.     | Invoked for each request to produce predictions. [1](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-batch-scoring-script?view=azureml-api-2) |
| **main**        | Standard Python script entry point (`if __name__ == "__main__"`). | Used for standalone/local debugging—**not** called by Azure ML.     |
| **azureml_main**| Legacy entry function in older Azure ML service deployments. | Superseded by `run`; you’ll almost always implement `run()` today. |


| **Option**            | **What it’s for** |
|------------------------|----------------------|
| **BatchDeployment**    | Deploy a model for **batch inference**, processing large datasets asynchronously (e.g., scoring millions of records overnight). |
| **OnlineDeployment**   | Deploy a model for **real-time inference**, serving predictions with low latency via REST API. |
| **Parallel**           | A compute target type for **parallel processing** of large data jobs (e.g., distributed data prep or scoring). |
| **Pipeline**           | An Azure ML construct for chaining multiple steps (data prep, training, evaluation) into a reproducible workflow. |


| **Action**                                         | **When to use it** |
|----------------------------------------------------|----------------------|
| **Clone the GitHub/Azure DevOps Repository**       | Use when you need a local copy of the repo to work on code, pipelines, or configuration files. |
| **Raise a Pull Request (PR) to GitHub/Azure DevOps Repository** | Use when you want to merge changes from your feature branch into the main branch after review. |
| **Create a new branch in GitHub/Azure DevOps Repository** | Use when starting a new feature, bug fix, or experiment to keep changes isolated from the main branch. |
| **Stash the changes done in GitHub/Azure DevOps Repository** | Use when you need to temporarily save uncommitted changes without committing them, often before switching branches. |


| **Format**   | **When to use it** | **Recommended Max Size** |
|--------------|----------------------|----------------------------|
| **Python (Pickle)** | Use for serializing Python objects (e.g., trained models, dictionaries). Best for internal use, not for large tabular data. | ~100 MB (avoid very large pickles for portability and security). |
| **Parquet**  | Use for large tabular datasets; supports compression and is optimized for big data and distributed processing. | **Up to several GBs** (commonly 1–2 GB per file for best performance). |
| **CSV**      | Use for simple tabular data exchange; human-readable but inefficient for very large datasets. | ~100 MB to 1 GB (beyond that, performance drops significantly). |
| **XLSX**     | Use for small datasets or business reporting; not recommended for ML pipelines. | ~50 MB (Excel has practical limits on rows/columns). |


| **Method**                                | **What it’s for** |
|-------------------------------------------|----------------------|
| **data.to_pandas_dataframe()**           | Converts the Azure ML `Dataset` into a **Pandas DataFrame** for in-memory processing in Python. Best for small to medium datasets that fit in memory. |
| **data.get()**                            | Downloads the dataset to the local or compute node file system. Use when you need raw files for custom processing or non-Python tools. |
| **data.to_spark_dataframe()**            | Converts the Azure ML `Dataset` into a **Spark DataFrame** for distributed processing. Best for large datasets that require big data frameworks. |
| **data.sample()**                         | Returns a **random sample** of the dataset. Useful for quick exploration or testing without loading the entire dataset. |


| **Option**                                                                 | **When to use it** |
|----------------------------------------------------------------------------|----------------------|
| **Create and register an environment while running the first experiment. Then, specify the environment while replicating the experiment.** | Best practice for Azure ML. Use when you want reproducibility and minimal redundancy. Registered environments can be reused across experiments and pipelines. |
| **Save a Dockerfile for the first experiment. Share the Dockerfile with other data scientists.** | Use when you need full control over the environment (custom OS, libraries, system dependencies) and want portability outside Azure ML. |
| **Create and register an environment while running the first experiment. Then, create a duplicate environment while replicating the experiment.** | Avoid this unless absolutely necessary. It creates redundant environments and increases maintenance overhead. |
| **Save a requirements.txt file for the initial experiment. Then, use the file to create a new environment while replicating the experiment.** | Use when you want a lightweight way to share Python dependencies, but note that this does not guarantee full reproducibility (system packages and OS differences can still cause issues). |


| **Option**                                                                 | **When to use it** |
|----------------------------------------------------------------------------|----------------------|
| **Use Azure Machine Learning studio to stop the compute instance**         | Use for **manual control** when you want to stop the instance immediately through the UI. Best for ad-hoc cost control. |
| **Create a script that uses the Azure Machine Learning SDK for Python. Schedule the script to run.** | Use when you need **automation** for stopping/starting compute on a schedule or as part of a pipeline. Ideal for CI/CD or MLOps workflows. |
| **Set up a schedule in Azure Machine Learning studio and include a shutdown time.** | Use when you want a **simple, no-code solution** to automatically shut down compute at a specific time daily. |
| **Set the autoscale of the compute instance to 0.**                        | Use for **compute clusters**, not single instances. This ensures the cluster scales down to zero nodes when idle, minimizing costs. |


| **Compute Type**       | **When to use it** | **Cost Considerations** |
|-------------------------|----------------------|---------------------------|
| **Inference Cluster**   | Use for **real-time or batch inference** at scale in production. | High cost due to AKS infrastructure and always-on nodes. Best for production workloads with strict latency requirements. |
| **Local Compute**       | Use for **development and quick testing** on your local machine. | No Azure cost, but limited scalability and performance. |
| **Compute Cluster**     | Use for **distributed training or large-scale batch jobs**. | Pay-as-you-go for active nodes. Use **autoscale to 0** when idle to minimize costs. |
| **Compute Instance**    | Use for **interactive development** (e.g., Jupyter notebooks). | Billed per hour while running. Set **shutdown schedules** to avoid unnecessary charges. |


Below is a self-contained scenario and code snippet showing how to route 10% of your inference traffic to the “blue” deployment and 90% to “green” using the Azure ML Python SDK v2.

---

## 📖 Scenario

A retail company uses Azure Machine Learning to forecast inventory levels.  
You’ve trained two versions of your forecasting model and deployed them side-by-side in a **blue/green** setup:  

- **“blue”** → the new candidate model  
- **“green”** → the current production model  

Now you want to perform a canary rollout: send **10%** of real-time inference requests to **blue** and **90%** to **green**.  

---

## 🔧 Python SDK v2 Code

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import ManagedOnlineEndpoint

# 1. Authenticate and connect to your workspace
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="YOUR_SUBSCRIPTION_ID",
    resource_group_name="YOUR_RESOURCE_GROUP",
    workspace_name="YOUR_WORKSPACE"
)

# 2. Load your existing endpoint
endpoint = ml_client.online_endpoints.get(name="inventory-forecast-endpoint")

# 3. Update traffic split for blue/green deployments
#    Send 10% of traffic to "blue", 90% to "green"
endpoint.traffic = {
    "blue": 10,
    "green": 90
}

# 4. Push the update
poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
endpoint = poller.result()

print(f"Updated traffic split: {endpoint.traffic}")
# → {'blue': 10, 'green': 90}
```

### ✏️ Key Points
- The `traffic` property is a **dict** whose keys are your deployment names and values are integer percentages.
- Percentages **must sum to 100** (e.g., 10 + 90 = 100).
- After calling `begin_create_or_update`, the endpoint will smoothly start routing requests according to this split.

---

That’s all you need—pass  
```python
{"blue": 10, "green": 90}
```  
as the `traffic` parameter to achieve a 10/90 canary rollout.


Here are the steps in the correct order for setting up Git access in Azure Machine Learning using SSH:

1. **Open the terminal window** in the Azure Machine Learning Notebook tab.  
2. **Generate a new SSH key** on the compute instance (e.g., `ssh-keygen -t rsa -b 4096`).  
3. **Add the public key to your Git account** (GitHub, Azure DevOps, etc.) under SSH keys.  
4. **Clone the Git repository** using the `git clone` command with the SSH URL.  

✅ This ensures secure, password-less Git operations from your Azure ML environment.

Do you want me to **also create a quick reference table** for these steps with **commands and tips** (e.g., where to find the public key, how to test the connection)? Or should I **add this to the DP-100 comprehensive cheat sheet under “MLOps & Git Integration”**?


Here’s the correct **step order (1, 2, 3, 4)** for setting up Azure ML with the Python SDK based on the context:

1. **Use the Azure portal to create the workspace**  
   (or CLI if you prefer automation, but the portal is the simplest starting point).  

2. **Install the Azure ML Python SDK**  
   (so you can interact with the workspace programmatically).  

3. **Import the workspace class**  
   ```python
   from azure.ai.ml.entities import Workspace
   ```  
   (needed for referencing or creating workspace objects in your code).  

4. **Use the `az ml workspace create` command**  
   (optional if you didn’t create the workspace in the portal; this is for CLI-based creation).  

---

✅ **Typical DP-100 Best Practice:**  
- If the question asks for enabling SDK usage → Step 2 is the key.  
- If the question asks for full setup → Follow 1 → 2 → 3 → 4 in that order.  

Do you want me to **prepare a quick visual flow or a Markdown table** showing **Portal vs CLI vs SDK setup paths** for your DP-100 cheat sheet?

Here’s an **overview for your study guide** based on these code snippets, focusing on **creating a workspace with the Azure ML Python SDK (v2)**:

---

### ✅ **Key Concepts**
- **Workspace**: The top-level resource in Azure ML that holds experiments, models, endpoints, and compute.
- **Customer Managed Key (CMK)**: Optional security feature to encrypt workspace data using your own key in Azure Key Vault.
- **Tags**: Metadata for organizing and tracking resources (e.g., `purpose="DP-100 Exam"`).

---

### **Correct Steps to Create a Workspace with SDK**
1. **Import required classes**:
   ```python
   from azure.ai.ml.entities import Workspace, CustomerManagedKey
   ```
2. **Define the workspace object**:
   - Include **name**, **location**, and optionally **display_name**, **description**, **tags**.
   - If using CMK, provide:
     - `key_vault` (full resource ID of Key Vault)
     - `key_uri` (the key identifier URI)
3. **Submit the creation request**:
   ```python
   ml_client.workspaces.begin_create(ws)
   ```

---

### ✅ **Best Practice Example**
```python
from azure.ai.ml.entities import Workspace, CustomerManagedKey

ws = Workspace(
    name="my_workspace",
    location="eastus",
    display_name="My Workspace",
    description="Example for DP-100",
    customer_managed_key=CustomerManagedKey(
        key_vault="/subscriptions/<SUBSCRIPTION_ID>/resourcegroups/<RESOURCE_GROUP>/providers/microsoft.keyvault/vaults/<VAULT_NAME>",
        key_uri="<KEY-IDENTIFIER>"
    ),
    tags={"purpose": "DP-100 Exam"}
)

ml_client.workspaces.begin_create(ws)
```

---

### **Common Mistakes to Avoid**
- Forgetting to call `ml_client.workspaces.begin_create(ws)` → The workspace object alone does not create the resource.
- Misplacing `customer_managed_key` or `tags` outside the `Workspace()` constructor.
- Using incorrect Key Vault resource ID or key URI format.

---

✅ Do you want me to **create a Markdown table summarizing “When to use CMK, tags, or basic workspace creation”** and then **add this to the DP-100 Comprehensive Cheat Sheet**?  
Or should I **prepare a full “Workspace Management” section for your study guide** with:
- **Portal vs CLI vs SDK**
- **Security options (CMK)**
- **Automation tips for DP-100**?

| **Option**          | **When to use it** |
|----------------------|----------------------|
| **FileDataset**      | Use when you need to work with **raw files or folders** (e.g., images, text files) without converting them into a structured table. Ideal for unstructured data. |
| **TabularDataset**   | Use when you need **structured, row-column data** (e.g., CSV, Parquet) for ML training or analysis. Supports direct conversion to Pandas or Spark DataFrames. |
| **mltable**          | Use when you want a **standardized, reusable dataset definition** for Azure ML pipelines and AutoML. Recommended for v2 SDK workflows. |
| **uri_folder**       | Use when you need to **reference a folder in storage** (e.g., for model artifacts or batch scoring) without creating a formal dataset object. |


| **Option**          | **When to use it** |
|----------------------|----------------------|
| **ComputeInstance**  | Use for **interactive development** (e.g., Jupyter notebooks, experimentation). Ideal for single-user environments and lightweight tasks. |
| **Environment**      | Use to **define and manage dependencies** (Python packages, Conda, Docker image) for training or inference jobs. Ensures reproducibility. |
| **Model**            | Use to **register and version trained models** in the Azure ML workspace for deployment or reuse. |
| **AmlCompute**       | Use for **scalable training or batch inference** on clusters. Supports autoscaling and distributed training for large datasets. |


| **Option**                                                   | **When to use it** |
|-------------------------------------------------------------|----------------------|
| **Use the Workspace object**                                | Use when you need to **connect to or manage the Azure ML workspace** (e.g., listing resources, submitting jobs, accessing datasets). It’s the entry point for most SDK operations. |
| **Apply the script to set up the environment for all compute instances** | Use when you want to **manually configure dependencies** on compute nodes outside of Azure ML’s managed environments. This is rarely recommended because it’s hard to maintain and not reproducible. |
| **Use the Environment object**                              | Use when you need to **define and manage dependencies (Conda, Docker)** for training or inference jobs. This ensures reproducibility and is the best practice for DP-100. |
| **Use the same compute instance for all scripts**           | Use only for **quick development or testing** when you don’t need scalability. Not recommended for production or collaborative scenarios. |


| **Task**                                | **When to do it** |
|-----------------------------------------|----------------------|
| **Set up the tracking environment**    | Do this **first**, before logging metrics or running experiments. It involves connecting to the Azure ML workspace and initializing the MLClient or Run context. |
| **Set the experiment name**            | Do this **after setting up the tracking environment**, so all runs are grouped under the correct experiment for organization and reporting. |
| **Start the training run**             | Do this **after defining the experiment name** and configuring the environment, so metrics and outputs are logged properly. |
| **Create an Azure Databricks resource**| Do this **only if your workflow requires Spark-based distributed processing** or big data integration. Not needed for standard Azure ML pipelines. |

Here’s a **comprehensive overview of common metrics, their formulas, interpretation, and model category** for your DP-100 study guide:

---

### ✅ **Model Evaluation Metrics Cheat Sheet**

| **Metric**       | **Formula** | **What it measures** | **Model Category** |
|-------------------|-------------|------------------------|----------------------|
| **Accuracy**      | (TP + TN) / (TP + TN + FP + FN) | Overall correctness of predictions. | Classification |
| **Precision**     | TP / (TP + FP) | Of predicted positives, how many are correct? | Classification |
| **Recall (Sensitivity)** | TP / (TP + FN) | Of actual positives, how many did we catch? | Classification |
| **F1-score**      | 2 × (Precision × Recall) / (Precision + Recall) | Balance between Precision and Recall. | Classification |
| **AUC (Area Under ROC Curve)** | — | Ability to rank positive vs negative classes. | Classification |
| **Log Loss**      | — | Penalizes incorrect predictions with confidence. | Classification |
| **R² (Coefficient of Determination)** | 1 - (SS_res / SS_tot) | How well the model explains variance in target. | Regression |
| **MSE (Mean Squared Error)** | (Σ (y - ŷ)²) / n | Average squared error; penalizes large errors. | Regression |
| **RMSE (Root Mean Squared Error)** | √MSE | Same as MSE but in original units. | Regression |
| **MAE (Mean Absolute Error)** | (Σ |y - ŷ|) / n | Average absolute error; robust to outliers. | Regression |
| **MAPE (Mean Absolute Percentage Error)** | (Σ |(y - ŷ)/y|) × 100 / n | Error as a percentage of actual values. | Regression |

---

### ✅ **Quick Guidance**
- **Classification Models** → Accuracy, Precision, Recall, F1, AUC, Log Loss.
- **Regression Models** → R², MSE, RMSE, MAE, MAPE.
- **Imbalanced Classes** → Focus on Recall, Precision, F1, and AUC (not Accuracy).
- **Business Impact**:
  - **High cost of false negatives** → Optimize Recall.
  - **High cost of false positives** → Optimize Precision.

---

🔥 Do you want me to **create a visual summary (diagram) showing which metrics apply to which model type** and **add this as part of the DP-100 Comprehensive Cheat Sheet**?  
Or should I **prepare the full cheat sheet now with all sections we discussed (Compute, Pipelines, Endpoints, Metrics, Fairness, Scheduling, etc.) in one Markdown file**?
