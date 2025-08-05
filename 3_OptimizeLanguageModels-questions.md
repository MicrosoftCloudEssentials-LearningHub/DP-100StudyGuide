# Optimize Language Models for AI Applications: <br/> Sample Questions and Answers

Costa Rica

[![GitHub](https://img.shields.io/badge/--181717?logo=github&logoColor=ffffff)](https://github.com/)
[brown9804](https://github.com/brown9804)

Last updated: 2025-01-04

----------

> [!NOTE]
> The questions and answers provided in this study guide are for practice purposes only and are not official practice questions.
> They are intended to help you prepare for the [DP-100 Microsoft certification exam](https://learn.microsoft.com/en-us/credentials/certifications/azure-data-scientist-associate/).
> For additional preparation materials and the most up-to-date information, please refer to the [official Microsoft documentation](https://learn.microsoft.com/en-us/training/paths/develop-ai-solutions-azure-openai/).

## Topics Covered
- Azure AI Foundry and Azure OpenAI integration
- Language model evaluation and optimization
- Retrieval Augmented Generation (RAG)
- Fine-tuning language models
- Responsible AI for language models

<details>
<summary><b>List of References </b> (Click to expand)</summary>

- [Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-studio/)
- [Prompt engineering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering)
- [Fine-tuning with Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning)

</details>

<details>
<summary><b>List of questions/answers </b> (Click to expand)</summary>

- [Q1: Azure AI Services vs Azure AI Foundry](#q1-azure-ai-services-vs-azure-ai-foundry)
- [Q2: Language Model Evaluation Metrics](#q2-language-model-evaluation-metrics)
- [Q3: RAG vs Fine-tuning](#q3-rag-vs-fine-tuning)
- [Q4: Search Methods in RAG](#q4-search-methods-in-rag)
- [Q5: Fine-tuning Parameters](#q5-fine-tuning-parameters)
- [Q6: Responsible AI Framework](#q6-responsible-ai-framework)
- [Q7: Model Benchmarks](#q7-model-benchmarks)
- [Q8: AI Project SDKs](#q8-ai-project-sdks)
- [Q9: Foundation Model Selection](#q9-foundation-model-selection)
- [Q10: Prompt Engineering Techniques](#q10-prompt-engineering-techniques)

</details>

> [!TIP]
> **Azure AI Resource Types:**

| Resource Type | Services Included | Best For |
|---------------|-------------------|----------|
| **Azure AI Services** | Speech, Language, Translator, Vision, Face, Custom Vision, Document Intelligence | Traditional AI services, single endpoint |
| **Azure AI Foundry** | OpenAI, Speech, Language, Content Safety, Translator, Vision, Face, Document Intelligence, Content Understanding | Generative AI development, integrated workflow |

> [!TIP]
> **Project Types in Azure AI:**

| Project Type | Resources | Use Case |
|--------------|-----------|----------|
| **Foundry Projects** | Azure AI Foundry resource | Standard generative AI development, chat apps, agents |
| **Hub-based Projects** | Azure AI Foundry + hub resource, managed compute, Prompt Flow, storage, key vault | Advanced AI development, Prompt Flow, fine-tuning, collaborative projects |

> [!TIP]
> **Language Model Evaluation Metrics:**

| Metric | Description | Measures |
|--------|-------------|----------|
| **Groundedness** | Alignment with input source/database | Factual accuracy |
| **Relevance** | Pertinence to the given input | Response appropriateness |
| **Coherence** | Logical flow and readability | Text quality |
| **Fluency** | Grammatical and linguistic accuracy | Language correctness |
| **Similarity** | Match between output and ground truth | Semantic similarity |

> [!TIP]
> **Search Methods in RAG:**

| Method | Description | Best For |
|--------|-------------|----------|
| **Keyword Search** | Exact keyword matching | Specific terms, structured queries |
| **Semantic Search** | Meaning-based retrieval | Natural language queries, context understanding |
| **Vector Search** | Mathematical vector similarity | Semantic similarity, unstructured data |
| **Hybrid Search** | Combines multiple methods | Comprehensive retrieval, best of all methods |

> [!TIP]
> **Fine-tuning Parameters:**

| Parameter | Description | Impact |
|-----------|-------------|---------|
| **batch_size** | Training examples per update | Larger = more stable, needs more memory |
| **learning_rate_multiplier** | Learning rate adjustment | Higher = faster learning, risk of overfitting |
| **n_epochs** | Full cycles through training data | More = better learning, risk of overfitting |
| **seed** | Reproducibility control | Same seed = reproducible results |

> [!TIP]
> **Responsible AI Framework (4 Ms):**

| Stage | Description |
|-------|-------------|
| **Map** | Identify potential harms relevant to your solution |
| **Measure** | Assess presence of harms in model outputs |
| **Mitigate** | Implement safeguards at multiple layers |
| **Manage** | Deploy and operate responsibly |

> [!TIP]
> **AI Evaluation Metrics Types:**

| Category | Metrics | Purpose |
|----------|---------|---------|
| **AI Quality** | Coherence, Relevance, Groundedness, Fluency | Assess response quality |
| **Risk and Safety** | Toxicity, Bias, Harmful content | Identify safety issues |
| **NLP Metrics** | BLEU, METEOR, ROUGE, F1-score | Traditional text evaluation |

## Q1: Azure AI Services vs Azure AI Foundry

> You're building a generative AI chat application that requires OpenAI models and content safety features. Which resource type should you choose?

- [ ] **Azure AI Services** ❌: `Incorrect. Azure AI Services doesn't include OpenAI models.`
- [ ] **Azure AI Foundry** ✅: `Correct. Azure AI Foundry includes OpenAI models and content safety features specifically for generative AI applications.`
- [ ] **Azure Machine Learning** ❌: `Incorrect. While possible, Azure AI Foundry is specifically designed for this use case.`
- [ ] **Azure Cognitive Services** ❌: `Incorrect. This is the older name for Azure AI Services.`

## Q2: Language Model Evaluation Metrics

> You want to ensure that your language model's responses are factually accurate and align with your knowledge base. Which metric should you prioritize?

- [ ] **Coherence** ❌: `Incorrect. Coherence measures logical flow, not factual accuracy.`
- [ ] **Fluency** ❌: `Incorrect. Fluency measures grammar, not factual accuracy.`
- [ ] **Groundedness** ✅: `Correct. Groundedness measures how well the model's output aligns with the input source or database.`
- [ ] **Similarity** ❌: `Incorrect. Similarity compares to ground truth but doesn't ensure factual accuracy.`

## Q3: RAG vs Fine-tuning

> When should you use RAG instead of fine-tuning for improving model performance?

- [ ] **When you need to change the model's behavior permanently** ❌: `Incorrect. This describes fine-tuning.`
- [ ] **When you want to incorporate up-to-date information from external sources** ✅: `Correct. RAG retrieves current information from external sources before generation.`
- [ ] **When you have a large dataset of example conversations** ❌: `Incorrect. This is better suited for fine-tuning.`
- [ ] **When you need to reduce model inference latency** ❌: `Incorrect. RAG adds retrieval overhead.`

## Q4: Search Methods in RAG

> You're implementing RAG for a system that needs to handle both specific technical terms and natural language queries. Which search method should you use?

- [ ] **Keyword Search only** ❌: `Incorrect. This won't handle natural language queries well.`
- [ ] **Semantic Search only** ❌: `Incorrect. This might miss specific technical terms.`
- [ ] **Vector Search only** ❌: `Incorrect. Similar limitation to semantic search.`
- [ ] **Hybrid Search** ✅: `Correct. Hybrid search combines multiple methods to handle both specific terms and natural language queries.`

## Q5: Fine-tuning Parameters

> You're fine-tuning a language model and want to balance learning speed with stability. You have a large dataset. How should you configure the parameters?

- [ ] **Large batch_size, high learning_rate_multiplier** ❌: `Incorrect. High learning rate with large batches can cause instability.`
- [ ] **Large batch_size, moderate learning_rate_multiplier** ✅: `Correct. Large batch sizes work well with large datasets and provide stable learning with moderate learning rates.`
- [ ] **Small batch_size, high learning_rate_multiplier** ❌: `Incorrect. Small batches with high learning rates can be very unstable.`
- [ ] **Small batch_size, low learning_rate_multiplier** ❌: `Incorrect. This would be very slow for large datasets.`

## Q6: Responsible AI Framework

> What is the correct order of the four stages in Microsoft's Responsible AI framework for generative models?

- [ ] **Measure, Map, Mitigate, Manage** ❌: `Incorrect. You need to identify harms before measuring them.`
- [ ] **Map, Measure, Mitigate, Manage** ✅: `Correct. First identify potential harms, then measure their presence, implement safeguards, and manage responsibly.`
- [ ] **Mitigate, Map, Measure, Manage** ❌: `Incorrect. You need to understand the problems before implementing solutions.`
- [ ] **Manage, Map, Measure, Mitigate** ❌: `Incorrect. Management comes after implementing safeguards.`

## Q7: Model Benchmarks

> You want to evaluate how well your model's generated text matches the grammatical and syntactic quality of human-written text. Which benchmark should you use?

- [ ] **Accuracy** ❌: `Incorrect. Accuracy is for classification tasks, not text quality.`
- [ ] **Coherence** ❌: `Incorrect. Coherence measures logical flow, not grammar.`
- [ ] **Fluency** ✅: `Correct. Fluency assesses grammatical rules, syntactic structures, and vocabulary usage.`
- [ ] **GPT Similarity** ❌: `Incorrect. This measures semantic similarity, not grammatical quality.`

## Q8: AI Project SDKs

> You're developing a Python application that needs to connect to an Azure AI Foundry project and access multiple AI services. Which SDK should you use?

- [ ] **Azure OpenAI SDK** ❌: `Incorrect. This only provides access to OpenAI services.`
- [ ] **Azure AI Services SDKs** ❌: `Incorrect. These are service-specific, not project-level.`
- [ ] **Azure AI Projects for Python** ✅: `Correct. This SDK enables connection to Azure AI Foundry projects and access to all connected resources.`
- [ ] **Azure Machine Learning SDK** ❌: `Incorrect. This is for traditional ML workflows, not AI Foundry projects.`

## Q9: Foundation Model Selection

> You're choosing a foundation model for fine-tuning. Your task involves processing short text snippets for sentiment analysis. Which consideration is most important?

- [ ] **Model capabilities for short text understanding** ✅: `Correct. Models like BERT are specifically designed for understanding short texts.`
- [ ] **Model size for faster inference** ❌: `Incorrect. While important, capability alignment is more critical.`
- [ ] **Pretraining data size** ❌: `Incorrect. Data quality and relevance matter more than size.`
- [ ] **Model popularity in the community** ❌: `Incorrect. Task alignment is more important than popularity.`

## Q10: Prompt Engineering Techniques

> You want to improve your language model's performance without fine-tuning or changing the model. Which approach should you use first?

- [ ] **Implement RAG immediately** ❌: `Incorrect. RAG adds complexity; try simpler approaches first.`
- [ ] **Use prompt engineering techniques** ✅: `Correct. Prompt engineering is quick, easy, and often very effective for improving model performance.`
- [ ] **Fine-tune the model** ❌: `Incorrect. Fine-tuning is more complex and resource-intensive.`
- [ ] **Switch to a larger model** ❌: `Incorrect. This increases costs without addressing the root issue.`

## Code Examples

### Azure AI Foundry Project Connection

```python
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# Connect to AI Foundry project
project_client = AIProjectClient(
    endpoint="https://<project-name>.<region>.api.azureml.ms",
    credential=DefaultAzureCredential()
)

# List available connections
connections = project_client.connections.list()
for conn in connections:
    print(f"Connection: {conn.name}, Type: {conn.connection_type}")

# Get specific connection with credentials
openai_connection = project_client.connections.get(
    connection_name="openai-connection",
    include_credentials=True
)
```

### RAG Implementation with Search

```python
from azure.search.documents import SearchClient
from azure.ai.projects import AIProjectClient
import openai

# Initialize search client
search_client = SearchClient(
    endpoint="https://<search-service>.search.windows.net",
    index_name="knowledge-base",
    credential=DefaultAzureCredential()
)

# Initialize OpenAI client through AI Foundry
project_client = AIProjectClient(
    endpoint="https://<project-name>.<region>.api.azureml.ms",
    credential=DefaultAzureCredential()
)

def generate_with_rag(query):
    # Retrieve relevant context
    search_results = search_client.search(
        search_text=query,
        search_mode="hybrid",  # Combines keyword and semantic search
        top=3
    )
    
    context = "\n".join([result["content"] for result in search_results])
    
    # Generate response with context
    messages = [
        {"role": "system", "content": "Use the provided context to answer questions accurately."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
    
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.1
    )
    
    return response.choices[0].message.content
```

### Fine-tuning Configuration

```python
from azure.ai.projects import AIProjectClient

# Create fine-tuning job
fine_tuning_job = {
    "model": "gpt-35-turbo",
    "training_file": "file-abc123",  # Previously uploaded training file
    "validation_file": "file-def456",  # Optional validation file
    "hyperparameters": {
        "batch_size": 32,
        "learning_rate_multiplier": 0.1,
        "n_epochs": 3,
        "seed": 42
    },
    "suffix": "custom-model-v1"
}

# Submit fine-tuning job
job = project_client.fine_tuning.jobs.create(**fine_tuning_job)
print(f"Fine-tuning job created: {job.id}")

# Monitor job status
while job.status not in ["succeeded", "failed", "cancelled"]:
    job = project_client.fine_tuning.jobs.retrieve(job.id)
    print(f"Status: {job.status}")
    time.sleep(30)
```

### Model Evaluation

```python
from azure.ai.evaluation import evaluate

# Define evaluation data
eval_data = [
    {
        "query": "What is machine learning?",
        "response": "Machine learning is a subset of AI...",
        "ground_truth": "Machine learning is a method of data analysis..."
    }
]

# Run evaluation
results = evaluate(
    data=eval_data,
    evaluators={
        "groundedness": GroundednessEvaluator(),
        "relevance": RelevanceEvaluator(),
        "coherence": CoherenceEvaluator(),
        "fluency": FluencyEvaluator()
    }
)

# View results
for metric, score in results.items():
    print(f"{metric}: {score}")
```

### Responsible AI Assessment

```python
# Define potential harms assessment
potential_harms = {
    "bias": "Model may favor certain demographic groups",
    "toxicity": "Model might generate offensive content",
    "misinformation": "Model could generate false information"
}

# Implement safety measures
safety_config = {
    "content_filters": {
        "hate": {"severity_threshold": "medium"},
        "sexual": {"severity_threshold": "medium"},
        "violence": {"severity_threshold": "medium"},
        "self_harm": {"severity_threshold": "medium"}
    },
    "prompt_injection_detection": True,
    "groundedness_detection": True
}

# Monitor and log safety metrics
def safe_generate(prompt, model="gpt-4"):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **safety_config
        )
        
        # Log safety metrics
        log_safety_metrics(prompt, response)
        
        return response.choices[0].message.content
    except Exception as e:
        log_safety_violation(prompt, str(e))
        return "I'm sorry, I can't assist with that request."
```

## Best Practices

1. **Model Selection**:
   - Evaluate model capabilities against your specific task
   - Consider pretraining data relevance and potential biases
   - Test multiple models before committing to fine-tuning

2. **RAG Implementation**:
   - Use hybrid search for comprehensive retrieval
   - Implement proper chunking strategies for documents
   - Monitor and update knowledge base regularly

3. **Fine-tuning**:
   - Start with prompt engineering before fine-tuning
   - Use high-quality, diverse training data
   - Monitor for overfitting with validation sets

4. **Responsible AI**:
   - Implement content filters and safety measures
   - Regular bias and fairness assessments
   - Maintain human oversight for high-stakes applications

5. **Evaluation**:
   - Use multiple metrics for comprehensive assessment
   - Implement continuous monitoring in production
   - Regular model performance reviews

<!-- START BADGE -->
<div align="center">
  <img src="https://img.shields.io/badge/Total%20views-1391-limegreen" alt="Total views">
  <p>Refresh Date: 2025-08-05</p>
</div>
<!-- END BADGE -->
