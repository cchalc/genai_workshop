# Databricks notebook source
# MAGIC %md
# MAGIC # Agentic Workflows With Databricks
# MAGIC
# MAGIC This demo goes over deploying Agentic workflows with Databricks, LangChain, and MLFlow. We are going to cover:
# MAGIC 1. **Deploying a custom model with LangChain**
# MAGIC 2. **Reviewing and evaluating our foundation model with our Review App and MLFlow**
# MAGIC 3. Adding a vector search retriever as an agent
# MAGIC 4. Adding a SQL database as an agent
# MAGIC 5. Adding genie as an agent
# MAGIC
# MAGIC This notebook covers the first two items - setting up a custom foundation model, deploying it, and then using Databricks tooling to review and evaluate it.

# COMMAND ----------

# MAGIC %run "./0 Initialize"

# COMMAND ----------

CATALOG = 'shm'
SCHEMA = 'default'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploying a Custom Model
# MAGIC
# MAGIC This is an elementary example of deploying a custom model. It sets the stage for evolving it agentic tooling. There are a couple things introduced below:
# MAGIC
# MAGIC - Use MLFlow for model tracing & evaluation
# MAGIC - Load a model configuration from CICD
# MAGIC - Setup a custom `ChatPromptTemplate`
# MAGIC - Use the `ChatDatabricks` inferface --> endpoint
# MAGIC - Create a `chain` and invoke it
# MAGIC
# MAGIC Things are moving extremely fast in the generative AI space and all the orchestration packages (DsPy, LangChain, LlamaIndex, etc.) are changing rapidly. 
# MAGIC
# MAGIC Serving an endpoint is the the only way to establish any kind of stability.
# MAGIC
# MAGIC MLFLow helps a lot here through LLM [integrations with LangChain, OpenAI and LlamaIndex](https://mlflow.org/docs/latest/llms/tracing/index.html#automatic-tracing)

# COMMAND ----------

fm_chain_config = {
    "fm_serving_endpoint_name": "databricks-dbrx-instruct",
    "fm_prompt_template": """You are an assistant that answers questions. Your goal is to provide an accurate response. If you don't know the answer answer with 'I don't know'.""",
    "temperature": 0.01,
    "max_tokens": 500
}

# COMMAND ----------

import mlflow
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

# Enable MLflow tracing & logging
mlflow.langchain.autolog()

# Load the chain configuration
model_config = mlflow.models.ModelConfig(development_config=fm_chain_config)

# Define a prompt template
prompt = ChatPromptTemplate.from_messages(
    [  
        ("system", model_config.get("fm_prompt_template")),
        ("user", "{question}")
    ]
)

# Setup a foundation model
model = ChatDatabricks(
    endpoint=model_config.get("fm_serving_endpoint_name"),
    extra_params={
      "temperature": model_config.get("temperature"), "max_tokens": model_config.get("max_tokens")}
)

# Define the basic chain
chain = (
    prompt
    | model
    | StrOutputParser()
)

#Let's try our prompt:
answer = chain.invoke({'question':'How to start a Databricks cluster?'})
display_txt_as_html(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register the Model
# MAGIC
# MAGIC We want to deploy this foundation model with our custom prompt template. In order to do that, we need to generate an [MLFlow signature](https://mlflow.org/docs/latest/model/signatures.html) that tells users how to infer with the model, specifically the expected input and output. We then register the model in Unity Catalog as a LangChain model. Because the models don't have binary files (e.g. scikit-learn pickle files), we need to provide a path to the chain (`fm_chain.py`) and the configuration (`fm_chain_config.yaml`)

# COMMAND ----------

## Log the model and register it to Unity Catalog ##
input_example = {"messages": [
  {"role": "user", "content": "What is Retrieval-augmented Generation?"}
  ]}

output_example = 'To start a Databricks cluster, hit the button'

fm_model_name = "shm-fm"

signature = infer_signature(input_example, output_example)

# Log the model to MLflow
with mlflow.start_run(run_name="shm-fm"):
  logged_chain_info = mlflow.langchain.log_model(
          lc_model=os.path.join(os.getcwd(), '1_fm_chain.py'),
          model_config=fm_chain_config, 
          artifact_path="chain",
          input_example=input_example,
          signature=signature,
          example_no_conversion=True,
      )


fm_model_path = f"{CATALOG}.{SCHEMA}.{fm_model_name}"

# Register to UC
uc_registered_model_info = mlflow.register_model(
  model_uri=logged_chain_info.model_uri, 
  name=fm_model_path
  )

# COMMAND ----------

## Deploy the Custom Model as a REST Endpoint ##
if False:
  deployment_info = agents.deploy(
    fm_model_path,
    model_version=3,
    scale_to_zero=True
    )

  instructions_to_reviewer = f"""
  ## Testing Instructions
  Your inputs are essential for the development team - help us improve our bot.
  """

  # Add the user-facing instructions to the Review App
  agents.set_review_instructions(fm_model_path, instructions_to_reviewer)

# COMMAND ----------

## Evaluate the Model Based on User Feedback With MLFLow##
eval_dataset = (
    spark.table("shm.dbdemos_llm_rag.eval_set_databricks_documentation")
    .limit(10)
    .toPandas()
)
display(eval_dataset)

with mlflow.start_run(run_id=logged_chain_info.run_id):
    # Evaluate the logged model
    eval_results = mlflow.evaluate(
        data=eval_dataset,
        model=logged_chain_info.model_uri,
        model_type="databricks-agent",
    )

# COMMAND ----------


