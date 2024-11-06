# Databricks notebook source
# MAGIC %md
# MAGIC # Agentic Workflows With Databricks
# MAGIC
# MAGIC This demo goes over deploying Agentic workflows with Databricks, LangChain, and MLFlow. We are going to cover:
# MAGIC 1. Deploying a custom model with LangChain
# MAGIC 2. Reviewing and evaluating our foundation model with our Review App and MLFlow
# MAGIC 3. **Adding a vector search retriever as a retrieval augmented generation (RAG) agent**
# MAGIC 4. Adding a SQL database as an agent
# MAGIC 5. Adding genie as an agent
# MAGIC
# MAGIC This notebook covers item three - setting up a model with vector search and adding it to the chain to create a RAG application. In this example we are going to use the Databricks documentation as an example.

# COMMAND ----------

# MAGIC %run "./0 Initialize"

# COMMAND ----------

# define some globals
CATALOG = 'shm'
SCHEMA = 'dbdemos_llm_rag'
RAG_SCHEMA = 'dbdemos_llm_rag'

# COMMAND ----------

# MAGIC %md
# MAGIC ## The Journey to RAG
# MAGIC This demo won't go over loading documents. But the steps are roughly the same for every RAG chain. Our primary goal is to establish a vector search and use it as a retriever in LangChain. A Vector Search is a powerful way to interact with unstructured text like maintenance manuals, shop floor reports, etc. In order to setup a vector search we need the following:
# MAGIC
# MAGIC 1. Load the raw documentation
# MAGIC 2. Parse and chunk the documentation into a table
# MAGIC 3. Create a `vector index` with managed embeddings
# MAGIC 4. Create a `vector search endpoint` to query the `vector index`
# MAGIC 5. Convert the endpoint to a retrieval tool.
# MAGIC
# MAGIC A detailed review of document chunking is provided in our [LLM RAG Chatbot Demo](https://www.databricks.com/resources/demos/tutorials/data-science-and-ai/lakehouse-ai-deploy-your-llm-chatbot)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The raw documentation
# MAGIC SELECT *
# MAGIC FROM shm.dbdemos_llm_rag.raw_documentation
# MAGIC LIMIT 1

# COMMAND ----------

# MAGIC %sql
# MAGIC -- The parsed and chunked table
# MAGIC SELECT * 
# MAGIC FROM shm.dbdemos_llm_rag.databricks_documentation
# MAGIC LIMIT 6

# COMMAND ----------

# Create a managed vector search index
vsc = VectorSearchClient()

vs_endpoint_name = 'dbdemos_vs_endpoint'
vs_index_fullname = f"{CATALOG}.{RAG_SCHEMA}.databricks_documentation_vs_index"
source_table_fullname = f"{CATALOG}.{RAG_SCHEMA}.databricks_documentation"

# Create a served endpoint 
if not endpoint_exists(vsc, vs_endpoint_name):
    vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")

# Create a managed delta sync vector index
if not index_exists(vsc, vs_endpoint_name, vs_index_fullname):
    vsc.create_delta_sync_index(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_fullname,
        source_table_name=source_table_fullname,
        pipeline_type="TRIGGERED",
        primary_key="id",
        embedding_source_column='content',
        embedding_model_endpoint_name='databricks-gte-large-en' 
    )

# Setup the index as a LangChain retriever
vs_index = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=vs_index_fullname)

# Create the retriever
# 'k' is the number of results to return
embedding_model = DatabricksEmbeddings(
    endpoint="databricks-bge-large-en"
    )
    
vs_retriever = DatabricksVectorSearch(
    vs_index, 
    text_column="content", 
    embedding=embedding_model
    ).as_retriever(search_kwargs={"k": 1})

# COMMAND ----------

vs_retriever.invoke(
  "How can I generate a vector search with my own embeddings?"
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup the model configuration
# MAGIC As we add complexity, using CICD practices becomes important, so we store our configuration in diffable files.

# COMMAND ----------

rag_chain_config = {
    "resources": {
        "llm_endpoint_name": "databricks-dbrx-instruct",
        "vector_search_endpoint_name": vs_endpoint_name,
    },
    "input_example": {
        "messages": [{"content": "How do I use embeddings in Databricks?", "role": "user"}]
    },
    "output_example": {
        'content': 'To start a Databricks cluster, hit the button',
        'response_metadata': {'prompt_tokens': 65,'completion_tokens': 108, 'total_tokens': 173},
        'type': 'ai',
        'id': 'run-d304fcfb-09e6-4eb3-83df-b6692fdc1c0a-0'
    },
    "llm": {
        "hyperparameters": {"max_tokens": 500, "temperature": 0.01},
        "prompt_template": "You are a trusted AI assistant that helps answer questions based only on the provided information. If you do not know the answer to a question, say you do not know. Here is the history of the current conversation you are having with your user: {chat_history}. Here is some context which may or may not help you answer the following question: {context}. Answer directly, do not repeat the question and be concise. Start with 'Mangabot here'. Based on this context, answer this question: {question}",
        "prompt_template_variables": ["context", "chat_history", "question"],
    },
    "retriever": {
        "chunk_template": "Passage: {chunk_text}\n",
        "data_pipeline_tag": "poc",
        "parameters": {"k": 3, "query_type": "ann"},
        "schema": {"chunk_text": "content", "document_uri": "url", "primary_key": "id"},
        "vector_search_index": f"{CATALOG}.{RAG_SCHEMA}.databricks_documentation_vs_index",
    },
}

# save for model config
with open('./2_rag_chain_config.yaml', 'w') as f:
    yaml.dump(rag_chain_config, f)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and test the RAG Chain
# MAGIC Here we bring everything together - a foundation model, a vector store retriever, and a configuration setup. We now have a functional RAG model. The complexity starts to ramp up with juggling some of the LangChain syntax, but all we are really doing is:
# MAGIC
# MAGIC 1. Enable MLFLow tracing
# MAGIC 2. Get the configuration from the file
# MAGIC 3. Setup the retriever
# MAGIC 4. Build the chain
# MAGIC
# MAGIC The chunk below creates the entire deployed model using the work we built above.

# COMMAND ----------

import os
import mlflow
from operator import itemgetter
from databricks.vector_search.client import VectorSearchClient
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

## Enable MLflow Tracing
mlflow.langchain.autolog()

# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

def extract_previous_messages(chat_messages_array):
    messages = "\n"
    for msg in chat_messages_array[:-1]:
        messages += (msg["role"] + ": " + msg["content"] + "\n")
    return messages

def combine_all_messages_for_vector_search(chat_messages_array):
    return extract_previous_messages(chat_messages_array) + extract_user_query_string(chat_messages_array)

# Get the configuration from our YAML file
model_config = mlflow.models.ModelConfig(development_config='2_rag_chain_config.yaml')
resources_config = model_config.get("resources")
retriever_config = model_config.get("retriever")
llm_config = model_config.get("llm")

# Connect to the Vector Search Index
vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index(
    endpoint_name=resources_config.get("vector_search_endpoint_name"),
    index_name=retriever_config.get("vector_search_index"),
)
vector_search_schema = retriever_config.get("schema")

# Turn the Vector Search index into a LangChain retriever
vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column=vector_search_schema.get("chunk_text"),
    columns=[
        vector_search_schema.get("primary_key"),
        vector_search_schema.get("chunk_text"),
        vector_search_schema.get("document_uri"),
    ],
).as_retriever(search_kwargs=retriever_config.get("parameters"))

# Required to:
# 1. Enable the RAG Studio Review App to properly display retrieved chunks
# 2. Enable evaluation suite to measure the retriever
mlflow.models.set_retriever_schema(
    primary_key=vector_search_schema.get("primary_key"),
    text_column=vector_search_schema.get("chunk_text"),
    doc_uri=vector_search_schema.get("document_uri")
)

# Method to format the docs returned by the retriever into the prompt
def format_context(docs):
    chunk_template = retriever_config.get("chunk_template")
    chunk_contents = [
        chunk_template.format(
            chunk_text=d.page_content,
        )
        for d in docs
    ]
    return "".join(chunk_contents)


# Prompt Template for generation
prompt = PromptTemplate(
    template=llm_config.get("prompt_template"),
    input_variables=llm_config.get("prompt_template_variables"),
)

# FM for generation
model = ChatDatabricks(
    endpoint=resources_config.get("llm_endpoint_name"),
    extra_params=llm_config.get("hyperparameters"),
)

# Setup the RAG Chain
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(combine_all_messages_for_vector_search)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_previous_messages)
    }
    | prompt
    | model
    | StrOutputParser()
)

# Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=chain)

# COMMAND ----------

chain.invoke({"messages": [
  {"content": "How do I use embeddings in Databricks?", "role":"user"}
  ]})

# COMMAND ----------

# MAGIC %md
# MAGIC # Register the Model
# MAGIC
# MAGIC We want to deploy this foundation model with our custom prompt template. In order to do that, we need to generate an [MLFlow signature](https://mlflow.org/docs/latest/model/signatures.html) that tells users how to infer with the model, specifically the expected input and output. We then register the model in Unity Catalog as a LangChain model. Because the models don't have binary files (e.g. scikit-learn pickle files), we need to provide a path to the chain (`fm_chain.py`) and the configuration (`fm_chain_config.yaml`)

# COMMAND ----------

## Log the model and register it to Unity Catalog ##

rag_model_name = "shm-rag"

signature = infer_signature(
  model_config.get('input_example'), 
  model_config.get('output_example')
  )

# Log the model to MLflow
with mlflow.start_run(run_name=rag_model_name):
  logged_chain_info = mlflow.langchain.log_model(
          lc_model=os.path.join(os.getcwd(), '2_rag_chain.py'),
          model_config=rag_chain_config, 
          artifact_path="chain",
          input_example=model_config.get('input_example'),
          signature=signature
      )

rag_model_path = f"{CATALOG}.{SCHEMA}.{rag_model_name}"

# Register to UC
uc_registered_model_info = mlflow.register_model(
  model_uri=logged_chain_info.model_uri, 
  name=rag_model_path
  )

# COMMAND ----------

## Deploy the Custom Model as a REST Endpoint ##
# from databricks import agents
# deployment_info = agents.deploy(
#   rag_model_path,
#   model_version=1,
#   scale_to_zero=True
#   )

# instructions_to_reviewer = f"""
# ## Testing Instructions
# This is the RAG Review App.
# """

# # Add the user-facing instructions to the Review App
# agents.set_review_instructions(rag_model_path, instructions_to_reviewer)

# COMMAND ----------

## Evaluate the Model Based on User Feedback With MLFLow##
eval_dataset = (
    spark.table("shm.dbdemos_llm_rag.eval_set_databricks_documentation")
    .limit(50)
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
