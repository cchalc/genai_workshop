# Databricks notebook source
# MAGIC %md
# MAGIC # Agentic Workflows With Databricks
# MAGIC
# MAGIC This demo goes over deploying Agentic workflows with Databricks, LangChain, and MLFlow. We are going to cover:
# MAGIC 1. Deploying a custom model with LangChain
# MAGIC 2. Reviewing and evaluating our foundation model with our Review App and MLFlow
# MAGIC 3. Adding a vector search retriever as a retrieval augmented generation (RAG) agent
# MAGIC 4. **Adding a SQL database as an agent**
# MAGIC 5. **Adding genie as an agent**
# MAGIC
# MAGIC This notebook covers four and five - we introduce two more agents, one for a Genie space and one for a serverless SQL database. We layer these along with RAG to create a compound chain where the agent can select which tool to use based on the query.

# COMMAND ----------

# MAGIC %run "./0 Initialize"

# COMMAND ----------

# define some globals
CATALOG = 'shm'
SCHEMA = 'dbdemos_llm_rag'
RAG_SCHEMA = 'dbdemos_llm_rag'

# Setup our foundation model (used extensively)
model = ChatDatabricks(
    endpoint='databricks-dbrx-instruct',
    extra_params={"temperature": 0.1, "max_tokens": 500}
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Genie as a Generative AI Tool
# MAGIC Genie spaces are amazing because you can curate multiple tables and use text-to-sql to return a focused response to a query. We can leverage this in a compound Gen AI application by creating a tool. To do this we leverage the Genie API (currently in private preview) to do a REST API call to the Genie space. This call initiates a conversation, gets a query back, executed the query, and gets the results of that query back. 
# MAGIC
# MAGIC Note how the example below implements the [ReAct Chain of Thought](https://arxiv.org/abs/2210.03629) framework by default.

# COMMAND ----------

from langchain_core.prompts import PromptTemplate

template = """
You are a trusted AI assistant that helps answer questions based only on the provided information. If you do not know the answer to a question, say you do not know. Here is the history of the current conversation you are having with your user: {chat_history}.

Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question.

Always replace COUNT(\*) with COUNT(*)

Begin!

Question: {question}
Thought:{agent_scratchpad}

"""

input_variables = ['tools', 'tool_names','question','agent_scratchpad']

prompt = PromptTemplate(
  template=template,
  input_variables=input_variables
)

# COMMAND ----------

from langchain.agents import create_react_agent

# Setup a Genie API Tool (source code in initialize)
genie_tool = Tool(
        name="Genie Tool",
        func=GenieTool().query,
        description="Useful for quering specific data about our operations, especially plant information. You should format requests to this tool as specific questions that are closely related to the user input."
    )

# Initialize an agent to use the tool
agent = create_react_agent(
    prompt=prompt,
    tools=[genie_tool], 
    llm=model,
)

agent_executor = AgentExecutor(
    agent=agent, 
    tools=[genie_tool],
    verbose=True
)

def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

def extract_previous_messages(chat_messages_array):
    messages = "\n"
    for msg in chat_messages_array[:-1]:
        messages += (msg["role"] + ": " + msg["content"] + "\n")
    return messages

chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_previous_messages)
    }
    | agent_executor
)

chain.invoke({"messages": [
  {"content": "What is the total number of malfunctions? Ignore malfunctions with a type of NULL", "role":"user"}
  ]})

# COMMAND ----------

# MAGIC %md
# MAGIC ## SQL as a Generative AI Tool
# MAGIC It is also possible to add a table using serverless SQL. This can provide a low latency way to retrieve information that is more determinstic than a Genie space. For example if the shop floor wanted part inventory numbers in real time, we could provide a database to query and allow the foundation model to determine how to query the database based on it's schema.
# MAGIC
# MAGIC

# COMMAND ----------

# Set up Databricks SQL connection
db = SQLDatabase.from_databricks(catalog='shm', schema='iot_turbine')
agent = create_sql_agent(
    llm=model, 
    db=db,
    verbose=True,
    handle_parsing_errors=True
    )

chain = (
    {
        "input": itemgetter("messages") | RunnableLambda(extract_user_query_string),
    }
    | agent
)

chain.invoke({"messages": [
  {"content": "How many turbine anomalies have we detected?", "role":"user"}
  ]})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieval as a Tool
# MAGIC We previously saw how RAG was useful for augmenting a standard query. In this case, we are converting our Vector search to yet another tool in our arsenal. We can then allow our foundation model to route the query to either retrieval, or another toolkit.

# COMMAND ----------

# Get configuration
import mlflow
model_config = mlflow.models.ModelConfig(development_config='2_rag_chain_config.yaml')
resources_config = model_config.get("resources")
retriever_config = model_config.get("retriever")

# Retrieval as a tool
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
).as_retriever()

# setup the retriever as an optional tool
from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever=vector_search_as_retriever,
    name="Documentation Search",
    description="Searches and returns relevant documents. Use this tool when you need to find specific information on technical questions related to Databricks, or programming related questions."
)

retriever_tool.invoke("How do we use embeddings in Databricks?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Making a compound agentic system
# MAGIC We are now going to initialize an agent that can use all the above tools. We are going to use the SQL agent as the base of this chain, but there are numerous ways to setup a chain. This example is slightly contrived and intended to demonstrate how agents switch between tools and demonstrate non-determinism (why we need good observability).

# COMMAND ----------

## Enable MLflow Tracing
mlflow.langchain.autolog()

# Prompt Template for generation
prompt = PromptTemplate(
    template=template,
    input_variables=input_variables,
)

agent = create_sql_agent(
    llm=model, 
    db=db,
    prompt=prompt,
    extra_tools=[retriever_tool, genie_tool],
    verbose=True,
    handle_parsing_errors=True,
    output_parser=StrOutputParser()
    )

chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "chat_history": itemgetter("messages") | RunnableLambda(extract_previous_messages),
        "input": itemgetter("messages") | RunnableLambda(extract_user_query_string),
    }
    | agent
)

chain.invoke({"messages": [
  {"content": "How many parts do I have in the database?", "role":"user"}
  ]})

# COMMAND ----------


