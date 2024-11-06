# Databricks notebook source
# MAGIC %pip install -U --quiet databricks-sdk==0.28.0 databricks-agents mlflow-skinny mlflow mlflow[gateway] databricks-vectorsearch langchain==0.2.1 langchain_core==0.2.5 langchain_community==0.2.4 langchain_openai databricks-sql-connector==2.9.3 sqlalchemy==1.4.50
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
import os
import yaml
import pandas as pd
import requests
import time
from datetime import datetime
from operator import itemgetter

from databricks import agents
from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from langchain.agents import Tool, initialize_agent, AgentType, create_sql_agent, AgentExecutor
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.embeddings import DatabricksEmbeddings
from langchain.llms.openai import OpenAI
from langchain.sql_database import SQLDatabase
from langchain_core.messages.ai import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from mlflow.models import infer_signature
from pyspark.sql.types import *

# COMMAND ----------

type_mapping = {
    'STRING': StringType(),
    'INT': IntegerType(),
    'LONG': LongType(),
    'FLOAT': FloatType(),
    'DOUBLE': DoubleType(),
    'SHORT': ShortType(),
    'BYTE': ByteType(),
    'BOOLEAN': BooleanType(),
    'DATE': DateType(),
    'TIMESTAMP': TimestampType(),
    'BINARY': BinaryType(),
    'DECIMAL': DecimalType(10, 0)
}

class GenieTool:
    """
    This tool uses a `requests` based approach to communicating with Genie 
    and passes the results into LangChain using a pandas dataframe.

    The approach:
    1. Start a conversation
    2. Create a Message
    3. Get the Message
    4. Get the SQL Query Response
    5. Parse the SQL into a Dataframe
    6. Pass the Dataframe back via the Agentic framework
    """
    def __init__(self):
        self.host = 'adb-984752964297111.11.azuredatabricks.net'
        self.api_base = 'api/2.0/genie/spaces'
        self.space_id = '01ef9b9112741d24ae265133816bf71f'
        self.token = dbutils.secrets.get("shm", "genie")
        self.headers = {
            "Context-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }
        self.base_url = f"https://{self.host}/{self.api_base}/{self.space_id}"

    def convert_genie_query_result_to_df(self):
        columns = self.query_result['statement_response']['manifest']['schema']['columns']
        rows = []

        for item in self.query_result['statement_response']['result']['data_typed_array']:
            row = []
            for i, value in enumerate(item['values']):
                type_name = columns[i]['type_name']
                val = value.get('str')

                try:
                    if type_name == 'STRING':
                        row.append(val)
                    elif type_name in {'INT', 'LONG', 'SHORT', 'BYTE'}:
                        row.append(int(val))
                    elif type_name in {'FLOAT', 'DOUBLE', 'DECIMAL'}:
                        row.append(float(val))
                    elif type_name == 'BOOLEAN':
                        row.append(val.lower() == 'true')
                    elif type_name == 'DATE':
                        row.append(datetime.strptime(val, '%Y-%m-%d').date())
                    elif type_name == 'TIMESTAMP':
                        row.append(datetime.strptime(val, '%Y-%m-%dT%H:%M:%S.%fZ'))
                    elif type_name == 'BINARY':
                        row.append(bytes(val, 'utf-8'))
                    else:
                        row.append(None)
                except:
                    if type_name == 'STRING':
                        row.append(val)
                    elif type_name in {'INT', 'LONG', 'SHORT', 'BYTE'}:
                        row.append(int(val))
                    elif type_name in {'FLOAT', 'DOUBLE', 'DECIMAL'}:
                        row.append(float(val))
                    elif type_name == 'BOOLEAN':
                        row.append(val.lower() == 'true')
                    elif type_name == 'DATE':
                        row.append(datetime.strptime(val, '%Y-%m-%d').date())
                    elif type_name == 'TIMESTAMP':
                        row.append(datetime.strptime(val, '%Y-%m-%d %H:%M:%S'))
                    elif type_name == 'BINARY':
                        row.append(bytes(val, 'utf-8'))
                    else:
                        row.append(None)
            rows.append(row)

        schema = StructType([StructField(col['name'], type_mapping[col['type_name']], True) for col in columns])
        self.result_df = spark.createDataFrame(rows, schema).toPandas()
        
    def start_conversation(self, question: str):
        """
        A POST request to start a new conversation and return the `conversation_id` and get the `message_id` back.
        """
        question_json = {"content": question}
        convo_url = f"{self.base_url}/start-conversation"
        response = requests.post(
            url=convo_url,
            headers=self.headers,
            json=question_json
        ).json()
        self.convo_id = response['conversation_id']
        self.message_id = response['message_id']
        self.convo_url = f"{self.base_url}/conversations/{self.convo_id}/messages/{self.message_id}"

    def get_query(self):
        self.query = self.response['attachments'][0]['query']['query']

    def get_description(self):
        self.description = self.response['attachments'][0]['query']['description']

    def get_query_result_df(self):
        query_result_url = f"{self.convo_url}/query-result"
        self.query_result = requests.get(url=query_result_url, headers=self.headers).json()
        self.convert_genie_query_result_to_df()

    def retrieve_query_results(self):
        retries = 0
        while retries < 10:
            try:
                time.sleep(2)
                self.response = requests.get(url=self.convo_url, headers=self.headers).json()
                self.get_query()
                self.get_description()
                self.get_query_result_df()
                break
            except Exception as e:
                print("Genie here, I'm thinking")
                retries += 1

    def query(self, question: str) -> pd.DataFrame:
        self.start_conversation(question)
        self.retrieve_query_results()
        return self.result_df

# COMMAND ----------

def endpoint_exists(vsc, vs_endpoint_name):
  try:
    return vs_endpoint_name in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]
  except Exception as e:
    #Temp fix for potential REQUEST_LIMIT_EXCEEDED issue
    if "REQUEST_LIMIT_EXCEEDED" in str(e):
      print("WARN: couldn't get endpoint status due to REQUEST_LIMIT_EXCEEDED error. The demo will consider it exists")
      return True
    else:
      raise e

def index_exists(vsc, endpoint_name, index_full_name):
    try:
        vsc.get_index(endpoint_name, index_full_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue.')
            raise e
    return False

# COMMAND ----------

def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

def extract_previous_messages(chat_messages_array):
    messages = "\n"
    for msg in chat_messages_array[:-1]:
        messages += (msg["role"] + ": " + msg["content"] + "\n")
    return messages

def combine_all_messages_for_vector_search(chat_messages_array):
    return extract_previous_messages(chat_messages_array) + extract_user_query_string(chat_messages_array)

def display_txt_as_html(txt):
    if isinstance(txt, str):
        txt = txt.replace('\n', '<br/>')
    elif isinstance(txt, AIMessage):
        txt = txt.content.replace('\n', '<br/>')
    displayHTML(f'<div style="max-height: 150px">{txt}</div>')
