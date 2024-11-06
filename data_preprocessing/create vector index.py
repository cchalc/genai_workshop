# Databricks notebook source
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install langchain_community

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Enable CDC for Vector Search Delta Sync | chunked_docs
# MAGIC ALTER TABLE nam_workshop.default.spotify_dataset 
# MAGIC SET TBLPROPERTIES (delta.enableChangeDataFeed = true)

# COMMAND ----------

catalog = 'nam_workshop'
table_name = 'spotify_dataset'

# COMMAND ----------

from pyspark.sql import functions as F

(
  spark.table(f"{catalog}.default.{table_name}")
  .withColumn("chunk_id", F.monotonically_increasing_id())
  .write.format("delta")
  .mode("overwrite")
  .option("mergeSchema", "true")  # Enable schema merging
  .saveAsTable(f"{catalog}.default.{table_name}")
)

df = spark.table(f'{catalog}.default.{table_name}')
display(df.limit(5))

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

# Get the vector search index
vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

endpoint = 'vs_endpoint_1'#'one-env-shared-endpoint-nam'
index_name = f'{catalog}.default.vectorsearch_index_spotify'
column_content = 'Review' #text
force_delete = False

def find_index(vsc, endpoint_name, index_name):
    all_indexes = vsc.list_indexes(name=endpoint_name).get("vector_indexes", [])
    return index_name in map(lambda i: i.get("name"), all_indexes)
  
if find_index(
  vsc,
  endpoint_name=endpoint, 
  index_name=index_name):
    if force_delete:
        vsc.delete_index(
          endpoint_name=endpoint, index_name=index_name
          )
        create_index = True
    else:
        create_index = False
else:
    create_index = True

if create_index:
    vsc.create_delta_sync_index_and_wait(
        endpoint_name=endpoint,
        index_name=index_name,
        primary_key="chunk_id",
        source_table_name=f'{catalog}.default.{table_name}',
        pipeline_type='TRIGGERED',
        embedding_source_column=column_content,
        embedding_model_endpoint_name='databricks-bge-large-en'
    )

# COMMAND ----------

index_name = f"{catalog}.default.vectorsearch_spotify" #vectorsearch_spotify

# COMMAND ----------

# Setup the index as a LangChain retriever
vs_index = vsc.get_index(
  endpoint_name=endpoint, 
  index_name=index_name
)

# COMMAND ----------

# Import the necessary class
from langchain.embeddings import DatabricksEmbeddings
from langchain.vectorstores import DatabricksVectorSearch

# Create the retriever
# 'k' is the number of results to return
embedding_model = DatabricksEmbeddings(
    endpoint="databricks-bge-large-en"
)

vs_retriever = DatabricksVectorSearch(
    vs_index, 
    text_column=column_content
).as_retriever(search_kwargs={"k": 10})

# COMMAND ----------

vs_retriever.invoke("song")

# COMMAND ----------


