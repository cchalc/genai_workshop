# Databricks notebook source
# MAGIC %pip install pypdf beautifulsoup4

# COMMAND ----------

from pdf_parser import parse_pdf_to_json, load_json, chunk_json_data_to_df
from pdf_scraper import find_pdfs
from pathlib import Path

# COMMAND ----------

url = 'https://www.cn.ca/en/supplier-portal/supplier-portal/policies-and-guidelines'
pdf_folder = '/Volumes/shm/default/raw_pdfs/'
json_folder = '/Volumes/shm/default/raw_json/'

# COMMAND ----------

find_pdfs(url, folder = pdf_folder)

# COMMAND ----------

pdf_dir = Path(pdf_folder)
pdf_files = list(pdf_dir.glob('*.pdf'))
for pdf_file in pdf_files:
    parse_pdf_to_json(pdf_file, json_folder)

# COMMAND ----------

json_dir = Path(json_folder)
json_files = list(json_dir.glob('*.json'))

combined_df = None
for file_path in json_files:
    data = load_json(file_path)
    df = chunk_json_data_to_df(data)
    if combined_df is None:
      combined_df = df
    else: 
      combined_df = combined_df.unionByName(
        df, allowMissingColumns=False
        )

# COMMAND ----------

display(combined_df)

# COMMAND ----------

(combined_df.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("shm.default.chunked_docs")
)
