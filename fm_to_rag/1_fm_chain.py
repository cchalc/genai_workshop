import mlflow
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain_core.output_parsers import StrOutputParser

## Enable MLflow Tracing
mlflow.langchain.autolog()

## Load the chain's configuration from a file
model_config = mlflow.models.ModelConfig(development_config="fm_chain_config.yaml")

prompt = ChatPromptTemplate.from_messages(
    [  
        ("system", model_config.get("fm_prompt_template")),
        ("user", "{question}")
    ]
)

# Our foundation model answering the final prompt
model = ChatDatabricks(
    endpoint=model_config.get("fm_serving_endpoint_name"),
    extra_params={
      "temperature": model_config.get("temperature"), 
      "max_tokens": model_config.get("max_tokens")
      }
)

# Set the chain
chain = (
    prompt
    | model
    | StrOutputParser()
)

# Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=chain)