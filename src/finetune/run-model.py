import openai
import os
from dotenv import load_dotenv
from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms import OpenAI
import argparse

# Set your API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    openai.api_key = openai_api_key
else:
    os.environ["OPENAI_API_KEY"] = "your_default_api_key"
    openai.api_key = "openai_api_key"

data_path = "./src/test/regression/regression_test003"

model_id = "ft:gpt-3.5-turbo-0613:personal::8XCvxg1X"

documents = SimpleDirectoryReader(data_path).load_data()

ft_context = ServiceContext.from_defaults(
    llm=OpenAI(model=model_id, temperature=0.3),
    context_window=2048
)

index = VectorStoreIndex.from_documents(documents, service_context=ft_context)
query_engine = index.as_query_engine(service_context=ft_context)

# Using argparse to get the question input from the user
parser = argparse.ArgumentParser(description='Query the index with a question')
parser.add_argument('question', type=str, help='The question to query the index')
args = parser.parse_args()

response = query_engine.query(args.question)
print(response)