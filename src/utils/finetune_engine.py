from llama_index.finetuning import OpenAIFinetuneEngine
import os
from dotenv import load_dotenv
import openai

# Set your API key

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key is not None:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    openai.api_key = openai_api_key
else:
    # Handle the absence of the environment variable
    # You might want to log an error, raise an exception, or provide a default value
    # For example, setting a default value
    os.environ["OPENAI_API_KEY"] = "your_default_api_key"
    openai.api_key = "openai_api_key"

data_path = "./src/test/regression/regression_test004"


finetune_engine = OpenAIFinetuneEngine(
    "gpt-3.5-turbo",
    f'{data_path}/finetuning_events.jsonl',
    # start_job_id="<start-job-id>"  # if you have an existing job, can specify id here
)

finetune_engine.finetune()

finetune_engine.get_current_job()

ft_llm = finetune_engine.get_finetuned_model(temperature=0.3)