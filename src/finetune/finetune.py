import os
from dotenv import load_dotenv
import openai
import time

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

data_path = "./src/test/regression/regression_test003"

file_upload = openai.files.create(file=open(f'{data_path}/generated_data/finetuning_events.jsonl', "rb"), purpose="fine-tune")
print("Uploaded file id", file_upload.id)

while True:
    print("Waiting for file to process...")
    file_handle = openai.files.retrieve(file_id=file_upload.id)
    if file_handle and file_handle.status == "processed":
        print("File processed")
        break
    time.sleep(3)


try:
    job = openai.fine_tuning.jobs.create(training_file=file_upload.id, model="gpt-3.5-turbo")

    while True:
        print("Waiting for fine-tuning to complete...")
        job_handle = openai.fine_tuning.jobs.retrieve(fine_tuning_job_id=job.id)
        if job_handle.status == "succeeded":
            print("Fine-tuning complete")
            print("Fine-tuned model info", job_handle)
            print("Model id", job_handle.fine_tuned_model)
            break
        time.sleep(3)
except Exception as e:
    print(f"An error occurred during fine-tuning: {e}")
