import io
import os
import sys

# Get the current script's directory
current_script_directory = os.path.dirname(os.path.abspath(__file__))

# Get the project root path
project_root = os.path.abspath(os.path.join(current_script_directory, os.pardir))

# Append the project root and current script directory to the system path
sys.path.append(project_root)
sys.path.append(current_script_directory)

from fastapi import Depends, FastAPI, File, Response, UploadFile
from starlette.responses import RedirectResponse
from starlette.status import HTTP_201_CREATED

from finetune.FineTuningClass import FineTuningClass

# Create a FastAPI application
app = FastAPI(swagger_ui_parameters={"tryItOutEnabled": True})


# Define a route to handle the root endpoint and redirect to the API documentation
@app.get("/")
async def root():
    return RedirectResponse(app.docs_url)


@app.get("/finetune", status_code=HTTP_201_CREATED)
async def finetune(api_key: str, data_path: str, model='gpt-3.5-turbo', temperature=0.3, max_retries=5):
    fine_tune = FineTuningClass(api_key=api_key, data_path=data_path, model=model, temperature=temperature, max_retries=max_retries)
    fine_tune.train_generation()
    fine_tune.jsonl_generation()
    fine_tune.finetune()
