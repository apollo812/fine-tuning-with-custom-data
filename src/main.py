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
from chatting.ChattingClass import ChattingClass

from models.finetune_model import FineTuneModel
from models.chatting_model import ChattingModel

# Create a FastAPI application
app = FastAPI(swagger_ui_parameters={"tryItOutEnabled": True})


# Define a route to handle the root endpoint and redirect to the API documentation
@app.get("/")
async def root():
    return RedirectResponse(app.docs_url)


@app.post("/finetune", status_code=HTTP_201_CREATED)
async def finetune(req_body: FineTuneModel):
    fine_tune = FineTuningClass(api_key=req_body.api_key, data_path=req_body.data_path, model=req_body.model, temperature=req_body.temperature, max_retries=req_body.max_retries)
    fine_tune.train_generation()
    fine_tune.jsonl_generation()
    model_id = fine_tune.finetune()
    return {"model_id": model_id}


@app.post("/chatting", status_code=HTTP_201_CREATED)
async def chatting(req_body: ChattingModel):
    chatbot = ChattingClass(model_id=req_body.model, data_path=req_body.data_path, api_key=req_body.api_key, temperature=req_body.temperature)
    response = chatbot.ask_question(req_body.question)
    print(response)
    return {"answer": response}