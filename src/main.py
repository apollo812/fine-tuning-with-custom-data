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

from PIL import Image
import json

from utils.pdf2img import pdf2img
from utils.construct_index import construct_index

# Create a FastAPI application
app = FastAPI(swagger_ui_parameters={"tryItOutEnabled": True})


# Define a route to handle the root endpoint and redirect to the API documentation
@app.get("/")
async def root():
    return RedirectResponse(app.docs_url)

# @app.post("/pdf2img", status_code=HTTP_201_CREATED)
# async def p2i(pdf_name: str):
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     file_path  = os.path.join(current_dir, "pdfs", f'{pdf_name}.pdf')
#     r = pdf2img(file_path, current_dir, pdf_name)

#     print("here", r)
#     return Response(content=r)

# @app.get("/construct", status_code=HTTP_201_CREATED)
# async def cindex(file_name: str):
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     # file_path  = os.path.join(current_dir, "pdfs", f'{pdf_name}.pdf')
#     folder_path  = os.path.join(current_dir, "pdfs")
#     r = construct_index(folder_path)
