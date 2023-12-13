# img2txt-fine-tuning-api
Image to text translator using Open AI vision API


Before installing the project, ensure you have Python 3.x installed on your system.

### Clone the Repository
```bash
git clone https://github.com/oridosai/img2txt-fine-tuning-api.git
cd img2txt-fine-tuning-api
```

### Install Dependencies
```bash
pip install . 
```

## Usage

To run the application, use the following command from the project directory:
```bash
python3 main.py --payload_dir <your_payload_dir>
```

## Modules

- **Image Translator**: Handles image processing and image translation tasks (convert image to text)
- **ChatGPT Communicator**: Manages communication with ChatGPT-like services and pass the output of Image Translator to revise the output or ask any specific questions (this is optional)

## Manual

```
py -m venv venv

Window:
venv\Scripts\Activate

MacOS:
venv\bin\activate

pip install -r requirements.txt

uvicorn src.main:app --reload
```

- http://localhost:8000

## Using Docker

### Build
```make build```

### Test
```make test```

### Run
```make up``` 

- http://localhost:5000