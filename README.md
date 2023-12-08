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