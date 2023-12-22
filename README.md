## Manual

```
py -m venv venv

Window:
venv\Scripts\Activate

MacOS:
venv\bin\activate

pip install -r requirements.txt

```

## Step-by-step

### Fine Tuning with specific data
- Generate the new folder at src\test\regression\regression_testxxx
Copy and paste the files at here

- Fine Tuning
```
py .\finetuning.py --payload_dir="payload_dir" 

example: 
py .\finetuning.py --payload_dir="./test/regression/regression_testxxx/payload/finetuning_payload.json"  
```

- After fine tuning, you will get the model id, you can check the model_id at console or data_path/generated_data/model.txt
- Please check the "./test/regression/regression_testxxx/payload/chatting_payload.json"
After successfully complete the fine tuning, will update the model_id automatically.
If you want to update this, please change the model_id with your specific id.

### Conversational Agent
```
py .\chatting.py --payload_dir="payload_dir" --question="question"

example: 
py .\chatting.py --payload_dir="./test/regression/regression_testxxx/payload/chatting_payload.json" --question="what's the golf?"
```

You will see the result at console.