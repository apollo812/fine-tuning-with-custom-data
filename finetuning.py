import os
import gc
import argparse
from pathlib import Path

from src.utils.read_json import read_json
from src.finetune.FineTuningClass import FineTuningClass

def run_fine_tuning(args):
    """
    main entry point
    """

    # Payload
    payload_data = read_json(args.payload_dir)
    
    # Call class instance
    fine_tune = FineTuningClass(
        data_path=payload_data["data_path"],
        api_key=payload_data["api_key"],
        model=payload_data["model"],
        temperature=payload_data["temperature"],
        max_retries=payload_data["max_retries"])
    
    # Generate the train and eval data
    fine_tune.train_generation()

    # Generate the jsonl
    fine_tune.jsonl_generation()

    # Fine tuning
    fine_tune.finetune()

    gc.collect()

if __name__ == "__main__":
    """
    Form command lines
    """
    # Clean up buffer memory
    gc.collect()

    # Current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Payload directory
    test_name    = "regression_test003"
    payload_name = "finetuning_payload.json"
    payload_dir  = os.path.join(current_dir, "test", "regression", test_name, "payload", payload_name)

    # Add options
    p = argparse.ArgumentParser()
    p = argparse.ArgumentParser(description="Fine tuning.")
    p.add_argument("--payload_dir", type=Path, default=payload_dir, help="payload directory to the test example")
    args = p.parse_args()

    run_fine_tuning(args)
