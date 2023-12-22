import os
import gc
import argparse
from pathlib import Path

from src.utils.read_json import read_json
from src.chatting.ChattingClass import ChattingClass

def chatting(args):
    """
    main entry point
    """

    # Payload
    payload_data = read_json(args.payload_dir)

    # Call class instance
    chatting = ChattingClass(
        data_path=payload_data["data_path"],
        api_key=payload_data["api_key"],
        model_id=payload_data["model_id"],
        temperature=payload_data["temperature"])
        
    response = chatting.ask_question(args.question)
    print(response)

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
    payload_name = "chatting_payload.json"
    payload_dir  = os.path.join(current_dir, "test", "regression", test_name, "payload", payload_name)

    # Add options
    p = argparse.ArgumentParser()
    p = argparse.ArgumentParser(description="Conversational Agent.")
    p.add_argument("--payload_dir", type=Path, default=payload_dir, help="payload directory to the test example")
    p.add_argument("--question", type=str)
    args = p.parse_args()

    chatting(args)
