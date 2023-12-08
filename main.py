
import os
import io
import gc
import time
import shutil
import argparse
from pathlib import Path

from src.read_json import read_json
from src.image_translator import ImageTranslator
from src.chatgpt_communicator import ChatGPTCommunicator


def main(args):
    """
    main entry point
    """
    # Timer
    start_time = time.time()

    # Payload
    payload_data = read_json(args.payload_dir)
    
    # Read images from the image directory
    image_dir = os.path.dirname(args.payload_dir)
    image_list = [img for img in os.listdir(image_dir) if img.endswith(".png") or img.endswith(".jpeg") or img.endswith(".jpg")]

    # Call class instance
    img_translator = ImageTranslator(api_key=payload_data["api_key"])
    
    # Loop over number of images and append all images
    # NOTE: User can upload image and add image URLs or just upload image or just add image URLs
    images = []
    if (len(image_list) > 0) and (len(payload_data["image_url"]) > 0):
        for image in image_list:
            image_path = os.path.join(image_dir, image)
            # Encode image
            base64_image = img_translator.encode_image(image_path)
            images.append((base64_image, False, "auto"))
        for img_url in payload_data["image_url"]:
            images.append((img_url, True, "auto"))
    elif (len(image_list) > 0) and (len(payload_data["image_url"]) == 0):
        for image in image_list:
            image_path = os.path.join(image_dir, image)
            # Encode image
            base64_image = img_translator.encode_image(image_path)
            images.append((base64_image, False, "auto"))
    elif (len(image_list) == 0) and (len(payload_data["image_url"]) > 0):
        for img_url in payload_data["image_url"]:
            images.append((img_url, True, "auto"))

    # Get the response (using max retry if fails)
    img_translator_response = None
    max_retries = 5
    last_error = ""

    for attempt in range(max_retries):
        try:
            response = img_translator.analyze_images(images)

            if "choices" in response and response["choices"]:
                first_choice = response["choices"][0]
                if "message" in first_choice and "content" in first_choice["message"] and first_choice["message"]["content"]:
                    img_translator_response = first_choice["message"]["content"]
                    break  # Successful response, break out of the loop
                else:
                    last_error = "No valid content in the response."
            else:
                last_error = "The response structure is not as expected."

        except Exception as e:
            last_error = f"Attempt {attempt + 1} failed: {e}"

        if img_translator_response:
            break  # If a successful response is obtained, exit the loop

    if img_translator_response is None:
        raise Exception("Failed to get a valid response after " + str(max_retries) + " attempts. Last error: " + last_error)

    # NOTE: this part is optional to continue interaction with chatgpt
    if payload_data["continue_conversation"]:
        # Extract image to text response and asking ChatGPT to revise it
        updated_response = payload_data["backend_prompt"] + img_translator_response

        # Create chatGPT communicator
        chatgpt_communicator = ChatGPTCommunicator(api_key=payload_data["api_key"], language_model=payload_data["language_model"])

        # Start conversation with ChatGPT using the transcribed or translated text
        chatgpt_communicator.create_chat(updated_response)

        # Get conversation with ChatGPT
        max_retries = 3
        chatgpt_response = None

        for attempt in range(max_retries):
            try:
                chatgpt_response = chatgpt_communicator.get_response()
                # Check if the response is valid (not None and not empty)
                if chatgpt_response:
                    break  # Valid response, break out of the loop
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to get a valid response from ChatGPT after {max_retries} attempts. Last error: {e}")

        # Print response and use it somewhere else
        print(chatgpt_response)
    else:
        print(img_translator_response)
    
    # Write into log file
    end_time = time.time()
    msg = f"Total processing time: {end_time - start_time} seconds"
    print(msg)

    # Delete class objects and clean the buffer memory using the garbage collection
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
    test_name    = "regression_test001"
    payload_name = "payload.json"
    payload_dir  = os.path.join(current_dir, "test", test_name, payload_name)

    # Add options
    p = argparse.ArgumentParser()
    p = argparse.ArgumentParser(description="Translate text within an image.")
    p.add_argument("--payload_dir", type=Path, default=payload_dir, help="payload directory to the test example")
    args = p.parse_args()

    main(args)