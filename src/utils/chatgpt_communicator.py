
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()


class ChatGPTCommunicator:
    """
    A class to interface with the OpenAI GPT model for interactive chat.
    """
    def __init__(self, api_key=None, language_model="gpt-4"):
        """
        Initialize with the provided API key and model name.
        
        Args:
            api_key (str, optional): API key for OpenAI. Defaults to global API_KEY.
            language_model (str, optional): Language model name for OpenAI. Default is "gpt-4".
            choices=["gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo", "gpt-4", "gpt-4 turbo"]
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.language_model = language_model
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]


    def create_chat(self, initial_message):
        """
        Resets the chat and starts with a system message and an optional user message.
        
        Args:
            initial_message (str): Initial message from the user.
        """
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        if initial_message:
            self.messages.append({"role": "user", "content": initial_message})


    def add_message(self, role, content):
        """
        Adds a message to the chat.
        
        Args:
            role (str): Role of the message sender ("system", "user", or "assistant").
            content (str): Content of the message.
        """
        if not hasattr(self, 'messages'):
            raise ValueError("Chat not initialized. Call create_chat() first.")
            
        self.messages.append({"role": role, "content": content})


    def get_response(self):
        """
        Fetches a response from the GPT model based on the chat history.

        Returns:
            str: Response content from the model.
        """
        if not hasattr(self, 'messages'):
            raise ValueError("Chat not initialized. Call create_chat() first.")
            
        try:
            response = client.chat.completions.create(model=self.language_model,
                                                    messages=self.messages)
            # Directly accessing the content of the message from the response
            if response.choices and hasattr(response.choices[0].message, 'content'):
                return response.choices[0].message.content
            else:
                print("No valid message in the response. retrying...")
                return ""
        except Exception as e:
            raise Exception(f"Error fetching response from GPT: {e}")