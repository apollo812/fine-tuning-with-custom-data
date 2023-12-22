
from setuptools import setup, find_packages

setup(
    name="OpenAI Vision (Image to text) API Client",
    version="1.0",
    description="A client for interacting with OpenAI Vision (Image to text) API",
    author="OridosAI",
    author_email="oridos.production@gmail.com",
    packages=find_packages(),
    install_requires=[
        "openai",
        "requests",
        "python-dotenv"
    ],
)
