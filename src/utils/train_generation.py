from llama_index import SimpleDirectoryReader, ServiceContext
from llama_index.llms import OpenAI
from llama_index.evaluation import DatasetGenerator
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

data_path = "./src/test/regression/regression_test004"

documents = SimpleDirectoryReader(
    data_path
).load_data()

# Shuffle the documents
import random

random.seed(42)
random.shuffle(documents)

gpt_35_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-3.5-turbo", temperature=0.3)
)

question_gen_query = (
    "You are a Teacher/ Professor. Your task is to setup "
    "a quiz/examination. Using the provided context, formulate "
    "a single question that captures an important fact from the "
    "context. Restrict the question to the context information provided."
)

dataset_generator = DatasetGenerator.from_documents(
    documents[:50],
    question_gen_query=question_gen_query,
    service_context=gpt_35_context,
)

# NOTE: this may take some time. Go grab a coffee!
questions = dataset_generator.generate_questions_from_nodes(num=40)
print("Generated ", len(questions), " questions")

with open(f'{data_path}/train_questions.txt', "w") as f:
    for question in questions:
        f.write(question + "\n")


dataset_generator = DatasetGenerator.from_documents(
    documents[50:],  # since we generated ~1 question for 40 documents, we can skip the first 40
    question_gen_query=question_gen_query,
    service_context=gpt_35_context,
)
# NOTE: this may take some time. Go grab a coffee!
questions = dataset_generator.generate_questions_from_nodes(num=40)
print("Generated ", len(questions), " questions")


with open(f'{data_path}/eval_questions.txt', "w") as f:
    for question in questions:
        f.write(question + "\n")