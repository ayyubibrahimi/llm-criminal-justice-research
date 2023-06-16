import logging
import datetime
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from dotenv import find_dotenv, load_dotenv
import textwrap
import csv

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

query_memory = []

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def create_db_from_text_files(text_directory) -> FAISS:
    all_docs = []
    for file_name in os.listdir(text_directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(text_directory, file_name)
            logger.info(f"Processing text file: {file_path}")

            with open(file_path, "r") as file:
                lines = file.readlines()

            logger.info(f"Text loaded from text file: {file_path}")

            for line in lines:
                doc_content = line.strip()
                doc = Document(page_content=doc_content, metadata={'source': file_path})
                all_docs.append(doc)

    db = FAISS.from_documents(all_docs, embeddings)

    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    db.save_local(os.path.join(cache_dir, "faiss_index"))
    logger.info("Combined database created from all text files")
    return db


def get_response_from_query(db, query, k=3):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """
    logger.info("Performing query...")
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        As an AI assistant, my task is to provide the names of each law enforcement individual named in the trial transcript:

        Query: {question}

        Trial Transcript: {docs}

        Here's what you can expect in the response:

        1. The first and/or last of individuals who have the title "Officer", "Detective", "Det.", "Homicide Detective", "Sergeant", or "Captain". These are all police officers.
        2. Name each police officer 
        3. Describe the role of each police officer

        Guidelines for the AI assistant:

        - Only derive responses from factual information found within the transcript.
        - Provide as much detail as the transcript allows.
        - If there is insufficient information to answer the query, simply respond with "Insufficient information to answer the query".
    """,
    )


    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs


def answer_query(query: str, embeddings) -> str:
    text_directory = "../../data/convictions"  # Update the directory path accordingly

    faiss_index_path = "cache/faiss_index"
    if os.path.exists(faiss_index_path):
        db = FAISS.load_local(faiss_index_path, embeddings)
        logger.info("Loaded database from faiss_index")
    else:
        db = create_db_from_text_files(text_directory)

    response, _ = get_response_from_query(db, query)

    # Save the data to a CSV file
    with open("../../data/convictions/output_txt.csv", mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["response"])
        writer.writerow([response])

    print("Bot response:")
    print(textwrap.fill(response, width=85))
    print()

    query_memory.append(query)
    return response


while True:
    query = input("Enter your query (or 'quit' to exit): ")
    if query == "quit":
        break

    response = answer_query(query, embeddings)  # Pass the 'embeddings' argument

print("Query memory:")
print(query_memory)
