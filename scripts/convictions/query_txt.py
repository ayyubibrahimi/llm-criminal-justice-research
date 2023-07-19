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
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.llms import OpenAI

load_dotenv(find_dotenv())
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

query_memory = []


def generate_hyde():
    llm = OpenAI()
    prompt_template = """
    You're a criminal justice researcher focusing on the names and context of mention of law enforcement personnel, including police officers, detectives, homicide units, and crime lab personnel, as described in trial transcripts. Be aware that the titles "Detective" and "Officer" might be used interchangeably.

    Question: {question}

    Roles and Responses:"""
    prompt = PromptTemplate(input_variables=["question"], template=prompt_template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    base_embeddings = OpenAIEmbeddings()

    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain, base_embeddings=base_embeddings
    )
    return embeddings


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


def create_db_from_text_files(text_directory, embeddings) -> FAISS:
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
                doc = Document(page_content=doc_content, metadata={"source": file_path})
                all_docs.append(doc)

    db = FAISS.from_documents(all_docs, embeddings)

    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    faiss_index_path = os.path.join(
        cache_dir, f"faiss_index_{os.path.basename(file_path)}"
    )
    db.save_local(faiss_index_path)

    logger.info("Combined database created from all text files")
    return db, faiss_index_path


def get_response_from_query(db, query, k=3):
    logger.info("Performing query...")
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        As an AI assistant, my task is to extract the names and context of mention for each law enforcement individual, including police officers, detectives, homicide units, and crime lab personnel named in the trial transcript. Be aware that the titles "Detective" and "Officer" might be used interchangeably:

        Query: {question}

        Trial Transcript: {docs}

        Here's what you can expect in the response:

        1. The name of a law enforcement personnel, including a police officer, police witnesses, detective, homicide deputies, lieutenant, sergeant, captain, crime lab personnel, and homicide officers. Please prefix the name with "Officer Name: ". 
        2. Provide the context or the reason for the identified person. Please prefix the name with "Officer Context: ". 
        3. Continue this pattern, for each officer and each context, until all law enforcement personnel are identified.

        Guidelines for the AI assistant:

        - Derive responses from factual information found within the transcript.
        - If the context of an identified person's mention is not clear in the transcript, provide their name and note that the context is not specified.
        - If there is insufficient information to answer the query, simply respond with "Insufficient information to answer the query".
    """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace(" ", "\n")
    return response, docs


queries = [
    "Enumerate all law enforcement personnel, including police officers, detectives, homicide officers, and crime lab personnel from the transcript and provide the context of their mention, if available.",
    "Can you list all individuals related to law enforcement such as police officers, detectives, homicide units, and crime lab personnel mentioned in the transcript and elaborate on the context of their mention?",
    "Please produce a roster of all persons involved with law enforcement, including police officers, detectives, homicide units, crime lab personnel from the transcript and explain why they are mentioned, if stated.",
    "Identify all the law enforcement entities, notably police officers, detectives, homicide units, crime lab personnel, stated in the transcript and describe the reason for their mention, if specified.",
    "Could you outline all individuals from law enforcement, especially police officers, detectives, homicide units, crime lab personnel, referenced in the transcript and their context of mention, if defined?",
    "Please pinpoint all law enforcement associates, mainly police officers, detectives, homicide units, crime lab personnel, cited in the transcript and specify their mention context, if outlined.",
]


def answer_query_for_each_doc(embeddings) -> None:
    doc_directory = (
        r"../../data/convictions/v1/testimony" 
    )

    for file_name in os.listdir(doc_directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(doc_directory, file_name)
            output_data = {}

            faiss_index_path = "cache/faiss_index_" + file_name
            if os.path.exists(faiss_index_path):
                db = FAISS.load_local(faiss_index_path, embeddings)
                logger.info(f"Loaded database from {faiss_index_path}")
            else:
                db, faiss_index_path = create_db_from_text_files(
                    doc_directory, embeddings
                )  

            for query in queries:
                response, _ = get_response_from_query(db, query)
                output_data[query] = response

                print("Bot response for query: ", query)
                print(textwrap.fill(response, width=85))
                print()


def main():
    embeddings = generate_hyde()
    answer_query_for_each_doc(embeddings)


if __name__ == "__main__":
    main()
