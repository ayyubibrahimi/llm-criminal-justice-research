import logging
import os
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
import textwrap
from langchain.llms import OpenAI
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
import csv
import re
import string
import pandas as pd


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())
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


def process_single_document(file_path, embeddings):
    logger.info(f"Processing Word document: {file_path}")

    loader = Docx2txtLoader(file_path)
    text = loader.load()
    logger.info(f"Text loaded from Word document: {file_path}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1000)
    docs = text_splitter.split_documents(text)
    print(docs)

    db = FAISS.from_documents(docs, embeddings)

    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    faiss_index_path = os.path.join(
        cache_dir, f"faiss_index_{os.path.basename(file_path)}"
    )
    db.save_local(faiss_index_path)
    logger.info(f"Database created for the Word document: {file_path}")
    return db, faiss_index_path


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
        As an AI assistant, my task is to extract the names and context of mention for each law enforcement individual, including police officers, detectives, homicide units, and crime lab personnel named in the trial transcript. Be aware that the titles "Detective" and "Officer" might be used interchangeably:

        Query: {question}

        Trial Transcript: {docs}

        Here's what you can expect in the response:

        1. The names of all law enforcement personnel, including police officers, police witnesses, detectives, homicide deputies, lieutenants, sergeants, captains, crime lab personnel, and homicide officers mentioned in the trial transcript.
        2. Provide the context or the reason each identified person is mentioned in the trial transcript.

        Guidelines for the AI assistant:

        - Derive responses from factual information found within the transcript.
        - If the context of an identified person's mention is not clear in the transcript, provide their name and note that the context is not specified.
        - If there is insufficient information to answer the query, simply respond with "Insufficient information to answer the query".
    """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
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
    doc_directory = "../../data/convictions/testimony"  # Update the directory path accordingly

    for file_name in os.listdir(doc_directory):
        if file_name.endswith(".docx"):
            file_path = os.path.join(doc_directory, file_name)
            output_data = {}

            faiss_index_path = "cache/faiss_index_" + file_name
            if os.path.exists(faiss_index_path):
                db = FAISS.load_local(faiss_index_path, embeddings)
                logger.info(f"Loaded database from {faiss_index_path}")
            else:
                db, faiss_index_path = process_single_document(file_path, embeddings)

            for query in queries:
                response, _ = get_response_from_query(db, query)
                output_data[query] = response

                print("Bot response for query: ", query)
                print(textwrap.fill(response, width=85))
                print()

            # Convert the data to a pandas DataFrame and then save it to a CSV file
            output_df = pd.DataFrame(output_data, index=[0])
            output_df.to_csv(
                os.path.join(doc_directory, f"output_{file_name}.csv"), index=False
            )


def main():
    embeddings = generate_hyde()
    answer_query_for_each_doc(embeddings)


if __name__ == "__main__":
    main()
