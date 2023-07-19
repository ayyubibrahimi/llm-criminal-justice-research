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
import re
import pandas as pd
import logging
from summarizer import Summarizer


def summarize_context(context):
    model = Summarizer()
    result = model(context, min_length=60)
    summary = "".join(result)
    return summary


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())
query_memory = []


def generate_hyde():
    llm = OpenAI()
    prompt_template = """
    You're an AI assistant specializing in criminal justice research. 
    Your main focus is on identifying the names and providing detailed context of mention for each law enforcement personnel. 
    This includes police officers, detectives, deupties, lieutenants, sergeants, captains, technicians, and district attorneys, 
    as described in court transcripts.
    Be aware that the titles "Detective" and "Officer" might be used interchangeably.
    Be aware that the titles "Technician" and "Tech" might be used interchangeably.

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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1500)
    docs = text_splitter.split_documents(text)

    db = FAISS.from_documents(docs, embeddings)

    return db


def clean_name(officer_name):
    return re.sub(
        r"(Detective|Officer|Deputy|Captain|[CcPpLl]|Sergeant|Lieutenant|Techn?i?c?i?a?n?)\.?\s+",
        "",
        officer_name,
    )


def extract_officer_data(formatted_response):
    officer_data = []
    response_lines = formatted_response.split("\n")

    for line in response_lines:
        if line.startswith("Officer Name"):
            officer_name = line.split(":", 1)[1].strip()
            officer_title = re.search(
                r"(Detective|Officer|Deputy|Captain|[CcPpLl]|Sergeant|Lieutenant|Techn?i?c?i?a?n?)\.?",
                officer_name,
            )
            if officer_title:
                officer_title = officer_title.group()
            else:
                officer_title = ""
            officer_name = clean_name(officer_name)
        elif line.startswith("Officer Context"):
            split_line = line.split(":", 1)
            if len(split_line) > 1:
                officer_context = split_line[1].strip()
            else:
                officer_context = ""  
            officer_data.append(
                {
                    "Officer Name": officer_name,
                    "Officer Context": officer_context,
                    "Officer Title": officer_title,
                }
            )

    return officer_data


def get_response_from_query(db, query, k=3):
    logger.info("Performing query...")
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613")

    ### add investigator, ex: "Orleans Parish Coroner's Office Investigator Purnell Lewis" - 05 NOPD Supplemental Report 
    ### add crime lab! technician
    ### need to add abbreviations to the prompt such as cpl., sgt., off.

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        As an AI assistant, my role is to meticulously identify the names and provide a detailed explanation of the situations and interactions 
        in which each law enforcement personnel was mentioned in the court transcripts. 

        Query: {question}

        Court Transcripts: {docs}

        The response will contain:

        1) The name of a law enforcement personnel. Law enforcement names will include be prefixed with the titles officer,
           detective, deupty, lieutenant, sergeant, captain, or technician. Only identify law enforcement personnel.
           Please prefix the name with "Officer Name: ". 
        
        2)  A detailed explanation of the situation and interactions involving the identified personnel and the context of their mention.
            Please prefix this information with "Officer Context: ". 

        Continue this pattern, for each officer and each context, until all law enforcement personnel are identified. 

        Guidelines for the AI assistant:

        - Derive responses from factual information found within the transcript.
        - If the context of an identified person's mention is not clear in the transcript, provide their name and note that the context is not specified.
        - If there is insufficient information to answer the query, simply respond with "Insufficient information to answer the query".
    """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content, temperature=0)

    formatted_response = ""
    officers = response.split("Officer Name:")
    for i, officer in enumerate(officers):
        if officer.strip() != "":
            formatted_response += f"Officer Name {i}:{officer.replace('Officer Context:', 'Officer Context ' + str(i) + ':')}\n\n"

    officer_data = extract_officer_data(formatted_response)
    return officer_data, docs


queries = [
    "Enumerate all law enforcement personnel, including police officers, sergeants, lieutentants, captains, detectives, homicide officers, crime lab personnel, and district attorneys from the transcript and provide the context of their mention, if available.",
    "Can you list all individuals related to law enforcement such as police officers, sergeants, lieutentants, captains, detectives, homicide units, crime lab personnel, and district attorneys mentioned in the transcript and elaborate on the context of their mention?",
    "Please produce a roster of all persons involved with law enforcement, including police officers,  sergeants, lieutentants, captains, detectives, homicide units, crime lab personnel, and district attorneys from the transcript and explain why they are mentioned, if stated.",
    "Identify all the law enforcement entities, notably police officers, sergeants, lieutentants, captains,  detectives, homicide units, crime lab personnel, and district attorneys stated in the transcript and describe the reason for their mention, if specified.",
    "Could you outline all individuals from law enforcement, especially police officers, sergeants, lieutentants, captains, detectives, homicide units, crime lab personnel, and district attorneys referenced in the transcript and their context of mention, if defined?",
    "Please pinpoint all law enforcement associates, mainly police officers, sergeants, lieutentants, captains, detectives, homicide units, crime lab personnel, and district attorneys cited in the transcript and specify their mention context, if outlined.",
]


def answer_query_for_each_doc(embeddings):
    doc_directory = "../../data/convictions/evaluate/reports"

    for file_name in os.listdir(doc_directory):
        if file_name.endswith(".docx"):
            csv_output_path = os.path.join(doc_directory, f"{file_name}.csv")
            if os.path.exists(csv_output_path):
                logger.info(f"CSV output for {file_name} already exists. Skipping...")
                continue

            file_path = os.path.join(doc_directory, file_name)
            output_data = []

            db = process_single_document(file_path, embeddings)

            for query in queries:
                officer_data, _ = get_response_from_query(db, query)
                for item in officer_data:
                    item["Query"] = query
                output_data.extend(officer_data)

                print("Bot response for query: ", query)
                print(textwrap.fill(str(officer_data), width=85))
                print()


            output_df = pd.DataFrame(output_data)
            officer_title_df = output_df[
                ["Officer Name", "Officer Title"]
            ].drop_duplicates()
            output_df = (
                output_df.groupby("Officer Name")["Officer Context"]
                .apply("; ".join)
                .reset_index()
            )
            output_df["Officer Context"] = output_df["Officer Context"].apply(
                summarize_context
            )
            output_df = pd.merge(
                output_df, officer_title_df, on="Officer Name", how="outer"
            )
            output_df.to_csv(csv_output_path, index=False)


def main():
    embeddings = generate_hyde()
    answer_query_for_each_doc(embeddings)


if __name__ == "__main__":
    main()
