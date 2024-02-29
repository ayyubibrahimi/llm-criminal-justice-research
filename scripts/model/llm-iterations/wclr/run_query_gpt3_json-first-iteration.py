import os
from langchain.document_loaders import Docx2txtLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import logging
from helper import generate_hypothetical_embeddings, PROMPT_TEMPLATE_HYDE, sort_retrived_documents
import re
from langchain.chat_models import AzureChatOpenAI

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv())
query_memory = []

CHUNK_SIZE = 500
CHUNK_OVERLAP = 250
TEMPERATURE = 1
k = 20

input_path_transcripts = r"../../data/wrongful-convictions/json/input/transcripts"
input_path_reports = r"../../data/wrongful-convictions/json/input/reports"

output_path_transcripts = r"../../data/wrongful-convictions/json/output-4-og-params/transcripts"
output_path_reports = r"../../data/wrongful-convictions/json/output-4-og-params/reports"


def clean_name(officer_name):
    return re.sub(
        r"(Detective|Officer|Deputy|Captain|[CcPpLl]|Sergeant|Lieutenant|Techn?i?c?i?a?n?|Investigator|^-|\d{1}\)|\w{1}\.)\.?\s+",
        "",
        officer_name,
    )

def extract_officer_data(text):
    officer_data = []

    normalized_text = re.sub(r"\s*-\s*", "", text)

    officer_sections = re.split(r"\n(?=Officer Name:)", normalized_text)

    for section in officer_sections:
        if not section.strip():
            continue

        officer_dict = {}

        name_match = re.search(
            r"Officer Name:\s*(.*?)\s*Officer Context:", section, re.DOTALL
        )
        context_match = re.search(
            r"Officer Context:\s*(.*?)\s*Officer Role:", section, re.DOTALL
        )
        role_match = re.search(r"Officer Role:\s*(.*)", section, re.DOTALL)

        if name_match and name_match.group(1):
            officer_dict["Officer Name"] = clean_name(name_match.group(1).strip())
        if context_match and context_match.group(1):
            officer_dict["Officer Context"] = context_match.group(1).strip()
        if role_match and role_match.group(1):
            officer_dict["Officer Role"] = role_match.group(1).strip()

        if officer_dict:
            officer_data.append(officer_dict)

    return officer_data


PROMPT_TEMPLATE_MODEL = PromptTemplate(
    input_variables=["question", "docs"],
    template="""
    As an AI assistant, my role is to meticulously analyze criminal justice documents and extract information about law enforcement personnel.
  
    Query: {question}

    Documents: {docs}

    The response will contain:

    1) The name of a law enforcement personnel. The individual's name must be prefixed with one of the following titles to be in law enforcement: 
       Detective, Sergeant, Lieutenant, Captain, Deputy, Officer, Patrol Officer, Criminalist, Technician, Coroner, or Dr. 
       Please prefix the name with "Officer Name: ". 
       For example, "Officer Name: John Smith".

    2) If available, provide an in-depth description of the context of their mention. 
       If the context induces ambiguity regarding the individual's employment in law enforcement, please make this clear in your response.
       Please prefix this information with "Officer Context: ". 

    3) Review the context to discern the role of the officer. For example, Lead Detective (Homicide Division), Supervising Officer (Crime Lab), Detective, Officer on Scene, Arresting Officer, Crime Lab Analyst
       Please prefix this information with "Officer Role: "
       For example, "Officer Role: Lead Detective"

    
    The full response should follow the format below, with no prefixes such as 1., 2., 3., a., b., c., etc.:

    Officer Name: John Smith 
    Officer Context: Mentioned as someone who was present during a search, along with other detectives from different units.
    Officer Role: Patrol Officer

    Officer Name: 
    Officer Context:
    Officer Role: 

    - Do not include any prefixes
    - Only derive responses from factual information found within the police reports.
    - If the context of an identified person's mention is not clear in the report, provide their name and note that the context is not specified.
    - Do not extract information about victims and witnesses

    
""",
)


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["page_number"] = record.get("page_number")
    return metadata

def preprocess_document(file_path, embeddings):
    logger.info(f"Processing Word document: {file_path}")

    loader = JSONLoader(file_path, jq_schema=".messages[]", content_key="page_content", metadata_func=metadata_func)
    text = loader.load()
    logger.info(f"Text loaded from Word document: {file_path}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = text_splitter.split_documents(text)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, temperature, k):
    logger.info("Performing query...")

    doc_list = db.similarity_search_with_score(query, k=k)

    docs = sort_retrived_documents(doc_list)

    docs_page_content = " ".join([d[0].page_content for d in docs])
    print(docs_page_content)

    # Initialize an empty list to store page numbers
    page_numbers = []

    # Loop through docs and extract page numbers, appending them to the list
    for doc in docs:
        page_number = doc[0].metadata.get('page_number')
        if page_number is not None:
            page_numbers.append(page_number)

    # tuned_model = "ft:gpt-3.5-turbo-0613:hrdag::86QJloMs"
    # tuned_model = "ft:gpt-3.5-turbo-0613:hrdag::85G6LuP7"
    
    # fine-tuned 3.5-turbo-1106-
    # tuned_model = "ft:gpt-3.5-turbo-1106:personal::8TMxIQHR"

    ## done

    ## review non-fine tuned 164k, 300 and 1603-16k
    
    # fine-tuned 3.5-turbo-4k-0613-300-examples
    # ft:gpt-3.5-turbo-0613:eye-on-surveillance::8U1Swiff


    # fine-tuned 3.5-turbo-4k-0613-200-examples
    # ft:gpt-3.5-turbo-0613:personal::8TaO3ozh

    # fine-tuned 3.5-turbo-4k-0613-100-examples4
    # ft:gpt-3.5-turbo-0613:personal::8TcTi7tA

    # fine-tuned 3.5-turbo-4k-0613-50-examples
    # ft:gpt-3.5-turbo-0613:personal::8TbyiSUv

    # fine-tuned 3.5-turbo-4k-0613-25-examples
    # ft:gpt-3.5-turbo-0613:personal::8TfLWTfM 
    
    # non-finetuned 3.5-turbo-16k-0613
    # gpt-3.5-turbo-16k-0613

    # non-finetuned 3.5-turbo-4k-0613
    # gpt-3.5-turbo-0613

    ## left 

    
    llm = ChatOpenAI(model_name="gpt-4")

    prompt = PROMPT_TEMPLATE_MODEL

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content, temperature=temperature)
    print(response)

    return response, page_numbers


iteration_times = 6
max_retries = 10

QUERIES = [
"Identify each individual in the transcript, by name, who are directly referred to as officers, sergeants, lieutenants, captains, detectives, homicide officers, and crime lab personnel. Provide the context of their mention, focusing on key events, significant decisions or actions they made, interactions with other individuals, roles or responsibilities they held, noteworthy outcomes or results they achieved, and any significant incidents or episodes they were involved in, if available."
]

MULTIPLE_QUERIES = [
    "Identify individuals, by name, with the specific titles of officers, sergeants, lieutenants, captains, detectives, homicide officers, and crime lab personnel in the transcript. Specifically, provide the context of their mention related to key events in the case, if available.",
    "List individuals, by name, directly titled as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel mentioned in the transcript. Provide the context of their mention in terms of any significant decisions they made or actions they took.",
    "Locate individuals, by name, directly referred to as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Explain the context of their mention in relation to their interactions with other individuals in the case.",
    "Highlight individuals, by name, directly titled as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Describe the context of their mention, specifically noting any roles or responsibilities they held in the case.",
    "Outline individuals, by name, directly identified as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Specify the context of their mention in terms of any noteworthy outcomes or results they achieved.",
    "Pinpoint individuals, by name, directly labeled as officers, sergeants, lieutenants, captains, detectives, homicide units, and crime lab personnel in the transcript. Provide the context of their mention, particularly emphasizing any significant incidents or episodes they were involved in.",
]

def process_files_in_directory(input_path, output_path, embeddings, multiple_queries_mode=False):
    queries = MULTIPLE_QUERIES if multiple_queries_mode else QUERIES * iteration_times
    queries_label = "_six_queries" if multiple_queries_mode else "_one_query"

    for file_name in os.listdir(input_path):
        if file_name.endswith(".json"):
            csv_output_path = os.path.join(output_path, f"{file_name}{queries_label}.csv")
            if os.path.exists(csv_output_path):
                logger.info(f"CSV output for {file_name} already exists. Skipping...")
                continue

            file_path = os.path.join(input_path, file_name)
            output_data = []

            db = preprocess_document(file_path, embeddings)
            for idx, query in enumerate(queries, start=1):
                retries = 0
                while retries < max_retries:
                    try:
                        officer_data_string, page_numbers = get_response_from_query(db, query, TEMPERATURE, k)
                        break
                    except ValueError as e:
                        if "Azure has not provided the response" in str(e):
                            retries += 1
                            logger.warn(f"Retry {retries} for query {query} due to Azure content filter error.")
                        else:
                            raise

                if retries == max_retries:
                    logger.error(f"Max retries reached for query {query}. Skipping...")
                    continue

                officer_data = extract_officer_data(officer_data_string)

                for item in officer_data:
                    item["page_number"] = page_numbers
                    item["fn"] = file_name
                    item["Query"] = query
                    item["Prompt Template for Hyde"] = PROMPT_TEMPLATE_HYDE
                    item["Prompt Template for Model"] = PROMPT_TEMPLATE_MODEL
                    item["Chunk Size"] = CHUNK_SIZE
                    item["Chunk Overlap"] = CHUNK_OVERLAP
                    item["Temperature"] = TEMPERATURE
                    item["k"] = k
                    item["hyde"] = "1"
                    item["iteration"] = idx
                    item["num_of_queries"] = "6" if multiple_queries_mode else "1"
                    item["model"]  = "gpt-3.5-turbo-finetuned"
                output_data.extend(officer_data)

            output_df = pd.DataFrame(output_data)
            output_df.to_csv(csv_output_path, index=False)


def process_query():
    embeddings = generate_hypothetical_embeddings()
    
    process_files_in_directory(input_path_transcripts, output_path_transcripts, embeddings, False)
    process_files_in_directory(input_path_reports, output_path_reports, embeddings, False)

    process_files_in_directory(input_path_transcripts, output_path_transcripts, embeddings, True)
    process_files_in_directory(input_path_reports, output_path_reports, embeddings, True)

if __name__ == "__main__":
    process_query()