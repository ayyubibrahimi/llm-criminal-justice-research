import re
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
import logging
import time
from summarizer import Summarizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

f_path = r"../../data/convictions/transcripts/evaluate"

doc_directory = "../../data/convictions/transcripts"

PROMPT_TEMPLATE_MODEL = PromptTemplate(
    input_variables=["question", "docs"],
    template="""
    As an AI assistant, my role is to meticulously analyze court transcripts and extract information about law enforcement personnel.
    The names of law enforcement personnel will be prefixed by one of the following titles: officer, detective, deputy, lieutenant, 
    sergeant, captain, officer, coroner, investigator, criminalist, patrolman, or technician.

    Query: {question}

    Transcripts: {docs}

    The response will contain:

    1) The name of a officer, detective, deputy, lieutenant, 
       sergeant, captain, officer, coroner, investigator, criminalist, patrolman, or technician - 
       if an individual's name is not associated with one of these titles they do not work in law enforcement.
       Please prefix the name with "Officer Name: ". 
       For example, "Officer Name: John Smith".

    2) If available, provide an in-depth description of the context of their mention. 
       If the context induces ambiguity regarding the individual's employment in law enforcement, 
       remove the individual.
       Please prefix this information with "Officer Context: ". 

    Continue this pattern of identifying persons, until all law enforcement personnel are identified.  

    Additional guidelines for the AI assistant:
    - Titles may be abbreviated to the following Sgt., Cpl, Cpt, Det., Ofc., Lt., P.O. and P/O
    - Titles "Technician" and "Tech" might be used interchangeably.
    - Derive responses from factual information found within the police reports.
    - If the context of an identified person's mention is not clear in the report, provide their name and note that the context is not specified.
    - Do not extract information about victims and witnesses
""",
)


PROMPT_TEMPLATE_HYDE = PromptTemplate(
    input_variables=["question"],
    template="""
    You're an AI assistant specializing in criminal justice research. 
    Your main focus is on identifying the names and providing detailed context of mention for each law enforcement personnel. 
    This includes police officers, detectives, deupties, lieutenants, sergeants, captains, technicians, coroners, investigators, patrolman, and criminalists, 
    as described in court transcripts.
    Be aware that the titles "Detective" and "Officer" might be used interchangeably.
    Be aware that the titles "Technician" and "Tech" might be used interchangeably.

    Question: {question}

    Roles and Responses:""",
)


def generate_hypothetical_embeddings():
    llm = OpenAI()
    prompt = PROMPT_TEMPLATE_HYDE

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    base_embeddings = OpenAIEmbeddings()

    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain, base_embeddings=base_embeddings
    )
    return embeddings


def preprocess_single_document(file_path, embeddings, chunk_size, chunk_overlap):
    logger.info(f"Processing Word document: {file_path}")

    loader = Docx2txtLoader(file_path)
    text = loader.load()
    logger.info(f"Text loaded from Word document: {file_path}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(text)

    db = FAISS.from_documents(docs, embeddings)

    return db


def get_response_from_query(db, query, k):
    time.sleep(0.01)
    logger.info("Performing query...")
    docs = db.similarity_search(query, k=int(round(k)))  # Ensure k is an integer
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model_name="gpt-4")

    prompt = PROMPT_TEMPLATE_MODEL

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content, temperature=0)

    formatted_response = ""
    officers = response.split("Officer Name:")
    for i, officer in enumerate(officers):
        if officer.strip() != "":
            formatted_response += f"Officer Name {i}:{officer.replace('Officer Context:', 'Officer Context ' + str(i) + ':')}\n\n"

    officer_data = extract_officer_data(formatted_response)

    # Calculate the total token count in the 'Officer Context' field
    token_count = sum(len(item["Officer Context"].split()) for item in officer_data)

    return token_count



def summarize_context(context):
    model = Summarizer()
    result = model(context, min_length=60)
    summary = "".join(result)
    return summary


def generate_hypothetical_embeddings():
    llm = OpenAI()
    prompt = PROMPT_TEMPLATE_HYDE

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    base_embeddings = OpenAIEmbeddings()

    embeddings = HypotheticalDocumentEmbedder(
        llm_chain=llm_chain, base_embeddings=base_embeddings
    )
    return embeddings
