import os
import io
import json
import logging

from PyPDF2 import PdfFileReader, PdfFileWriter
import pandas as pd

from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.exceptions import HttpResponseError
import azure 

from docx import Document

doc_directory = "../../data/convictions/evaluate/reports"


def getcreds():
    with open("creds.txt", "r") as c:
        creds = c.readlines()
    return creds[0].strip(), creds[1].strip()


class DocClient:
    def __init__(self, endpoint, key):
        self.client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

    def close(self):
        self.client.close()

    def extract_content(self, analyze_result):
        contents = {
            "page": [],
            "content": [],
            "confidence": [],  # Assuming lines do not have a 'confidence' attribute
        }

        # Iterate over each page
        for page in analyze_result.pages:
            # Iterate over each line in the page
            for line in page.lines:
                contents["page"].append(page.page_number)
                contents["content"].append(line.content if line.content else None)
                # Add 'None' for 'confidence' as 'DocumentLine' does not seem to have a 'confidence' attribute
                contents["confidence"].append(None)

        return pd.DataFrame(contents)

    def pdf2df(self, pdf_path):
        with open(pdf_path, "rb") as file:
            pdf = PdfFileReader(file)
            total_pages = pdf.getNumPages()
            results = []

            # Update the range from 10 to 5.
            for i in range(0, total_pages, 10):  
                pdf_writer = PdfFileWriter()
                for j in range(i, min(i + 10, total_pages)):
                    pdf_writer.addPage(pdf.getPage(j))

                pdf_bytes = io.BytesIO()
                pdf_writer.write(pdf_bytes)
                pdf_bytes.seek(0)
                
                try:
                    poller = self.client.begin_analyze_document(
                        "prebuilt-document", document=pdf_bytes.read()
                    )
                    result = poller.result()

                    df_results = self.extract_content(result)
                    results.append(df_results)
                except azure.core.exceptions.HttpResponseError as e:
                    logging.error(f"Error processing file {pdf_path}: {e}")
                    # Optionally, add this file to a list to be retried later
                    # error_files.append(pdf_path)
                    continue

            # Combine all DataFrames into one
            combined_results = pd.concat(results)

            return combined_results

    def process(self, pdf_path):
        outname = os.path.basename(pdf_path).replace(".pdf", "")
        outstring = os.path.join(
            "../../data/convictions/evaluate/reports", "{}.docx".format(outname)
        )
        outpath = os.path.abspath(outstring)
        if os.path.exists(outpath):
            logging.info(f"skipping {outpath}, file already exists")
            return outpath

        logging.info(f"sending document {outname}")
        results = self.pdf2df(pdf_path)
        logging.info(f"writing to {outpath}")

        # Create a new Document
        doc = Document()
        for _, row in results.iterrows():
            # Add a paragraph for each row
            doc.add_paragraph(
                f"Page: {row['page']}, Content: {row['content']}, Confidence: {row['confidence']}"
            )

        # Save the Document
        doc.save(outpath)

        return outpath


if __name__ == "__main__":
    logger = logging.getLogger()
    azurelogger = logging.getLogger("azure")
    logger.setLevel(logging.INFO)
    azurelogger.setLevel(logging.ERROR)

    # Create output directory if it does not exist
    if not os.path.exists("../../data/convictions/evaluate/reports"):
        os.makedirs("../../data/convictions/evaluate/reports")

    endpoint, key = getcreds()
    client = DocClient(endpoint, key)

    files = [
        f
        for f in os.listdir(doc_directory)
        if os.path.isfile(os.path.join(doc_directory, f))
    ]
    logging.info(f"starting to process {len(files)} files")
    for file in files:
        client.process(os.path.join(doc_directory, file))

    client.close()
