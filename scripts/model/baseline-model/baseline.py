import csv
import re
import logging
import argparse
import os
from docx import Document

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--input")
parser.add_argument("--output")
args = parser.parse_args()

def parse_names(docx_path, csv_path):
    doc = Document(docx_path)
    content = "\n".join([para.text for para in doc.paragraphs])
    
    pattern = re.compile(
        r"(detective|sergeant|lieutenant|captain|corporal|deputy|criminalist|technician|investigator"
        r"|det\.|sgt\.|lt\.|cpt\.|cpl\.|dty\.|tech\.|dr\.)\s+([A-Z][A-Za-z]*(\s[A-Z][A-Za-z]*)?)",
        re.IGNORECASE,
    )
    matches = pattern.findall(content)

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Title", "Name"])
        for match in matches:
            writer.writerow([match[0], match[1]])
            

if __name__ == '__main__':
    docx_dir = args.input
    csv_dir = args.output 
    parse_names(docx_dir, csv_dir)
