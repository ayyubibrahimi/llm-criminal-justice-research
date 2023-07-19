import os
import PyPDF2
import textract
import re
import csv


def preprocess_pdf(file_path):
    text = textract.process(file_path, method="tesseract", language="eng")
    return text.decode("utf-8")


def sanitize_text(text):
    sanitized_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)
    return sanitized_text


def convert_pdfs_to_csv(pdf_directory):
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        input_path = os.path.join(pdf_directory, pdf_file)
        output_path = os.path.join(
            pdf_directory, f"{os.path.splitext(pdf_file)[0]}.csv"
        )

        text = preprocess_pdf(input_path)

        sanitized_text = sanitize_text(text)

        rows = sanitized_text.split("\n")

        with open(output_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["text"])  # Header
            for row in rows:
                writer.writerow([row])

        print(f"Converted {pdf_file} to {os.path.basename(output_path)}")


pdf_directory = "../data/lsp"
convert_pdfs_to_csv(pdf_directory)
