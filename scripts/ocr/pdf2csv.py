import os
import PyPDF2
import textract
import re
import csv


def preprocess_pdf(file_path):
    # Preprocess the PDF using textract
    text = textract.process(file_path, method="tesseract", language="eng")
    return text.decode("utf-8")


def sanitize_text(text):
    # Remove control characters and NULL bytes from the text
    sanitized_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)
    return sanitized_text


def convert_pdfs_to_csv(pdf_directory):
    # Create a list of PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

    # Loop through each PDF file and convert it to a CSV file
    for pdf_file in pdf_files:
        # Construct the paths for input and output files
        input_path = os.path.join(pdf_directory, pdf_file)
        output_path = os.path.join(
            pdf_directory, f"{os.path.splitext(pdf_file)[0]}.csv"
        )

        # Preprocess the PDF to make it OCRable
        text = preprocess_pdf(input_path)

        # Sanitize the extracted text
        sanitized_text = sanitize_text(text)

        # Split the sanitized text into rows
        rows = sanitized_text.split("\n")

        # Save the extracted text as a CSV file
        with open(output_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["text"])  # Header
            for row in rows:
                writer.writerow([row])

        print(f"Converted {pdf_file} to {os.path.basename(output_path)}")


# Usage example
pdf_directory = "../data/lsp"
convert_pdfs_to_csv(pdf_directory)
