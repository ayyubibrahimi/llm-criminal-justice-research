import logging
import numpy as np
from PIL import Image
import os
import cv2
import pytesseract
import re
from pdf2image import convert_from_path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

image_cache = {}

def preprocess_pdf(file_path):
    if file_path in image_cache:
        images = image_cache[file_path]
    else:
        # Convert PDF to images
        images = convert_from_path(file_path, dpi=250)
        # Store the images in the cache for future use
        image_cache[file_path] = images

    # Preprocess each image and extract text using Tesseract OCR
    text = ""
    for image in images:
        # Convert the PIL image to OpenCV format (numpy array)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Apply preprocessing operations
        processed_image = preprocess_image(img_cv)

        # Convert the processed image back to PIL format
        processed_image_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        # Set custom options for Tesseract OCR
        custom_config = r'--oem 3 --psm 6'

        # Extract text from the image using Tesseract OCR
        extracted_text = pytesseract.image_to_string(processed_image_pil, lang='eng', config=custom_config)
        text += extracted_text + "\n"
    return text

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoising the image
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Apply adaptive thresholding to binarize the image
    _, binary_image = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image

def sanitize_text(text):
    # Remove control characters and NULL bytes from the text
    sanitized_text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    return sanitized_text

def convert_pdfs_to_text_doc(pdf_directory):
    # Create a list of PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

    # Loop through each PDF file and convert it to a text-readable document
    for pdf_file in pdf_files:
        # Construct the paths for input and output files
        input_path = os.path.join(pdf_directory, pdf_file)
        output_path = os.path.join(pdf_directory, f'{os.path.splitext(pdf_file)[0]}.txt')

        try:
            # Check if the output file already exists
            if os.path.exists(output_path):
                logger.info(f"Skipping {pdf_file} because {os.path.basename(output_path)} already exists")
                continue

            # Preprocess the PDF to make it OCRable
            text = preprocess_pdf(input_path)

            # Sanitize the extracted text
            sanitized_text = sanitize_text(text)

            # Save the extracted text as a .txt file
            with open(output_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(sanitized_text)

            logger.info(f"Converted {pdf_file} to {os.path.basename(output_path)}")
        except Exception as e:
            logger.error(f"Error converting {pdf_file}: {str(e)}")

# Usage example
pdf_directory = "../../data/convictions"
convert_pdfs_to_text_doc(pdf_directory)
