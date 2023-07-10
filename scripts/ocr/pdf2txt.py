import logging
import numpy as np
from PIL import Image
import os
import cv2
import pytesseract
import re
from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path
import numpy as np
from skimage import filters
from skimage.transform import rotate
from scipy.ndimage import interpolation as inter
from skimage.util import img_as_ubyte


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_pdf(file_path, start_page=0, end_page=None, dpi=300):
    # Initialize the PDF file reader
    pdf = PdfFileReader(open(file_path, "rb"))

    # Get the number of pages
    num_pages = pdf.getNumPages()

    # Limit the range of processed pages
    if end_page is None:
        end_page = num_pages

    # Preprocess each page and extract text using Tesseract OCR
    text = ""
    for page_num in range(start_page, end_page):
        # Convert the page to an image
        image = convert_from_path(
            file_path, dpi=dpi, first_page=page_num + 1, last_page=page_num + 1
        )[0]

        # Convert the PIL image to OpenCV format (numpy array)
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Apply preprocessing operations (e.g., sharpening, denoising, binarization)
        processed_image = preprocess_image(img_cv)

        # Convert the processed image back to PIL format
        processed_image_pil = Image.fromarray(
            cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        )

        # Extract text from the image using Tesseract OCR
        extracted_text = pytesseract.image_to_string(processed_image_pil, lang="eng")
        text += extracted_text + "\n"

    return text


def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to convert the image to binary
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove noise
    denoised = cv2.fastNlMeansDenoising(binary, h=10)

    # Deskew the image
    def compute_skew(image):
        # Convert image to binary and invert
        image = cv2.bitwise_not(image)
        # Compute the coordinates of non-black pixels
        coords = np.column_stack(np.where(image > 0))
        # Fit a line to these points
        angle = cv2.minAreaRect(coords)[-1]
        # The `cv2.minAreaRect` function returns values in the range [-90, 0)
        # We need to adjust them to get angles in the range [0, 90)
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        return angle

    skew_angle = compute_skew(denoised)
    deskewed = rotate(denoised, skew_angle, mode="reflect")

    # Convert image back to uint8 format
    deskewed = img_as_ubyte(deskewed)

    return deskewed


def sanitize_text(text):
    # Remove control characters and NULL bytes from the text
    sanitized_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

    # Add additional text sanitization techniques here if needed
    # For example, remove excessive spaces, line breaks, or special characters
    sanitized_text = re.sub(r"\s+", " ", sanitized_text)
    return sanitized_text.strip()


def convert_pdfs_to_text_files(
    pdf_directory, start_page=0, end_page=None, batch_size=10, dpi=300
):
    # Create a list of PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

    # Loop through each PDF file and convert it to a text-readable document
    for pdf_file in pdf_files:
        # Construct the paths for input and output files
        input_path = os.path.join(pdf_directory, pdf_file)
        output_path = os.path.join(
            pdf_directory, f"{os.path.splitext(pdf_file)[0]}.txt"
        )

        try:
            # Check if the output file already exists
            if os.path.exists(output_path):
                logger.info(
                    f"Skipping {pdf_file} because {os.path.basename(output_path)} already exists"
                )
                continue

            # Preprocess the PDF to make it OCRable
            text = preprocess_pdf(
                input_path, start_page=start_page, end_page=end_page, dpi=dpi
            )

            # Sanitize the extracted text
            sanitized_text = sanitize_text(text)

            # Save the extracted text to a .txt file
            with open(output_path, "w") as text_file:
                text_file.write(sanitized_text)

            logger.info(f"Converted {pdf_file} to {os.path.basename(output_path)}")
        except Exception as e:
            logger.error(f"Error converting {pdf_file}: {str(e)}")


# Usage example
pdf_directory = "../../data/convictions/v1/testimony"
start_page = 0  # Start processing from page 1
end_page = None  # Process up to the last page
batch_size = 10
dpi = 300

convert_pdfs_to_text_files(
    pdf_directory,
    start_page=start_page,
    end_page=end_page,
    batch_size=batch_size,
    dpi=dpi,
)
