import logging
import numpy as np
from PIL import Image
import os
import cv2
import pytesseract
import re
from docx import Document
from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path
import numpy as np
from skimage import filters
from skimage.transform import rotate
from scipy.ndimage import interpolation as inter
from skimage.util import img_as_ubyte

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_pdf(file_path, start_page=0, end_page=None, dpi=300):
    pdf = PdfFileReader(open(file_path, "rb"))

    num_pages = pdf.getNumPages()

    if end_page is None:
        end_page = num_pages

    text = ""
    for page_num in range(start_page, end_page):
        image = convert_from_path(
            file_path, dpi=dpi, first_page=page_num + 1, last_page=page_num + 1
        )[0]

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        processed_image = preprocess_image(img_cv)

        processed_image_pil = Image.fromarray(
            cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        )

        extracted_text = pytesseract.image_to_string(processed_image_pil, lang="eng")
        text += extracted_text + "\n"

    return text


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    denoised = cv2.fastNlMeansDenoising(binary, h=10)

    def compute_skew(image):
        image = cv2.bitwise_not(image)
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
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
    sanitized_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)

    sanitized_text = re.sub(r"\s+", " ", sanitized_text)
    return sanitized_text.strip()


def convert_pdfs_to_text_doc(
    pdf_directory, start_page=0, end_page=None, batch_size=10, dpi=300
):
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        input_path = os.path.join(pdf_directory, pdf_file)
        output_path = os.path.join(
            pdf_directory, f"{os.path.splitext(pdf_file)[0]}.docx"
        )

        try:
            if os.path.exists(output_path):
                logger.info(
                    f"Skipping {pdf_file} because {os.path.basename(output_path)} already exists"
                )
                continue

            text = preprocess_pdf(
                input_path, start_page=start_page, end_page=end_page, dpi=dpi
            )

            sanitized_text = sanitize_text(text)

            doc = Document()
            doc.add_paragraph(sanitized_text)
            doc.save(output_path)

            logger.info(f"Converted {pdf_file} to {os.path.basename(output_path)}")
        except Exception as e:
            logger.error(f"Error converting {pdf_file}: {str(e)}")


pdf_directory = "../../data/convictions/evaluate/reports"
start_page = 0  
end_page = None  
batch_size = 10
dpi = 400

convert_pdfs_to_text_doc(
    pdf_directory,
    start_page=start_page,
    end_page=end_page,
    batch_size=batch_size,
    dpi=dpi,
)
