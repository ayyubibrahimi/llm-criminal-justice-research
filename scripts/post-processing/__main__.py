from dotenv import load_dotenv
from src import Reformat
import os
import asyncio

load_dotenv()

async def process_all_files(api_key, question, input_dir):
    assistant = Reformat(api_key)

    for filename in os.listdir(input_dir):
        if filename.endswith(".docx"):
            docx_path = os.path.join(input_dir, filename)
            await assistant.generate_and_save_response(docx_path, question)

def main():
    api_key = ""
    question = "This is a court transcript that was OCRd. As you can tell, the OCR has messed up for the formatting so that the document no longer reads like a court transcript. Use your knowledge of court transcripts to re-write this document so that it is formatted correctly"
    input_dir = "pdf"  

    asyncio.run(process_all_files(api_key, question, input_dir))

if __name__ == "__main__":
    main()