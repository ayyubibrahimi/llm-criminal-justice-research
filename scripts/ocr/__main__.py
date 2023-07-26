import os
import logging
from src import getcreds, DocClient

if __name__ == "__main__":
    logger = logging.getLogger()
    azurelogger = logging.getLogger("azure")
    logger.setLevel(logging.INFO)
    azurelogger.setLevel(logging.ERROR)

    f_path = "../../data/citycouncil/agendas"
    if not os.path.exists(f_path):
        os.makedirs(f_path)

    endpoint, key = getcreds()
    client = DocClient(endpoint, key)

    files = [
        f
        for f in os.listdir(f_path)
        if os.path.isfile(os.path.join(f_path, f))
    ]
    logging.info(f"starting to process {len(files)} files")
    for file in files:
        client.process(os.path.join(f_path, file))

    client.close()
