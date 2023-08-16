import csv
import re


def parse_names(txt_path, csv_path):
    with open(txt_path, "r", encoding="utf-8") as txt_file:
        content = txt_file.read()

    pattern = re.compile(
        r"(detective|sergeant|lieutenant|captain|corporal|deputy|criminalist|technician|investigator
        r"|det\.|sgt\.|lt\.|cpt\.|cpl\.|dty\.|tech\.|dr\.)\s+([A-Z][A-Za-z]*(\s[A-Z][A-Za-z]*)?)",
        re.IGNORECASE,
    )
    matches = pattern.findall(content)

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Title", "Name"])
        for match in matches:
            writer.writerow([match[0], match[1]])
