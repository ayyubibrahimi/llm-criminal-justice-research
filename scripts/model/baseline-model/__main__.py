import os
from baseline import parse_names

txt_dir = "../../data/convictions/transcripts"

csv_dir = "../../data/convictions/transcripts"

if __name__ == '__main__':
    for filename in os.listdir(txt_dir):
        if filename.endswith(".txt"):
            txt_path = os.path.join(txt_dir, filename)
            csv_path = os.path.join(csv_dir, filename.replace(".txt", ".csv"))
            parse_names(txt_path, csv_path)
