import glob
import argparse
import os

from utils import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class PreprocessConfig:

    def __init__(self):
        self.pdf_base_path = "./pdfs/"
        self.preprocessed_data_base_path = "./preprocessed_data"

    def check_validity(self):
        if self.pdf_base_path is None:
            print("pdf_base_path cannot be None")
            return False
        if self.preprocessed_data_base_path is None:
            print("preprocessed_data_base_path cannot be None")
            return False

        return True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main(config):

    # load NLP model
    nlp = load_nlp_model()

    if check_directories([config.pdf_base_path, config.preprocessed_data_base_path]):
        return

    pdf_file_paths = glob.glob(config.pdf_base_path + "*.PDF")
    print(f"Found {len(pdf_file_paths)} PDF files in {config.pdf_base_path}.")

    for pdf_path in pdf_file_paths:
        print(f"Processing {pdf_path}...", end="")
        processed_text = preprocess_pdf(pdf_path, nlp)
        save_preprocessed_data(processed_text, config.preprocessed_data_base_path, pdf_path)
        print("DONE!")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_config(config_path):

    config = PreprocessConfig()

    with open (config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
        config.pdf_base_path = config_dict.get("pdf_base_path", None)
        config.preprocessed_data_base_path = config_dict.get("preprocessed_data_base_path", None)
        config.min_count = config_dict.get("min_count", None)
        config.bigram_threshold = config_dict.get("bigram_threshold", None)
        config.trigram_threshold = config_dict.get("trigram_threshold", None)
        config.tfidf_threshold = config_dict.get("tfidf_threshold", None)

    return config

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)

    if config.check_validity():
        main(config)