import glob

from utils import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main(pdf_base_path = "./pdfs/",
         preprocessed_data_base_path = "./preprocessed_data"):

    if check_directories([pdf_base_path, preprocessed_data_base_path]):
        return


    pdf_file_paths = glob.glob(pdf_base_path + "*.PDF")

    print(f"Found {len(pdf_file_paths)} PDF files in {pdf_base_path}.")

    for pdf_path in pdf_file_paths:
        print(f"Processing {pdf_path}...", end="")
        processed_text = preprocess_pdf(pdf_path)
        save_preprocessed_data(processed_text, preprocessed_data_base_path, pdf_path)
        print("DONE!")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":
    main()