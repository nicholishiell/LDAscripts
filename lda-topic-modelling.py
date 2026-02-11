# Libraries for PDF processing, string and data hanlding
import glob
import os

# Visualization Libraries
import pyLDAvis
import pyLDAvis.gensim

# My helper library
from utils import *

# Gensim Libraries
import gensim
from gensim import corpora

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def perform_lda_topic_modelling(preprocessed_data_base_path):

    # Load preprocessed data
    preprocessed_data_file_paths = glob.glob(preprocessed_data_base_path + "/*.json")

    data_words = []
    for file_path in preprocessed_data_file_paths:
        data_words.append(load_data(file_path))

    # create bi_grams and tri_grams
    data_words = generate_n_grams(data_words)

    # now use the data with bigrams and trigrams to create the bag of words and corpus for LDA topic modelling
    id2word = corpora.Dictionary(data_words)

    corpus = [id2word.doc2bow(text) for text in data_words]

    # use the Tfidf model to remove extreme values from the bag of words
    corpus = remove_common_words_from_corpus(corpus, id2word)

    # now perform LDA topic modelling
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus[:-1],
                                                id2word=id2word,
                                                num_topics=5,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha="auto")

    return lda_model, corpus, id2word

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def check_directories(directories):

    for directory in directories:
        if not os.path.exists(directory):
            print(f"Please create a directory called '{directory}' and add the PDF files you want to process in there.")
            return True

    return False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main(pdf_base_path = "./pdfs/",
         preprocessed_data_base_path = "./preprocessed_data",
         do_preprocessing = True,
         do_lda_topic_modelling = True):

    if check_directories([pdf_base_path, preprocessed_data_base_path]):
        return

    # Preprocess PDFs
    if do_preprocessing:
        pdf_file_paths = glob.glob(pdf_base_path + "*.PDF")

        print(f"Found {len(pdf_file_paths)} PDF files in {pdf_base_path}.")

        for pdf_path in pdf_file_paths:
            print(f"Processing {pdf_path}...", end="")
            processed_text = preprocess_pdf(pdf_path)
            save_preprocessed_data(processed_text, preprocessed_data_base_path, pdf_path)
            print("DONE!")

    # Perform LDA topic modelling
    if do_lda_topic_modelling:
        lda_model, corpus, id2word = perform_lda_topic_modelling(preprocessed_data_base_path)

        vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
        pyLDAvis.save_html(vis, "lda_vis.html")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    # adjustable parameters
    # n_topics
    # n_passes
    # do_preprocessing
    # do_lda_topic_modelling

    main(do_preprocessing=False, do_lda_topic_modelling=True)