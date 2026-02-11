# Libraries for PDF processing, string and data hanlding
import glob
import argparse
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

class LDAConfig:

    def __init__(self, n_topics=5, n_passes=10):
        self.num_topics = n_topics
        self.random_state = 100
        self.update_every = 1
        self.chunksize = 100
        self.passes = n_passes
        self.alpha = "auto"
        self.preprocessed_data_base_path = "./preprocessed_data"
        self.R = 30
        self.min_count = 5
        self.bigram_threshold = 100
        self.trigram_threshold = 100
        self.tfidf_threshold = 0.03

    def check_validity(self):
        if self.num_topics is None:
            print("num_topics not set in config")
            return False
        if self.passes is None:
            print("passes not set in config")
            return False
        if self.alpha is None:
            print("alpha not set in config")
            return False
        if self.preprocessed_data_base_path is None:
            print("preprocessed_data_base_path not set in config")
            return False
        if self.R is None:
            print("R not set in config")
            return False
        if self.min_count is None:
            print("min_count not set in config")
            return False
        if self.bigram_threshold is None:
            print("bigram_threshold not set in config")
            return False
        if self.trigram_threshold is None:
            print("trigram_threshold not set in config")
            return False
        if self.tfidf_threshold is None:
            print("tfidf_threshold not set in config")
            return False

        if self.num_topics <= 0:
            print("num_topics must be greater than 0")
            return False
        if self.passes <= 0:
            print("passes must be greater than 0")
            return False
        if self.R <= 0:
            print("R must be greater than 0")
            return False

        return True

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def perform_lda_topic_modelling(config):

    # Load preprocessed data
    preprocessed_data_file_paths = glob.glob(config.preprocessed_data_base_path + "/*.json")

    data_words = []
    for file_path in preprocessed_data_file_paths:
        data_words.append(load_data(file_path))

    # create bi_grams and tri_grams
    data_words = generate_n_grams(data_words,
                                  min_count=config.min_count,
                                  bigram_threshold=config.bigram_threshold,
                                  trigram_threshold=config.trigram_threshold)

    # now use the data with bigrams and trigrams to create the bag of words and corpus for LDA topic modelling
    id2word = corpora.Dictionary(data_words)

    corpus = [id2word.doc2bow(text) for text in data_words]

    # use the Tfidf model to remove extreme values from the bag of words
    corpus = remove_common_words_from_corpus(corpus,
                                             id2word,
                                             tfidf_threshold=config.tfidf_threshold)

    # now perform LDA topic modelling
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus[:-1],
                                                id2word=id2word,
                                                num_topics=config.num_topics,
                                                random_state=config.random_state,
                                                update_every=config.update_every,
                                                chunksize=config.chunksize,
                                                passes=config.passes,
                                                alpha=config.alpha)

    return lda_model, corpus, id2word

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main(config):

    if check_directories([config.pdf_base_path, config.preprocessed_data_base_path]):
        return

    lda_model, corpus, id2word = perform_lda_topic_modelling(config)

    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, mds="mmds", R=config.R)

    pyLDAvis.save_html(vis, "lda_vis.html")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_config(config_path):

    config = LDAConfig()

    with open(config_path, "r") as f:
        config_dict = json.load(f)
        config.num_topics = config_dict.get("num_topics", None)
        config.passes = config_dict.get("passes", None)
        config.alpha = config_dict.get("alpha", None)
        config.pdf_base_path = config_dict.get("pdf_base_path", None)
        config.preprocessed_data_base_path = config_dict.get("preprocessed_data_base_path", None)
        config.R = config_dict.get("R", None)
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
