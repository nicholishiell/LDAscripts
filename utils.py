import fitz # PyMuPDF
import json
import os

# Gensim Libraries
import gensim
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import TfidfModel

# Spacy and NLTK for text processing
import spacy
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from spacy_langdetect import LanguageDetector
from spacy.language import Language

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def check_directories(directories):

    for directory in directories:
        if not os.path.exists(directory):
            print(f"Please create a directory called '{directory}' and add the PDF files you want to process in there.")
            return True

    return False

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_lang_detector(nlp, name):
    return LanguageDetector()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_nlp_model():
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    Language.factory("language_detector", func=get_lang_detector)
    nlp.add_pipe('sentencizer')
    nlp.add_pipe("language_detector", last=True)
    return nlp

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def extract_raw_text_from_pdf(pdf_path):

    doc = fitz.open(pdf_path)

    page_text = ''

    for page in doc:
       page_text += page.get_text("text", sort=False)

    return page_text

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_data(file):
    with open (file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (data)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def write_data(file, data):
    with open (file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def lemmatize_text(text, nlp, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):

    doc = nlp(text)
    new_text = []

    for token in doc:
        if token.pos_ in allowed_postags:
            new_text.append(token.lemma_) # this appends the lemmatized version of the token (word)

    text_out = " ".join(new_text)

    return text_out

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def remove_french(text, nlp):

    doc = nlp(text)

    english_sentences = []

    for sentence in doc.sents:
        if sentence._.language['language'] == 'en':
            english_sentences.append(sentence.text)

    text_out = " ".join(english_sentences)

    return text_out

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def preprocess_pdf(pdf_path):

    # load componets
    nlp = load_nlp_model()

    # start preprocessing of the pdf
    raw_text = extract_raw_text_from_pdf(pdf_path)

    english_only_text = remove_french(raw_text,nlp)

    lemmatized_text = lemmatize_text(english_only_text,nlp)

    preprocessed_text = simple_preprocess(lemmatized_text, deacc=True)

    return preprocessed_text

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def save_preprocessed_data(preprocessed_text, preprocessed_data_base_path, pdf_path):

    file_name = pdf_path.split("/")[-1].split(".pdf")[0]
    write_data(preprocessed_data_base_path + "/" + file_name + ".json", preprocessed_text)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def generate_n_grams(data_words):

    # create bi_grams and tri_grams
    bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)

    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)

    data_bigrams = [bigram[doc] for doc in data_words]
    data_bigrams_trigrams = [trigram[bigram[doc]] for doc in data_bigrams]

    return data_bigrams_trigrams

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def remove_common_words_from_corpus(corpus, id2word):

    # remove extreme values from the bag of words using Tfid model
    tfidf = TfidfModel(corpus, id2word=id2word)

    low_value = 0.03
    words  = []
    words_missing_in_tfidf = []

    for i, _ in enumerate(corpus):
        bow = corpus[i]
        low_value_words = [] #reinitialize to be safe. You can skip this.
        tfidf_ids = [id for id, value in tfidf[bow]]
        bow_ids = [id for id, value in bow]
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        drops = low_value_words+words_missing_in_tfidf
        for item in drops:
            words.append(id2word[item])
        words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf socre 0 will be missing

        new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
        corpus[i] = new_bow

    return corpus