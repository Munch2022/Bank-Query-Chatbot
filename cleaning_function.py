# Preprocessing the sentence

# lets first import all the neccessary libraries
import pandas as pd
import numpy as np
import string
# methods and stop words text processing
import spacy
import nltk
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
import warnings
warnings.filterwarnings('ignore')

# downloading stopwords
nlp= spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
stopwords= STOP_WORDS

# using this for noise removal
ignore_letters= ['?', '!', '.', ',', '#', '/']

class CustomTokenizer():
    def __init__(self):
        pass

    # writing a function for preprocessing
    def preprocess_text(self, sent):
        #convert all text to lower case
        sent= sent.lower()

        #remove stopwords ; before removing stopwords we need to convert sentence to tokens
        sent_tokens= word_tokenize(sent)
        filterted_words= [word for word in sent_tokens if word not in stopwords and word not in ignore_letters]
  
       #stemming
        ps= PorterStemmer()
        stemmed_words= [ps.stem(word) for word in filterted_words]

        # lemmatizing
        lammatizer= WordNetLemmatizer()
        lemma_words= [lammatizer.lemmatize(word) for word in stemmed_words]

        # to eliminate duplicates 
        words= (set(lemma_words))

        return " ".join(words)