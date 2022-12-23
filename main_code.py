import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report



qadf= pd.read_csv ("Original_BankFAQs.csv")

#  lets delete the duplicate questions and class to have a clean dataset
clean_qadf= qadf.drop_duplicates(subset= ['Question', 'Class'])

# methods and stop words text processing

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# creating a stopwords set and store them to a variable
import spacy
nlp= spacy.load('en_core_web_sm')

from spacy.lang.en.stop_words import STOP_WORDS
stopwords= STOP_WORDS
# print(stopwords)

# using this for noise removal
ignore_letters= ['?', '!', '.', ',', '#', '*', '/']


# writing a function for preprocessing
def preprocess_text(sent):
    #convert all text to lower case
    sent= sent.lower()

    #remove stopwords ; before removing stopwords we need to convert sentence to tokens
    sent_tokens= word_tokenize(sent)
    filterted_words= [word for word in sent_tokens if word not in stopwords and word not in ignore_letters]
  
    #stemming
    ps= PorterStemmer()
    stemmed_words= [ps.stem(word) for word in filterted_words]

    #lemmatizing
    lammatizer= WordNetLemmatizer()
    lemma_words= [lammatizer.lemmatize(word) for word in stemmed_words]

    #to eliminate duplicates 
    words= (set(lemma_words))

    return " ".join(words)

clean_qadf['Cleaned_Questions']= clean_qadf['Question'].apply(preprocess_text)
questions= clean_qadf['Cleaned_Questions'].values
clas= clean_qadf['Class'].values

from sklearn.preprocessing import LabelEncoder as LE

le = LE()
tfv = TfidfVectorizer(min_df=1, stop_words='english')

X= tfv.fit_transform(questions)
Y= le.fit_transform(clas)
# print(set(Y))

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.20, random_state= 20)

# using SVC algorithm
from sklearn.svm import SVC

model_svm = SVC(kernel='linear')
model_svm.fit(X_train, Y_train)
y1_pred= model_svm.predict(X_test)

print(classification_report(Y_test, y1_pred))

# Saving the trained model
import pickle
# pickle.dump(model_svm, open('svcmodel.pkl', 'wb'))

# tesitng new question 
txt1= 'Can a laptop be covered under a Home Insurance policy? '

# load the model 
model= pickle.load(open('svcmodel.pkl', 'rb'))
predict= model.predict(txt1)
print(predict)