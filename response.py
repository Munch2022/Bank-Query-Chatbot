import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

from cleaning_function import CustomTokenizer
token= CustomTokenizer()
le = LE()
tfv = TfidfVectorizer(min_df=1, stop_words='english')

# load the model from disk
svc = pickle.load(open('svcmodel.pkl', 'rb'))

# reading csv dataset file, removing duplicates and transforming X, Y (with tfv and le)
df= pd.read_csv ("Original_BankFAQs.csv")
clean_df= df.drop_duplicates(subset= ['Question', 'Class'])
clean_df['Cleaned_Questions']= clean_df['Question'].apply(token.preprocess_text)
le = LE()
tfv = TfidfVectorizer(min_df=1, stop_words='english')

X = tfv.fit_transform(clean_df['Cleaned_Questions'])
Y = le.fit_transform(clean_df['Class'])

class BotReply():
    def __init__(self):
        pass
    # writing function for response 
    from sklearn.metrics.pairwise import cosine_similarity
    def get_response(self, usrText):
        while True:
            t_usr = tfv.transform([token.preprocess_text(usrText.strip())])
            class_ = le.inverse_transform(svc.predict(t_usr))

            questionset = clean_df[clean_df['Class'].values == class_]

            cos_sims = []
            for question in questionset['Question']:
                sims = cosine_similarity(tfv.transform([question]), t_usr)
                cos_sims.append(sims)

            ind = cos_sims.index(max(cos_sims))

            b = [questionset.index[ind]]
            
            if max(cos_sims) > [[0.]]:
                a = clean_df['Answer'][questionset.index[ind]]+"   "
                return a
            elif max(cos_sims)==[[0.]]:
                return "sorry! I'm not able to solve this question at this moment. You can call to customer support 1860 000 0000 \U0001F615"