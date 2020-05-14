import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from string import punctuation
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle
from sklearn.externals import joblib
import os

#stop word for cleaning pitches that were input by users
stopwords = stopwords.words( 'english' ) + list(punctuation)
stemmer = PorterStemmer()

def run_model(input_pitch, input_amount, input_exchange, input_gender, input_category):
    #Define veriables from page POST method
    dealloc = "Deal_Status_model.pkl"
    dealvoc = "Deal_Status_vocab.pkl"
    sharkloc = "Deal_Shark 1_model.pkl"
    sharkvoc = "Deal_Shark 1_vocab.pkl"
    path = os.path.join("Shark_Data","static","py",dealloc)
    vocpath = os.path.join("Shark_Data","static","py",dealvoc)
    sharkpath = os.path.join("Shark_Data","static","py",sharkloc)
    sharkvocpath = os.path.join("Shark_Data","static","py",sharkvoc)

    #Load the trained models
    #This model predicts whether a deal would be made or not
    deal_model = joblib.load(path)
    #This model predicts who would make the deal
    shark_model = joblib.load(sharkpath)

    #Deal Model
    #Creates the vector for the pitch word input
    transformer = TfidfTransformer()
    loaded_vectorizer = TfidfVectorizer(decode_error="replace",vocabulary=pickle.load(open(vocpath, "rb")))
    tfidf = transformer.fit_transform(loaded_vectorizer.fit_transform(np.array(input_pitch)))
    input_df = pd.DataFrame(tfidf.toarray(), columns=loaded_vectorizer.get_feature_names())

    #Adds the numerical and catagorical factors to the table
    input_df.loc[0,"Amount_Asked_For"] = input_amount
    input_df.loc[0,"Exchange_For_Stake"] = input_exchange
    input_df.loc[0,f"Gender_{input_gender}"] = 1
    input_df.loc[0,f"Category_{input_category}"] = 1

    #If the model predicts 1, then a deal has been made, 0 no deal
    dealprediction = deal_model.predict(input_df)

    #If a deal has been made, run the second model to predict who might pick it up
    if (dealprediction[0] == 1):

        #Shark Model
        transformer = TfidfTransformer()
        loaded_vectorizer = TfidfVectorizer(decode_error="replace",vocabulary=pickle.load(open(sharkvocpath, "rb")))
        tfidf = transformer.fit_transform(loaded_vectorizer.fit_transform(np.array(input_pitch)))
        input_df_shark = pd.DataFrame(tfidf.toarray(), columns=loaded_vectorizer.get_feature_names())

        input_df_shark.loc[0,"Amount_Asked_For"] = input_amount
        input_df_shark.loc[0,"Exchange_For_Stake"] = input_exchange
        input_df_shark.loc[0,f"Gender_{input_gender}"] = 1
        input_df_shark.loc[0,f"Category_{input_category}"] = 1

        sharkprediction = shark_model.predict(input_df_shark)

        #Returns the deal status and the deal shark
        return dealprediction[0], sharkprediction
    
    else:
        #Returns just the deal status
        return dealprediction[0], np.nan

    