# Simple Linear Regression

'''
This class contains commonly used modules
'''

# Importing the libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import json
import os,re,pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sentence_transformers import SentenceTransformer
import spacy,string

base_folder = os.getcwd()
model_folder = "models_new"
model_path = os.path.join(base_folder, model_folder)

if not os.path.exists(model_path):
    os.makedirs(model_path)

#we will use SentenceTransformer for encoding our text data
sent_transformer_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

#We will use spacy for data cleaning
spacy_model = spacy.load("en_core_web_sm")
stop_words = spacy_model.Defaults.stop_words

def write_pickle(obj, fileName):
    pickle_file_path = os.path.join(model_path, fileName)
    pickle.dump(obj, open(pickle_file_path, "wb"))

def read_pickle(fileName):
    pickle_file_path = os.path.join(model_path, fileName)
    return pickle.load(open(pickle_file_path, "rb"))
    
    
#Clean errors in the text data
def clean_data(sentence):
    
    #Remove @ sign and the characters followed by @sign
    sentence = re.sub('http://\S+|https://\S+', '', sentence)
    
    sentence = re.sub("\n|\r|'","",sentence)
    
    #Keep only numbers, text and %
    sentence  = re.sub('[^A-Za-z]+', ' ', sentence)  
    
    #Remove exta space between words
    sentence = re.sub(' +', ' ', sentence)
    
    #fix wrong spellings and return
    #sentence = TextBlob(sentence).correct() 
    
    sentence = sentence.strip()
    return str(sentence).lower()


# Creating spacy tokenizer function
def create_spacy_tokens(sentence):
    
    # Creating token object for all sentence
    document = spacy_model(sentence)

    # Lemmatizing every token
    tokens = [ word.lemma_.strip() for word in document ]
    
    # Removing stopwords
    sentence = " ".join([ word for word in tokens if word not in stop_words ])
    
    return sentence
    