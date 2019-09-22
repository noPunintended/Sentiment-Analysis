from flask import Flask, request, jsonify
import traceback
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import deepcut
import re
from gensim.models import Word2Vec
from flask import Flask, request, jsonify
import string
import requests
import json
import time

app = Flask(__name__)
num_features = 300

@app.route('/analyze', methods=['GET'])

def analyze():
    if request.method == 'GET':
        start = time.time()
        text = request.args.get('text')
        text = preprocessing(text)
        data = []
        data.append(text)
        DataVecs = w2v(data, model)
        results = loaded_model.predict(DataVecs)
        final = convert(results)
        end = time.time()
        print(end - start)
    return final

def cleaning(text):
    # clean the text by getting rid of any punctuation
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

def split_words (sentence):
    return deepcut.tokenize(''.join(sentence.lower().split()))


def preprocessing(data):
    data = cleaning(data)
    data = split_words(data)
    return data

#This function put each word in a sentence into vector form
def featureVecMethod(words, model, num_features):
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
            if np.isnan(featureVec).sum()!= 0:
                print(word)
    
    # Dividing the result by number of words to get average
    if nwords != 0:
        featureVec = np.divide(featureVec, nwords)
    
    return featureVec

def getAvgFeatureVecs(texts, model, num_features):
    counter = 0
    textFeatureVecs = np.zeros((len(texts),num_features),dtype="float32")
    for text in texts:
            
        textFeatureVecs[counter] = featureVecMethod(text, model, num_features)
        counter = counter+1
        
    return textFeatureVecs

def w2v(data, model):
    DataVecs = getAvgFeatureVecs(data, model, num_features)
    return DataVecs

def convert(results):
    final = { "success": "Fail", "sentiment": "Fail" }
    if results[0] == 0:
        final['success'] = "True"
        final['sentiment'] = "negative"
    elif results[0] == 1:
        final['success'] = "True"
        final['sentiment'] = "neutral"
    elif results[0] == 2:
        final['success'] = "True"
        final['sentiment'] = "positive"

    return final

        


if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345
    loaded_model = joblib.load('XGBoost_finalized_model.sav')
    model = Word2Vec.load("300features_10minwords_10context.model")
    print ('Model loaded')
    app.run(port=port, debug=True, use_reloader=False)
