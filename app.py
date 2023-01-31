import numpy as np
from flask import Flask, request, jsonify, render_template
from common import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    text_comment = [str(x) for x in request.form.values()]
    
    vectorizer = "Sentence_Transformer_LR"
    
    # Load the model
    model_name = vectorizer+".pkl"
    clf = read_pickle(model_name)
    
    #Clean the text
    clean_str = clean_data(text_comment)
    test_str = pd.Series(clean_str)
    
    #Encode using sentence transformer
    X_test = sent_transformer_model.encode(test_str)
    
    # Make prediction using model loaded from disk as per the data.
    prediction = clf.predict(X_test)

    # Take the first value of prediction
    print("prediction :: ",prediction)
    output = prediction[0]

    return render_template('index.html', prediction_text='If Toxic {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)