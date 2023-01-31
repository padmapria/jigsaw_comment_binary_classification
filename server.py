# Create API of ML model using flask

'''
This code takes the JSON data while POST request an performs the prediction using loaded model and returns
the results in JSON format.
'''

# Import libraries and common modules
from flask import Flask, request, jsonify
from common import *

app = Flask(__name__)


@app.route('/api/',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    vectorizer = data['vectorizer']
    
    # Load the model
    model_name = vectorizer+".pkl"
    rf = read_pickle(model_name)
    
    
    #Clean the text
    clean_str = clean_data(data['comment_text'])
    test_str = pd.Series(clean_str)
    
    if(vectorizer == 'Sentence_Transformer_cleaned_LR' or vectorizer == 'Sentence_Transformer_cleaned_RF'):
        test_str = test_str.apply(create_spacy_tokens)
    
    #Encode using sentence transformer
    X_test = sent_transformer_model.encode(test_str)
    
    # Make prediction using model loaded from disk as per the data.
    prediction = rf.predict(X_test)

    # Take the first value of prediction
    print("prediction :: ",prediction)
    output = prediction[0]

    return str(output)

if __name__ == '__main__':
    try:
        app.run(port=5000, debug=True)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")