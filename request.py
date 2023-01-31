import requests

# URL
url = 'http://127.0.0.1:5000/api/'

# Change the comment_text and vectorizer that you want to test
#The different vectorizers to be used are 
'''
"Sentence_Transformer_LR", 
"Sentence_Transformer_cleaned_LR",
"Sentence_Transformer_RF"
"Sentence_Transformer_cleaned_RF"
'''
payload = {
	'comment_text':'Y to kiss my ass you guys sicken me. Ja rule is about pride in da music',
    'vectorizer':'Sentence_Transformer_LR'
}

r = requests.post(url,json=payload)

print(r.json())