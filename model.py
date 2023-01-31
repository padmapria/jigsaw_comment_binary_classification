# Simple Logistic Regression /RandomForestClassifier

'''
This model predicts if a comment is toxic or not using logistic regression / random forest model.
'''

#Importing  our module where commonly used functions are given
from common import *
    
# Importing the dataset
df = pd.read_csv('data/toxic_data_2000.csv')

##We only need comment_text and toxic label for our model
df = df[['comment_text', 'toxic']]
df['comment_text'] = df['comment_text'].str[:255]


print("Calling clean data $$$$$$$")
#Type1 
df['comment_text_cleaned'] = df['comment_text'].apply(clean_data)
df['comment_sent_encoder_embeddings'] = df['comment_text_cleaned'].apply(sent_transformer_model.encode)


print("Calling spacy_tokens *****")
#Type2 
df['comment_tokens'] = df['comment_text_cleaned'].apply(create_spacy_tokens)
df['comment_sent_encoder_embeddings_clean'] = df['comment_tokens'].apply(sent_transformer_model.encode)

print("Train Test split *****")
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
y_label= df['toxic']
X_label = df[['comment_sent_encoder_embeddings','comment_sent_encoder_embeddings_clean']]
X_train, X_test, y_train, y_test = train_test_split(X_label, y_label, test_size=0.33, stratify=y_label)


x_train_transformer = X_train.comment_sent_encoder_embeddings.to_list()
x_test_transformer = X_test.comment_sent_encoder_embeddings.to_list()

x_train_transformer_1 = X_train.comment_sent_encoder_embeddings_clean.to_list()
x_test_transformer_1 = X_test.comment_sent_encoder_embeddings_clean.to_list()


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

print("Training model *****")
def train_test_LR(X_train, y_train,X_test,y_test,best_C,vectorizer):
    clf = LogisticRegression(C=best_C,class_weight='balanced')
    clf.fit(X_train, y_train)
    
    #save the model for future use
    model_name = vectorizer+".pkl"
    write_pickle(clf, model_name)
    
    clf = read_pickle(model_name)
    predicted = clf.predict(X_test)
    print("Results for the vectorizer ::", vectorizer)
    print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
    print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
    print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))
    print()

def train_test_RF(X_train, y_train,  X_test,y_test,max_depth,n_estimators, vectorizer):
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=max_depth,
                                       n_estimators=n_estimators)
    rf.fit(X_train,y_train)
    
    model_name = vectorizer+".pkl"
    write_pickle(rf, model_name)
    
    rf = read_pickle(model_name)
    predicted = rf.predict(X_test)
    print("Results for the vectorizer ::", vectorizer)
    print("RF Accuracy:",metrics.accuracy_score(y_test, predicted))
    print("RF Precision:",metrics.precision_score(y_test, predicted))
    print("RF Recall:",metrics.recall_score(y_test, predicted))
    print()
    
train_test_LR(x_train_transformer, y_train,x_test_transformer,y_test,1,"Sentence_Transformer_LR")
train_test_LR(x_train_transformer_1, y_train,x_test_transformer_1,y_test,1,"Sentence_Transformer_cleaned_LR") 
train_test_RF(x_train_transformer, y_train,x_test_transformer,y_test, max_depth=9, n_estimators=500,vectorizer="Sentence_Transformer_RF")
train_test_RF(x_train_transformer_1, y_train,x_test_transformer_1,y_test, max_depth=8, n_estimators=1000,vectorizer="Sentence_Transformer_cleaned_RF")