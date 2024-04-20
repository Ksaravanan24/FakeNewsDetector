# from flask import Flask, render_template
# app = Flask(__name__)


# @app.route('/')
# def helloword():
#     return render_template('home.html')


# app.run()
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Dataset/Dataset.csv")
print(data.shape)
data = data.dropna()
print(data.shape)


def preprocess(data):
    # label encode target variable
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])
    # tokenize the description
    tokenizer = torch()
    tokenizer.fit_on_texts(data['text'])
    sequences = tokenizer.texts_to_sequences(data['text'])
    X = pad_sequence(sequences, maxlen=500)
    # get target variable
    y = data['label']
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0)
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = preprocess(data)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
