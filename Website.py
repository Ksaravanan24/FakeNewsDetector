from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])

# def predict():
#     if request.method == 'POST':
#         comment = request.form['comment']
        
#         # Tokenize and preprocess comment
#         count_vectorizer = CountVectorizer()
#         tfidf_transformer = TfidfTransformer()
#         X_counts = count_vectorizer.transform([comment])
#         X_tfidf = tfidf_transformer.transform(X_counts)
        
#         # Load pre-trained classifier
#         filename = 'model.pkl'
#         classifier = pickle.load(open(filename, 'rb'))
        
#         # Make prediction
#         my_prediction = classifier.predict(X_tfidf)
        
#         # Map prediction to 'Real' or 'Fake'
#         prediction = 'Real' if my_prediction[0] == 0 else 'Fake'
        
#         return render_template('result.html', prediction=prediction)

# Prediction function
def predict():
     if request.method == 'POST':
        comment = request.form['comment']
        # Load pre-trained classifier and vectorizer
        with open('model.pkl', 'rb') as model_file:
            classifier = pickle.load(model_file)

        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)

        # Tokenize and preprocess comment

        comment_transformed = vectorizer.transform([comment])

        # Make prediction
        my_prediction = classifier.predict(comment_transformed)

        # Map prediction to 'Real' or 'Fake'
        prediction = 'Real' if my_prediction[0] == 0 else 'Fake'

        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
