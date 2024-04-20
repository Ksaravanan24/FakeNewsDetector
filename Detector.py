import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load the data
data = pd.read_csv("E:/Code/Code/Dataset/fake_and_real_news.csv")

# Drop rows with missing values
data = data.dropna()

# Preprocessing function
def preprocess(data):
    # Label encode target variable
    le = LabelEncoder()
    data['label'] = le.fit_transform(data['label'])

    # Convert text data into numerical vectors
    count_vectorizer = CountVectorizer()
    X_counts = count_vectorizer.fit_transform(data['Text'])

    # Transform the count matrix to a normalized tf-idf representation
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    # Get target variable
    y = data['label']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.33, random_state=0)

    return X_train, X_test, y_train, y_test, count_vectorizer

# Preprocess the data
X_train, X_test, y_train, y_test, count_vectorizer = preprocess(data)
print(data)

# Train the classifier
classifier = MLPClassifier(random_state=0, max_iter=200)
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
clreport = classification_report(y_test, y_pred)
Tacc = "Accuracy on training set: {:.2f}".format(classifier.score(X_train, y_train))
Testacc = "Accuracy on test set: {:.3f}".format(classifier.score(X_test, y_test))
cm = confusion_matrix(y_test, y_pred)
print(Tacc)
print(Testacc)
# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of MLP Classifier')
plt.show()

# Save the model
model_filename = 'model.pkl'
pickle.dump(classifier, open(model_filename, 'wb'))

# Save the CountVectorizer
vectorizer_filename = 'vectorizer.pkl'
pickle.dump(count_vectorizer, open(vectorizer_filename, 'wb'))

print("succesfully model and vectorizer created")