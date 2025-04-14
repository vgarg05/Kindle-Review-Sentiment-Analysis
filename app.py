from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('models/nb_model_tfidf.pkl', 'rb'))
vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))

# Preprocessing function
def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub('[^a-zA-Z0-9 ]', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = BeautifulSoup(text, "lxml").get_text()
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    cleaned_review = clean_text(review)
    vectorized_review = vectorizer.transform([cleaned_review]).toarray()
    prediction = model.predict(vectorized_review)
    sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"
    return render_template('index.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=4070,debug=True)
