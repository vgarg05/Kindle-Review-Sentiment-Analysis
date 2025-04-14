# 📚 Kindle Review Sentiment Analysis

This project analyzes Kindle book reviews and classifies them as **Positive** or **Negative** using Natural Language Processing (NLP) and a Naive Bayes classifier. 
The application includes a web interface built with Flask, allowing users to input reviews and instantly see the sentiment.

🔗 **Live Demo:** [Click here to try it](https://your-demo-link.com)

---

## 🚀 Features

- Cleaned and preprocessed real-world Amazon Kindle reviews
- Sentiment classification using both BOW and TFIDF vectorization
- Naive Bayes classifier for efficient prediction
- Live web app built with Flask for real-time analysis
- Custom text preprocessing pipeline
- User-friendly interface

---

## 🧠 Model Details

- **Text Vectorization**: CountVectorizer (BoW) and TFIDF
- **Algorithm**: Multinomial Naive Bayes
- **Accuracy**: ~85% on test set, 59% overall

---

## 🛠️ Technologies Used

- Python (Numpy, Pandas, Scikit-learn, NLTK)
- Flask (Backend API)
- HTML,CSS (Frontend)
- Pickle (Model serialization)
- Jupyter Notebook (Model training and evaluation)

---

## 📁 Folder Structure
├── models/
│   └── nb_model_tfidf.pkl          # Pickled Naive Bayes model trained on TFIDF features
│   └── tfidf_vectorizer.pkl        # Pickled TfidfVectorizer used for TFIDF transformation           
│   └── nb_model_bow.pkl            # Pickled Naive Bayes model trained on BOW features
│   └── bow_vectorizer.pkl          # Pickled CountVectorizer used for BOW transformation
├── notebook/
│   └── kindle sentimental review analysis.ipynb   # Jupyter notebook for data analysis & model training
│   └── all_kindle_review.csv        # Dataset used for model training
├── templates/
│   └── index.html         # Home page
├── app.py                 # Flask backend for the web app 
├── requirements.txt       # Python dependencies
└── README.md              # Project overview

