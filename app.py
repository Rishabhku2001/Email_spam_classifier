from flask import Flask, render_template, request
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

app = Flask(__name__)

ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y.copy()
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y.copy()
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def hello_world():
  return render_template('home.html')

@app.route("/apply",methods=['post'])
def recommendations():
  data = dict(request.form)['Email']
  transformed_email = transform_text(data)
  vector_input = tfidf.transform([transformed_email])
  return render_template('output.html',data=model.predict(vector_input)[0])








if __name__ == "__main__":
  app.run(host='0.0.0.0', debug=True)