from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
   
@app.route('/predict', methods=['POST'])
def predict():
    mdl = pickle.load(open("mdl.pkl", "rb"))
    cv = pickle.load(open("cv.pkl", "rb"))

    if request.method=='POST':
        text=request.form['message']
        # data=[comment]
        vector = cv.transform([text]);
        pvalue = mdl.predict(vector);
    predict = "SPAM"
    if(pvalue == 0): predict = "NOT SPAM"
    # return str(my_prediction)
    return render_template('index.html', prediction='ENTERED MESSAGE IS {}'.format(predict))

if __name__== '__main__':
    app.run(debug=True)