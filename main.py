import re
from flask import Flask, request, render_template
import joblib


app = Flask(__name__)


@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    pclass = request.form['pclass']
    sex = request.form['sex']
    age = request.form['age']
    sibsp = request.form['sibsp']
    parch = request.form['parch']
    fare = request.form['fare']
    embarked = request.form['embarked']
    
    

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

