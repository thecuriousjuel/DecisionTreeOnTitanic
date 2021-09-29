from flask import Flask, request, render_template
import joblib


app = Flask(__name__)


@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pclass = request.form['pclass']

    print(pclass)

if __name__ == '__main__':
    app.run(debug=True)

