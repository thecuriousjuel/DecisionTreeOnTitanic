from flask import Flask, request, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
# Set maximum content length for file uploads
app.config['MAX_CONTENT_LENGTH'] = 10 * 1000 * 1000

@app.route('/', methods=['GET'])
def homepage():
    """
    Render the homepage.

    Returns:
        Rendered HTML template for the homepage.
    """
    return render_template('index.html')

@app.route('/predict_data', methods=['POST'])
def predict_data():
    """
    Handle POST requests to predict survival based on form data.

    Extracts data from the form, creates a DataFrame, preprocesses the data,
    and makes a prediction using the pre-trained model.

    Returns:
        Rendered HTML template with the prediction result.
    """
    # Extract form data
    pclass = request.form['pclass']
    sex = request.form['sex']
    age = request.form['age']
    sibsp = request.form['sibsp']
    parch = request.form['parch']
    fare = request.form['fare']
    embarked = request.form['embarked']

    # Create DataFrame from form data
    df = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

    # Preprocess data and make prediction
    output = preprocess_and_model(df)

    # Render output template with prediction result
    return render_template('output.html', data=df.values, columns=df.columns, output=output)

@app.route('/predict_file', methods=['POST'])
def predict_file():
    """
    Handle POST requests to predict survival based on uploaded CSV file.

    Reads the CSV file, preprocesses the data, and makes a prediction using the pre-trained model.

    Returns:
        Rendered HTML template with the prediction result.
    """
    # Get uploaded file
    file = request.files['file']

    # Check if the file is a CSV
    if file.filename.rsplit('.')[-1] not in ['csv', 'CSV']:
        return render_template('index.html', output='Invalid file. Allowed files [CSV]')

    # Read CSV file into DataFrame
    df = pd.read_csv(file)

    # Preprocess data and make prediction
    output = preprocess_and_model(df)
    
    # Render output template with prediction result
    return render_template('output.html', data=df.values, columns=df.columns, output=output)

def preprocess_and_model(df):
    """
    Preprocess the input DataFrame and make a prediction using the pre-trained model.

    Args:
        df (pd.DataFrame): Input data.

    Returns:
        np.ndarray: Prediction result.
    """
    # Define feature names
    features_name = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    # Select relevant features
    df = df[features_name].copy()

    # Load pre-trained transformers and model
    full_transformer = joblib.load('model/full_transformer.pickle')
    tree_clf = joblib.load('model/tree_clf.pickle')

    # Preprocess data
    prepared_data = full_transformer.transform(df)

    # Make prediction
    prediction = tree_clf.predict(prepared_data)

    return prediction

if __name__ == '__main__':
    # Run Flask app in debug mode
    app.run(debug=False)

