from flask import Flask, render_template, request
import joblib

# Load the trained machine learning model
app = Flask(__name__)
model = joblib.load('crop_app')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Define the route for the index page
@app.route('/index')
def index():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the input data from the form
        features = [float(x) for x in request.form.values()]

        # Make prediction using the loaded model
        prediction = model.predict([features])[0]

        # Render the prediction template with the result
        return render_template('prediction.html', prediction=prediction)
    else:
        # Handle GET request (if needed)
        return render_template('index.html')  # For example, render a form for input

if __name__ == '__main__':
    app.run(debug=True)