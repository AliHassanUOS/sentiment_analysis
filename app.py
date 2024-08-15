# app.py
from flask import Flask, render_template, request, jsonify
import joblib
from model import preprocess_text
from config import Config
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address


app = Flask(__name__)
app.config.from_object(Config)

# Load the trained model
model = joblib.load('sentiment_model.pkl')
print(model)
@app.route('/')
def home():
    return render_template('index.html')

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)



@app.route('/predict', methods=['POST'])
@limiter.limit("1 per minute")  # Rate limit for this endpoint
def predict():
    if request.method == 'POST':
        text = request.form['text']
        cleaned_text = preprocess_text(text)
        prediction = model.predict([cleaned_text])[0]
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        return render_template('index.html', prediction=sentiment)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    text = data.get('text', '')
    cleaned_text = preprocess_text(text)
    prediction = model.predict([cleaned_text])[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
