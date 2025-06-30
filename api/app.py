from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
clf = joblib.load('../model/classifier.pkl')
vectorizer = joblib.load('../model/vectorizer.pkl')

# Load phone specs
phone_specs = pd.read_csv('../data/phone_specs.csv')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    query = data.get('query', '')
    price = data.get('price', '')
    brand = data.get('brand', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    combined = f"{query} {price} {brand}"
    query_vec = vectorizer.transform([combined])
    category = clf.predict(query_vec)[0]

    # Filter recommendations by category, price, and brand if provided
    filtered = phone_specs[phone_specs['category'] == category]
    if price:
        # Example: handle "<300", "300-400"
        if '-' in price:
            low, high = map(int, price.split('-'))
            filtered = filtered[(filtered['price'] >= low) & (filtered['price'] <= high)]
        elif price.startswith('<'):
            high = int(price[1:])
            filtered = filtered[filtered['price'] <= high]
        elif price.startswith('>'):
            low = int(price[1:])
            filtered = filtered[filtered['price'] >= low]
    if brand:
        filtered = filtered[filtered['name'].str.contains(brand, case=False)]

    recommendations = filtered['name'].tolist()

    return jsonify({'category': category, 'recommendations': recommendations})

if __name__ == '__main__':
    app.run(port=8000)
