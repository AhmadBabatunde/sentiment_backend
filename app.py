from flask import Flask, request, jsonify
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
import numpy as np

# Load model and tokenizer
model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Set up Flask app
app = Flask(__name__)

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    input_text = data['text']

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors='tf', truncation=True, padding=True, max_length=128)

    # Get model output
    outputs = model(inputs)
    logits = outputs.logits
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]

    # Determine label and confidence
    label = 'positive' if np.argmax(probabilities) == 1 else 'negative'
    confidence = float(np.max(probabilities))

    return jsonify({
        'label': label,
        'confidence': round(confidence, 4)
    })

if __name__ == '__main__':
    app.run(debug=True)
