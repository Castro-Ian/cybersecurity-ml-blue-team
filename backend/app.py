from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import openai
import os
from dotenv import load_dotenv
import logging

# Load environment variables from the .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask setup
app = Flask(__name__)
CORS(app)

# OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define the Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20, 128)  # assuming 20 features
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = Net()

# Load pre-trained model weights (if available)
try:
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    logger.info("Model loaded successfully.")
except FileNotFoundError:
    logger.warning("Pre-trained model not found. Make sure to train the model.")

# Data Preprocessing
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array(data['input']).reshape(1, -1)
    input_data = preprocess_data(input_data, None)
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
    result = 'Malicious' if predicted.item() == 1 else 'Benign'
    return jsonify({'prediction': result})

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Cybersecurity Threat Intelligence: {message}",
        max_tokens=150
    )
    return jsonify({'response': response.choices[0].text.strip()})

if __name__ == '__main__':
    app.run(debug=True)
