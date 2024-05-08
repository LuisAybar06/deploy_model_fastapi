# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Houses import House
import numpy as np
import pandas as pd
from joblib import load
from pyngrok import ngrok
import nest_asyncio

# 2. Create the app object
app = FastAPI()

# 3. Load the model
classifier = load("linear_regression.joblib")
X_train = pd.read_csv('xtrain.csv')  # Load X_train data
features = pd.read_csv('selected_features.csv')
features = features['0'] .to_list() 

X_train = X_train[features]

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}


# 4. Expose the prediction functionality, make a prediction using X_train data
#    and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote():
    predictions = classifier.predict(X_train)
    return {
        'predictions': predictions.tolist()
    }

    # if(prediction[0]>11.14984174816577):
    #     prediction="Bad selection"
    # else:
    #     prediction="Approved"
    # return {
    #     'prediction': prediction
    # }

# 5. Expose to public url
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
