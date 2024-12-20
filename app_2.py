# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Houses import House
import pandas as pd
from joblib import load

 
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

