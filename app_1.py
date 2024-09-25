# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Houses import House
from joblib import load

# 2. Create the app object
app = FastAPI()

# 3. Load the model
classifier = load("linear_regression.joblib")

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted house price
@app.post('/predict')
def predict_houseprice(data:House):
    MSSubClass=data.MSSubClass
    MSZoning=data.MSZoning
    LotArea=data.LotArea
    LotShape=data.LotShape
    LandContour=data.LandContour
    LotConfig=data.LotConfig
    Neighborhood=data.Neighborhood
    OverallQual=data.OverallQual
    OverallCond=data.OverallCond
    YearRemodAdd=data.YearRemodAdd
    RoofStyle=data.RoofStyle
    Exterior1st=data.Exterior1st
    ExterQual=data.ExterQual
    Foundation=data.Foundation
    BsmtQual=data.BsmtQual
    BsmtExposure=data.BsmtExposure
    BsmtFinType1=data.BsmtFinType1
    HeatingQC=data.HeatingQC 
    CentralAir=data.CentralAir 
    stFlrSF=data.stFlrSF 
    ndFlrSF=data.ndFlrSF 
    GrLivArea=data.GrLivArea 
    BsmtFullBath=data.BsmtFullBath 
    FullBath=data.FullBath 
    HalfBath=data.HalfBath 
    KitchenQual=data.KitchenQual 
    TotRmsAbvGrd=data.TotRmsAbvGrd 
    Functional=data.Functional 
    Fireplaces=data.Fireplaces 
    FireplaceQu=data.FireplaceQu 
    GarageFinish=data.GarageFinish 
    GarageCars=data.GarageCars 
    PavedDrive=data.PavedDrive 
    WoodDeckSF=data.WoodDeckSF 
    ScreenPorch=data.ScreenPorch 
    SaleCondition=data.SaleCondition 


    prediction = classifier.predict([[MSSubClass, MSZoning, LotArea, LotShape, LandContour, LotConfig, Neighborhood, OverallQual, OverallCond, YearRemodAdd,
     RoofStyle, Exterior1st, ExterQual, Foundation, BsmtQual, BsmtExposure, BsmtFinType1, HeatingQC, CentralAir, stFlrSF, ndFlrSF, GrLivArea, BsmtFullBath,
     FullBath, HalfBath, KitchenQual, TotRmsAbvGrd, Functional, Fireplaces, FireplaceQu, GarageFinish, GarageCars, PavedDrive, WoodDeckSF, ScreenPorch, SaleCondition]])

    return {
        'predictions': prediction.tolist()
    }

    # if(prediction[0]>11.14984174816577):
    #     prediction="Bad selection"
    # else:
    #     prediction="Approved"
    # return {
    #     'prediction': prediction
    # }
 