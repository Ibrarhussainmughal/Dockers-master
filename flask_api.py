

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
#Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict')
def predict_note_authentication():
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted values is"+ str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
   df_test=pd.read_csv(request.files.get('file'))
   prediction=classifier.predict(df_test)
   return "The predicted values for the csv"+ str(list(prediction))




if __name__=='__main__':
    app.run()
    
    