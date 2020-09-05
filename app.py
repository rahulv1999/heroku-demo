import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('ML.pkl','rb'))

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    
    #to render result in html GUI
     features = [int(x) for x in request.form.values()]
     finalfeatures = np.array(features)
     finalfeatures =  finalfeatures.reshape(1,-1)
     prediction = model.predict(finalfeatures)
     output = round(prediction[0],2)
     
     return render_template('index.html', prediction_text = "Salary is : {}".format(output))
     
 
    
if __name__ == "__main__":
    app.run(debug=True)
