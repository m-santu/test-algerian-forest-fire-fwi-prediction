from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application


## import ridge_cv and standard scler pkl
ridge_model = pickle.load(open('models/ridge_cv.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))



@app.route('/', methods=['GET', 'POST'])
def index() :
    if request.method == 'POST' : 
        print('got post request')
        Temperature = int(request.form.get('Temperature'))
        RH = int(request.form.get('RH'))
        Ws = int(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = int(request.form.get('Classes'))
        Region = int(request.form.get('Region'))

        inputs = [Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]
        if None in inputs :
            raise ValueError('None is not acceptanle')

        
        new_data_scaled = standard_scaler.transform([inputs])
        predict = ridge_model.predict(new_data_scaled)

        return render_template('index.html', result=predict[0])


    return render_template('index.html')


if __name__ == '__main__' :
    app.run()