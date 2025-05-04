from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.exception import CustomException

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('home.html')
        # Get thepip list 
        # input data from the form
    else:
        data = CustomData(
            country=request.form.get('Country'),
            Gender=request.form.get('Gender'),
            Age=request.form.get('Age'),    
            Marital_status=request.form.get('Marital status'),
            Stratum_Urban=request.form.get('Stratum Urban'),
            Natur_of_work=request.form.get('Natur of work'),
            Level_of_Wealth=request.form.get('Level of Wealth'),
            Fathers_level_education=request.form.get('Fathers level education'),
            Parent_affiliated_with_SS=request.form.get('Parent affiliated with SS'),
            Participation_in_elections=request.form.get('Participation in elections'),
            Freedom_to_speach_out=request.form.get('Freedom to speach out'),
            Resort_to_nepotism=request.form.get('Resort to nepotism'),
            Trust_in_Parliment=request.form.get('Trust in Parliment'),
            Trust_in_employers=request.form.get('Trust in employers'),
            Trust_in_associations=request.form.get('Trust in associations'),
            Trust_in_political_parties=request.form.get('Trust in political parties'),
            Political_system=request.form.get('Political system'))
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        # Convert numerical prediction to label (1 = formal, 0 = informal)
        prediction_label = "Formal" if results[0] == 1 else "Informal"
        return render_template('home.html', results=prediction_label)
    
if __name__ == "__main__":
    app.run(host = '0.0.0.0',port = 5000,debug = True)            
            
            
        
