from flask import Flask, request,render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
application = Flask(__name__)

app = application

# Route to home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Elevation=float(request.form.get('Elevation')),
            Aspect = float(request.form.get('Aspect')),
            Slope = float(request.form.get('Slope')),
            Horizontal_Distance_To_Hydrology=float(request.form.get('Horizontal_Distance_To_Hydrology')),
            Vertical_Distance_To_Hydrology=float(request.form.get('Vertical_Distance_To_Hydrology')),
            Horizontal_Distance_To_Roadways=float(request.form.get('Horizontal_Distance_To_Roadways')),
            Hillshade_9am=float(request.form.get('Hillshade_9am')),
            Hillshade_Noon=float(request.form.get('Hillshade_Noon')),
            Hillshade_3pm=float(request.form.get('Hillshade_3pm')),
            Horizontal_Distance_To_Fire_Points=float(request.form.get('Horizontal_Distance_To_Fire_Points')),
            Wilderness_Area=int(request.form.get('Wilderness_Area')),
            Soil_Type=int(request.form.get('Soil_Type'))
        )
        model_name = request.form.get('Model')
        features=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(features,model_name)
        result_dict = {
                    1:"Spruce/Fir",
                    2:"Lodgepole Pine",
                    3:"Ponderosa Pine",
                    4:"Cottonwood/Willow",
                    5:"Aspen",
                    6:"Douglas-fir",
                    7:"Krummholz"
        }
        return render_template('home.html',results=result_dict[results[0]])
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug = True)