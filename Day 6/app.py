from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
#@app.route('/test')
#def test():
#    return "Flask is being used"
@app.route('/')
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method=='POST':
        try:


            User_ID=float(request.form['User_ID'])
            Gender=float(request.form['Gender'])
            Age=float(request.form['Age'])
            EstimatedSalary=float(request.form['EstimatedSalary'])
            pred_args=[User_ID, Gender, Age, EstimatedSalary]
            pred_args_arr=np.array(pred_args)
            pred_args_arr=pred_args_arr.reshape(1, -1)
            dc_tree=open("Decision_tree_model.pkl", "rb")
            ml_model=joblib.load(dc_tree)
            model_prediction=ml_model.predict(pred_args_arr)
            model_prediction=round(float(model_prediction), 2)


        except valueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction=model_prediction)


if __name__=="__main__":
    app.run()
