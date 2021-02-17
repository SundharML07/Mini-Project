import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
from joblib import load
app = Flask(__name__)
model = load("RF Classifier")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test = [[x for x in request.form.values()]]
    print(x_test)
    x_test[0][0]=int(x_test[0][0])
    x_test[0][1]=int(x_test[0][1])
    x_test[0][2]=int(x_test[0][2])
    x_test[0][3]=int(x_test[0][3])
    x_test[0][4]=int(x_test[0][4])
    x_test[0][5]=int(x_test[0][5])
    x_test[0][6]=float(x_test[0][6])
    x_test[0][7]=float(x_test[0][7])
    x_test[0][8]=float(x_test[0][8])
    x_test[0][9]=float(x_test[0][9])
    x_test[0][10]=float(x_test[0][10])
    x_test[0][11]=float(x_test[0][11])
    x_test[0][12]=float(x_test[0][12])
    x_test[0][13]=float(x_test[0][13])
    x_test[0][14]=float(x_test[0][14])
    #x=pd.DataFrame(StandardScaler().fit_transform(x_test))
    
    
    prediction = model.predict(x_test)
    print(prediction)
    output=prediction[0]
    if(output==0):
        pred="The Patient's Heart is in Healthy Condition"
    elif(output==1):
        pred="The Patient has Heart Disease, Please Consult a Doctor" 
    
    return render_template('index.html', prediction_text='{}'.format(pred))

'''@app.route('/predict_api',methods=['POST'])
def predict_api():
    
   # For direct API calls trought request

    data = request.get_json(force=True)
    prediction = model.y_predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)'''

if __name__ == "__main__":
    app.run(debug=True)
