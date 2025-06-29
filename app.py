from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

#Loading models
model = joblib.load('avocado_ripeness_model.joblib')
scaler = joblib.load('scaler.joblib')
color_encoder = joblib.load('color_encoder.joblib')
ripeness_encoder = joblib.load('ripeness_encoder.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collecting Inputs
        values = [float(request.form[k]) for k in ['firmness', 'hue', 'saturation', 'brightness', 'sound_db', 'weight', 'size']]
        color = request.form['color']
        
        # Preprocessing
        X_num = scaler.transform([values])
        X_cat = color_encoder.transform([color]).reshape(-1, 1)
        X_final = np.hstack((X_num, X_cat))

        # Prediction 
        y_pred = model.predict(X_final)[0]
        result = ripeness_encoder.inverse_transform([y_pred])[0]
        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
