from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
car = pd.read_csv("cleaned2 car.csv")

# Load your ML model (replace with your model file)
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_model = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    return render_template("index.html", companies=companies, car_model=car_model, year=year, fuel_type=fuel_type)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    company = data['company']
    car_model = data['car_model']
    year = int(data['year'])
    fuel_type = data['fuel']
    driven = int(data['killo_driven'])

    # Prediction input formatting
    input_df = pd.DataFrame([[car_model, company, year, driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    prediction = model.predict(input_df)[0]
    return jsonify({'price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
