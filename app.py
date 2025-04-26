# app.py
from flask import Flask, render_template, request
import numpy as np
import pickle
from visualizations import create_visuals

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
le_dict = pickle.load(open("label_encoders.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    visuals = create_visuals()
    input_data = {}

    if request.method == "POST":
        fields = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
                  'basement', 'hotwaterheating', 'airconditioning', 'parking',
                  'prefarea', 'furnishingstatus']

        data = []
        for field in fields:
            value = request.form[field]
            input_data[field] = value
            if field in le_dict:
                value = le_dict[field].transform([value])[0]
            else:
                value = int(value)
            data.append(value)

        prediction = round(model.predict(np.array(data).reshape(1, -1))[0], 2)

    return render_template("index.html", prediction=prediction, visuals=visuals, input_data = {key: "" for key in ['area', 'bedrooms', 'bathrooms', 'stories',
                                  'mainroad', 'guestroom', 'basement', 'hotwaterheating',
                                  'airconditioning', 'parking', 'prefarea', 'furnishingstatus']})

if __name__ == "__main__":
    app.run(debug=True)
