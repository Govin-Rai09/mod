import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
with open('classifier.pkl', 'rb') as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_feat = [int(x) for x in request.form.values()]
    fin_feat = [np.array(int_feat)]
    prediction = model.predict(fin_feat)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Note is real if 0 else fake {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
