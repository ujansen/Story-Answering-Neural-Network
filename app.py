from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import pickle

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from bAbi_SingleSupporting import tokenize, vectorize_stories, single_predict
# Define a flask app
app = Flask(__name__)

from keras.models import load_model
model = load_model('query_model.h5')
vocab = pickle.load(open("vocab.pkl", "rb"))
vocabulary = ''
for i in range(2, len(vocab)-1):
    vocabulary += vocab[i] + ', '
vocabulary += 'and ' + vocab[len(vocab) - 1]
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html', vocabulary = vocabulary)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        story = request.form.get('story')
        query = request.form.get('question')

        # Make prediction
        prediction = single_predict(story, query)
        #preds = predict_single(filepath, model)
        #return preds
        
        return render_template('index.html', prediction_text = prediction, vocabulary = vocabulary)
    return None


if __name__ == '__main__':
    app.run(debug=True)