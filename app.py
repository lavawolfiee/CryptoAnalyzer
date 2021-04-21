import random
import string
from flask import Flask, jsonify, request, redirect, send_from_directory, \
    render_template
import logging
import os

import predictor

app = Flask(__name__)


@app.route('/favicon.ico')
def favicon():
    """Handles browser's request for favicon"""
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico'
    )


@app.route('/', methods=['GET'])
def get():
    """This triggers when you first open the site with your browser"""
    return render_template('index.html', symbols=predictor.symbols,
                           intervals=predictor.intervals)


@app.route('/predict', methods=['GET'])
def predict():
    filename = ''.join(random.choices(string.ascii_lowercase + string.digits,
                                      k=10)) + '.svg'
    window = int(request.args.get('window', default=50))
    predict_steps = int(request.args.get('steps', default=10))
    symbol = str(request.args.get('symbol', default='BTCUSDT'))
    interval = str(request.args.get('interval', default='1h'))

    data, prediction, percentiles = predictor.predict(window, predict_steps,
                                                      symbol, interval)
    predictor.plot_prediction(data, prediction, percentiles, filename=filename)
    return 'results/' + filename


@app.route('/results/<filename>')
def results(filename=''):
    return send_from_directory(
        os.path.join(app.root_path, 'results'),
        filename
    )


app.run()
