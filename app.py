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
    return render_template('index.html')


app.run()
