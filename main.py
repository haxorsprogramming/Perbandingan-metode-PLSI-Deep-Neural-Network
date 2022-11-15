from flask import Flask, redirect, url_for, render_template, request, jsonify

import os
app = Flask(__name__)

BASE_URL = "http://127.0.0.1:5000/"

@app.route('/')
def index():
    dr = {'BASE_URL': BASE_URL}
    return render_template('home.html', dRes=dr)

@app.route('/data-training')
def dataTraining():
    dr = {'BASE_URL': BASE_URL}
    return render_template('data-training.html', dRes=dr)

@app.route('/proses-data-training', methods=('GET', 'POST'))
def prosesDataTraining():
    dr = {"status" : "success"}
    return jsonify(dr)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)