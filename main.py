from flask import Flask, redirect, url_for, render_template, request, jsonify, flash
from werkzeug.utils import secure_filename
import pandas as pd
import uuid
import os

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'xlsx'}
BASE_URL = "http://127.0.0.1:5000/"

UPLOAD_FOLDER = 'data_upload'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    idupload = uuid.uuid4()
    file = request.files['txtFile']
    if file and allowed_file(file.filename):
        filename = secure_filename(str(idupload)+".xlsx")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
    dr = {"status" : "success", "kdProses":idupload}
    # return jsonify(dr)
    return redirect('/mapping-data-training/'+str(idupload))


@app.route('/mapping-data-training/<token>')
def mappingDataTraining(token):
    dataTraining = pd.read_excel("data_upload/"+str(token)+".xlsx")
    dtnp = dataTraining.to_numpy()
    dataTemp = []
    ord = 1

    for x in dtnp:
        dSatuan = {}
        dSatuan['ord'] = ord
        dSatuan['judul'] = x[0]
        dSatuan['judul_url'] = x[1]
        dSatuan['judul_isi'] = x[2]
        dSatuan['label'] = x[3]
        dataTemp.append(dSatuan)
        ord += 1

    # print(dataTraining)


    # dr = {"status" : "success", "token":token}
    return render_template('mapping-data-training.html', dTraining=dataTemp)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)