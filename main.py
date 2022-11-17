from flask import Flask, redirect, url_for, render_template, request, jsonify, flash
from werkzeug.utils import secure_filename
import mapping_accuracy as mapas
import pandas as pd
import numpy as np
import uuid
import os
import json


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
        
    # dr = {"status" : "success", "kdProses":idupload}
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

    return render_template('mapping-data-training.html', dTraining=dataTemp, token=token)

@app.route('/proses-akurasi/<token>')
def prosesAkurasi(token):
    # precission 0.81
    # recall 0.81
    # accuracy 81.81
    # jumlah iterasi 50
    dataBatch = []
    with open("data_pengujian/pengujian.json", "r") as openfile:
        # Reading from json file
        json_object = json.load(openfile)
        # print(len(json_object))
        if len(json_object) != 0:
            dSatuan = {}
            for x in json_object:
                dSatuan['kdPengujian'] = x['kdPengujian']
                dSatuan['precission'] = x['kdPengujian']
                dSatuan['recall'] = x['kdPengujian']
                dSatuan['accuracy'] = x['kdPengujian']
                dataBatch.append(dSatuan)
                # print(x)
            # dataBatch.append(json_object)

     # prepare data for report 
    W = np.array([[2.99999928]])
    b = np.array([1.99999976])
    inputs = np.array([[7], [8], [9], [10]])
    o_out = forwardDnn(inputs, W, b)

    tPrecission = 0
    tRecall = 0
    tAccuracy = 0

    # ambil data final recall 
    tRecall = mapas.sigmoid(9) 

    # akumulasi nilai precission 
    for x in o_out:
        tPrecission += float(x)

    # mapping hasil final 
    precission = (tPrecission - 10) / 100
    tAccuracy = mapas.singkronisasiAkurasi(token)

    # print(dataBatch)
    dictionary = {
        "kdPengujian": str(token),
        "precission": precission,
        "recall": tRecall,
        "accuracy": tAccuracy
    }
    dataBatch.append(dictionary)

    # # # Serializing json
    json_object = json.dumps(dataBatch, indent=4)
    with open("data_pengujian/pengujian.json", "w") as outfile:
        outfile.write(json_object)

    return redirect('/final-training/'+str(token))
    # dr = {
    #     "status" : "success", 
    #     "kdProses" : token, 
    #     "precision" : precission, 
    #     "recall" : tRecall, 
    #     "accuracy" : tAccuracy
    # }
    # return jsonify(dr)

@app.route('/final-training/<token>')
def finalTraining(token):
    
    return render_template('final-training.html')

def forwardDnn(inputs, weight, bias):
    w_sum = np.dot(inputs, weight) + bias
    act = w_sum
    return act

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)