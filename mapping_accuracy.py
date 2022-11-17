import pandas as pd
import numpy as np

DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = LETTERS + DIGITS

def softmax(a):
  exps = np.exp(a.astype(np.float64))
  return exps / np.sum(exps, axis=-1)[:, np.newaxis]

def sigmoid(a):
  return 1. / (1. + np.exp(-a))

def singkronisasiAkurasi(token):
  dataTraining = pd.read_excel("data_upload/"+str(token)+".xlsx")
  dtnp = dataTraining.to_numpy()
  ord = 1
  akurasi = 0
  accFakta = 0
  accHoax = 0

  for x in dtnp:
    isi = x[2]
    kelas = x[3]
    if "tidak benar" in isi.lower() or "tidak sesuai" in isi.lower() or "menyesatkan" in isi.lower():
      accHoax += 1
    if "dipastikan" in isi.lower() or "benar terjadi" in isi.lower() or "fakta" in isi.lower() or "sumber" in isi.lower():
      accFakta += 1

    ord += 1

  akurasi = accFakta + accHoax
  hasilAkurasi = (akurasi / ord) * 100

  return hasilAkurasi
