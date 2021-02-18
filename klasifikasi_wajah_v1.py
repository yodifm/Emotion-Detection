# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
#untuk mengimport algoritma yang digunakan (svm)
from sklearn import metrics, datasets
from sklearn.utils import Bunch
from sklearn.svm import SVC
#untuk mengimport holdout validation
from sklearn.model_selection import GridSearchCV, train_test_split
#untuk mengimport pengolah gambar
from skimage.io import imread
from skimage.transform import resize
import skimage
import pickle
import cv2
import os
from scipy import ndimage
from scipy.ndimage.filters import convolve
import matplotlib.image as mpimg
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter
import imutils

# Class modul klasifikasi
class klasifikasi_mod(object):
    def rgb_ke_gray(rgb):
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    def sobel_filters(citra):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
        Ix = ndimage.filters.convolve(citra, Kx)
        Iy = ndimage.filters.convolve(citra, Ky)
        G = np.hypot(Ix, Iy)
        R = G / G.max() * 255
        return R
    def gaussian_filter(ukuran, sigma=1.4):
        ukuran = int(ukuran) // 2
        x, y = np.mgrid[-ukuran:ukuran+1, -ukuran:ukuran+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g
    def load_file_citra(path, dimensi=(64, 64)):
        
        image_dir = Path(path)
        folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
        categories = [fo.name for fo in folders]
        descr = "Dataset Citra Untuk Klasifikasi"
        images = []
        flat_data = []
        target = []
        for i, direc in enumerate(folders):
            for file in direc.iterdir():
                citra = skimage.io.imread(file)
                citra_gray=klasifikasi_mod.rgb_ke_gray(citra)
                citra_smooth=convolve(citra_gray, klasifikasi_mod.gaussian_filter(5, 1.4))
                citra_gradient=klasifikasi_mod.sobel_filters(citra_smooth)
                citra_resized = resize(citra_gradient, dimensi, anti_aliasing=True, mode='reflect')
                flat_data.append(citra_resized.flatten()) 
                images.append(citra_gradient)
                target.append(categories[i])            
        flat_data = np.array(flat_data)
        target = np.array(target)
        images = np.array(images)
        return Bunch(data=flat_data,
                     target=target,
                     target_names=categories,
                     DESCR=descr)
    def training(citra_dataset):
        X_train, X_test, y_train, y_test = train_test_split(citra_dataset.data, citra_dataset.target, test_size=0.4,random_state=0)
        target_names = citra_dataset.target_names
        svc = OneVsRestClassifier(SVC(kernel='linear',C=1,probability=True))
        svc.fit(X_train, y_train)
        # simpan parameter svm
        f = open("output/pengklasifikasi.pickle", "wb")
        f.write(pickle.dumps(svc))
        f.close()
        # simpan kelas yang digunakan
        f = open("output/kelas.pickle", "wb")
        f.write(pickle.dumps(target_names))
        f.close()
        y_pred = svc.predict(X_test)
        jml_data="Jumlah Data yang digunakan :\n"+ str(len(citra_dataset.data))
        jml_data_train="Jumlah Data Train: \n"+ str(len(y_train))+" "+ str(Counter(y_train))
        jml_data_test="Jumlah Data Test: \n"+ str(len(y_test))+" "+ str(Counter(y_test))
        precision="Prec: {0:.0f}%".format(metrics.precision_score(y_test, y_pred,average='macro')*100)
        recall="Rec: {0:.0f}%".format(metrics.recall_score(y_test, y_pred,average='macro')*100)
        accuracy="Acc:{0:.0f}%".format(metrics.accuracy_score(y_test, y_pred)*100)
        return jml_data,jml_data_train,jml_data_test,accuracy,precision,recall
    def simpan_dataset(dataset):
        citra_dataset=dataset
        f = open("output/citra_dataset.pickle", "wb")
        f.write(pickle.dumps(citra_dataset))
        f.close()
    def klasifikasi(fileName):
        # membuka data model dan pembaca model dari penyimpanan
        proto_file = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
        model_file = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
        pendeteksi_wajah = cv2.dnn.readNetFromCaffe(proto_file, model_file)
        # membuka model yang telah dilatih sebelumnya dengan svm dan nama target klasifikasi
        pengklasifikasi = pickle.loads(open("output/pengklasifikasi.pickle", "rb").read())
        kode_label = pickle.loads(open("output/kelas.pickle", "rb").read())
        #membuka citra yang akan diklasifikasi dan mengubah ukuran citra agar dapat dibaca oleh model deteksi
        citra = cv2.imread(fileName)
        citra = imutils.resize(citra, width=600)
        (h, w) = citra.shape[:2]
        # membuat blob citra dari citra yang diinput
        blob_citra = cv2.dnn.blobFromImage(cv2.resize(citra, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
        #melakukan deteksi wajah dengan model dan pembaca model (CNN CAFFE FRAMEWORK)
        pendeteksi_wajah.setInput(blob_citra)
        wajah_terdeteksi = pendeteksi_wajah.forward()
        # melakukan pendeteksian berulang pada citra untuk menemukan wajah pada setiap citra
        for i in range(0, wajah_terdeteksi.shape[2]):
            # melakukan validasi terkait kesalahan prediksi pada suatu wajah
            validasi_wajah = wajah_terdeteksi[0, 0, i, 2]
            # melakukan penyaringan hasil deteksi wajah yang lemah
            if validasi_wajah > 0.5:
                #membuat kotak untuk wajah berdasarkan koordinat (x,y)setelah wajah terdeteksi 
                kotak = wajah_terdeteksi[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = kotak.astype("int")
                # mengambil citra wajah (crop)
                wajah = citra[startY:endY, startX:endX]
                (tw, lw) = wajah.shape[:2]
                # memastikan ukuran citra wajah yang terambil 
                if lw < 20 or tw < 20:
                    continue
                # melakukan ekstraksi feature dengan sobel edge detection (onvolution) dan ubah ukuran ke 64*64
                citra_gray=klasifikasi_mod.rgb_ke_gray(wajah)
                citra_smooth=convolve(citra_gray, klasifikasi_mod.gaussian_filter(5, 1.4))
                citra_sobel=klasifikasi_mod.sobel_filters(citra_smooth)
                citra_sobel_resized= resize(citra_sobel, (64,64), anti_aliasing=True, mode='reflect')
                # melakukan flat array untuk dilakukan klasifikasi
                flat_citra=[]
                flat_citra.append(citra_sobel_resized.flatten())
                # melakukan klasifikasi wajah setelah dilakukan ekstraksi
                persentase_prediksi = pengklasifikasi.predict_proba(flat_citra)[0]
                j = np.argmax(persentase_prediksi)
                kemungkinan = persentase_prediksi[j]
                nama_pemilik_wajah =kode_label[j]
                # memberikan kotak pada gambar yang diinputkan pada bagian citra wajah serta
                #menyantumkan label hasil klasifikasi beserta persentase probability
                text = "{}: {:.2f}%".format(nama_pemilik_wajah, kemungkinan * 100)
        # menampilkan gambar hasil klasifikasi
        cv2.imwrite('output/wajah_terdeteksi.jpg',wajah)
        cv2.imwrite('output/wajah_sobel.jpg',citra_sobel)
        return text
        
# Class UI pengenalan wajah dengan citra
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 620)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(10, 50, 980, 201))
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(10, 10, 61, 16))
        self.label_2.setObjectName("label_2")
        self.edit_file = QtWidgets.QLineEdit(self.frame)
        self.edit_file.setGeometry(QtCore.QRect(80, 10, 151, 20))
        self.edit_file.setObjectName("edit_file")
        self.btn_load = QtWidgets.QPushButton(self.frame)
        self.btn_load.setGeometry(QtCore.QRect(150, 40, 81, 23))
        self.btn_load.setObjectName("btn_load")
        self.btn_train = QtWidgets.QPushButton(self.frame)
        self.btn_train.setGeometry(QtCore.QRect(60, 70, 81, 23))
        self.btn_train.setObjectName("btn_train")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(250, 10, 81, 16))
        self.label_3.setObjectName("label_3")
        self.lbl_lapor = QtWidgets.QListWidget(self.frame)
        self.lbl_lapor.setGeometry(QtCore.QRect(250, 30, 700, 151))
        palette = QtGui.QPalette()
        self.lbl_lapor.setPalette(palette)
        self.lbl_lapor.setFrameShape(QtWidgets.QFrame.Box)
        self.lbl_lapor.setObjectName("lbl_lapor")
        self.btn_pilihfile = QtWidgets.QPushButton(self.frame)
        self.btn_pilihfile.setGeometry(QtCore.QRect(60, 40, 81, 23))
        self.btn_pilihfile.setObjectName("btn_pilihfile")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 141, 31))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(10, 260, 161, 21))
        font = QtGui.QFont()
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(10, 290, 980, 261))
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.lbl_citramasuk = QtWidgets.QLabel(self.frame_2)
        self.lbl_citramasuk.setGeometry(QtCore.QRect(10, 20, 211, 201))
        self.lbl_citramasuk.setFrameShape(QtWidgets.QFrame.Box)
        self.lbl_citramasuk.setText("")
        self.lbl_citramasuk.setObjectName("lbl_citramasuk")
        self.label_7 = QtWidgets.QLabel(self.frame_2)
        self.label_7.setGeometry(QtCore.QRect(80, 0, 81, 16))
        self.label_7.setObjectName("label_7")
        self.btn_pilihcitra = QtWidgets.QPushButton(self.frame_2)
        self.btn_pilihcitra.setGeometry(QtCore.QRect(230, 20, 75, 23))
        self.btn_pilihcitra.setObjectName("btn_pilihcitra")
        self.btn_klasifikasi = QtWidgets.QPushButton(self.frame_2)
        self.btn_klasifikasi.setGeometry(QtCore.QRect(230, 50, 75, 23))
        self.btn_klasifikasi.setObjectName("btn_klasifikasi")
        self.btn_lihathasil = QtWidgets.QPushButton(self.frame_2)
        self.btn_lihathasil.setGeometry(QtCore.QRect(230, 80, 75, 23))
        self.btn_lihathasil.setObjectName("btn_lihathasil")
        self.lbl_citrawajah = QtWidgets.QLabel(self.frame_2)
        self.lbl_citrawajah.setGeometry(QtCore.QRect(320, 20, 191, 181))
        self.lbl_citrawajah.setFrameShape(QtWidgets.QFrame.Box)
        self.lbl_citrawajah.setText("")
        self.lbl_citrawajah.setObjectName("lbl_citrawajah")
        self.label_9 = QtWidgets.QLabel(self.frame_2)
        self.label_9.setGeometry(QtCore.QRect(370, 0, 91, 16))
        self.label_9.setObjectName("label_9")
        self.lbl_citrasobel = QtWidgets.QLabel(self.frame_2)
        self.lbl_citrasobel.setGeometry(QtCore.QRect(520, 20, 191, 181))
        self.lbl_citrasobel.setFrameShape(QtWidgets.QFrame.Box)
        self.lbl_citrasobel.setText("")
        self.lbl_citrasobel.setObjectName("lbl_citrasobel")
        self.label_11 = QtWidgets.QLabel(self.frame_2)
        self.label_11.setGeometry(QtCore.QRect(550, 0, 131, 16))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.frame_2)
        self.label_12.setGeometry(QtCore.QRect(750, 0, 47, 16))
        self.label_12.setObjectName("label_12")
        self.lbl_nama = QtWidgets.QLabel(self.frame_2)
        self.lbl_nama.setGeometry(QtCore.QRect(740, 20, 120, 18))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_nama.setFont(font)
        self.lbl_nama.setText("")
        self.lbl_nama.setObjectName("lbl_nama")
        self.btn_keluar = QtWidgets.QPushButton(self.centralwidget)
        self.btn_keluar.setGeometry(QtCore.QRect(900, 560, 85, 23))
        self.btn_keluar.setObjectName("btn_keluar")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 872, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.btn_pilihcitra.clicked.connect(self.setImage)
        self.btn_pilihfile.clicked.connect(self.setFile)
        self.btn_load.clicked.connect(self.load_file)
        self.btn_train.clicked.connect(self.training_data)
        self.btn_klasifikasi.clicked.connect(self.klasifikasi_data)
        self.btn_lihathasil.clicked.connect(self.lihat_hasil)
        self.btn_keluar.clicked.connect(self.keluar)
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pengenalan Wajah (Citra) | Created by Syarif"))
        self.label_2.setText(_translate("MainWindow", "File Dataset"))
        self.btn_load.setText(_translate("MainWindow", "Load Dataset"))
        self.btn_train.setText(_translate("MainWindow", "Train Data"))
        self.label_3.setText(_translate("MainWindow", "Laporan"))
        self.btn_pilihfile.setText(_translate("MainWindow", "Cari File"))
        self.label.setText(_translate("MainWindow", "Training Data"))
        self.label_4.setText(_translate("MainWindow", "Klasifikasi Citra"))
        self.label_7.setText(_translate("MainWindow", "Citra Masukan"))
        self.btn_pilihcitra.setText(_translate("MainWindow", "Pilih Citra"))
        self.btn_klasifikasi.setText(_translate("MainWindow", "Klasifikasi"))
        self.btn_lihathasil.setText(_translate("MainWindow", "Lihat Hasil"))
        self.label_9.setText(_translate("MainWindow", "Wajah Terdeteksi"))
        self.label_11.setText(_translate("MainWindow", "Hasil Edge Detection Sobel"))
        self.label_12.setText(_translate("MainWindow", "Nama"))
        self.btn_keluar.setText(_translate("MainWindow", "Keluar"))
    def setImage(self):
        global fileName
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Pilih Citra", "", "Image Files (*.png *.jpg *jpeg *.bmp);;All Files (*)") # Ask for file
        if fileName: 
            pixmap = QtGui.QPixmap(fileName) 
            pixmap = pixmap.scaled(self.lbl_citramasuk.width(), self.lbl_citramasuk.height(), QtCore.Qt.KeepAspectRatio) # Scale pixmap
            self.lbl_citramasuk.setPixmap(pixmap) 
            self.lbl_citramasuk.setAlignment(QtCore.Qt.AlignCenter)        
    def setFile(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(None, "Pilih Folder") # Ask for file
        self.edit_file.setText('{}'.format(folder))
        info_sel ="[INFO] Path data :"
        self.lbl_lapor.addItem(info_sel)
        self.lbl_lapor.addItem(str(folder))
    def load_file(self):
        path = self.edit_file.text()
        citra_dataset=klasifikasi_mod.load_file_citra(path)
        klasifikasi_mod.simpan_dataset(citra_dataset)
        info_sel ="[INFO] Proses Load Data Selesai"
        self.lbl_lapor.addItem(info_sel)
    def training_data(self):
        citra_dataset = pickle.loads(open("output/citra_dataset.pickle", "rb").read())
        laporan= klasifikasi_mod.training(citra_dataset)
        info_sel ="[INFO] Laporan Hasil Training"
        self.lbl_lapor.addItem(info_sel)
        for i in range(len(laporan)):
            self.lbl_lapor.addItem(laporan[i])
        info_sel ="[INFO] Proses Training Data Selesai"
        self.lbl_lapor.addItem(info_sel)
    def klasifikasi_data(self):
        global hasil_klasifikasi
        path_citra=fileName
        hasil_klasifikasi = klasifikasi_mod.klasifikasi(path_citra)
        info_sel ="[INFO] hasil_klasifikasi : "+ hasil_klasifikasi
        self.lbl_lapor.addItem(info_sel)
    def lihat_hasil(self):
        self.lbl_nama.setText(hasil_klasifikasi)
        path_citra_wajah ="output/wajah_terdeteksi.jpg" 
        path_citra_sobel="output/wajah_sobel.jpg"
        if path_citra_wajah: 
            pixmap = QtGui.QPixmap(path_citra_wajah) 
            #pixmap = pixmap.scaled(self.lbl_citrawajah.width(), self.lbl_citrawajah.height(), QtCore.Qt.KeepAspectRatio) 
            self.lbl_citrawajah.setPixmap(pixmap) 
            self.lbl_citrawajah.setAlignment(QtCore.Qt.AlignCenter)
        if path_citra_sobel: 
            pixmap = QtGui.QPixmap(path_citra_sobel) 
            #pixmap = pixmap.scaled(self.lbl_citrasobel.width(), self.lbl_citrasobel.height(), QtCore.Qt.KeepAspectRatio) 
            self.lbl_citrasobel.setPixmap(pixmap) 
            self.lbl_citrasobel.setAlignment(QtCore.Qt.AlignCenter)
    def keluar(self):
        quit()
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    klasifikasi_mod()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
    