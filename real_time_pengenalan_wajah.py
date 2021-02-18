from scipy.ndimage.filters import convolve
from skimage.transform import resize
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import pickle
import time
import cv2
import os

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
        return (R)
def gaussian_filter(ukuran, sigma=1.4):
        ukuran = int(ukuran) // 2
        x, y = np.mgrid[-ukuran:ukuran+1, -ukuran:ukuran+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g

#membuka data model dan pembaca model dari penyimpanan
proto_file = os.path.sep.join(["face_detection_model", "deploy.prototxt"])
model_file = os.path.sep.join(["face_detection_model", "res10_300x300_ssd_iter_140000.caffemodel"])
pendeteksi_wajah = cv2.dnn.readNetFromCaffe(proto_file, model_file)

# membuka model yang telah dilatih sebelumnya dengan svm dan nama target klasifikasi
pengklasifikasi = pickle.loads(open("output/pengklasifikasi.pickle", "rb").read())
kode_label = pickle.loads(open("output/kelas.pickle", "rb").read())

#membuka citra yang akan diklasifikasi dan mengubah ukuran citra agar dapat dibaca oleh model deteksi
#citra = cv2.imread("data_uji/edi1 - Copy.jpg")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# start the FPS throughput estimator
fps = FPS().start()

while True:
    # grab the frame from the threaded video stream
    citra = vs.read()
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
            citra_gray=rgb_ke_gray(wajah)
            citra_deteksi_sobel=sobel_filters(citra_gray)
            citra_sobel=citra_deteksi_sobel[0]
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
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(citra, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            cv2.putText(citra, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

    # update the FPS counter
    fps.update()

    # show the output frame
    cv2.imshow("Frame", citra)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
