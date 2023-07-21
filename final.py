import cv2
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import time 
import os

#Carregnado o modelo treinado

model = load_model('./modelo_02_expressoes.h5')

arquivo_video = "video_filme.mp4"
cap = cv2.VideoCapture(arquivo_video)
conectado, video = cap.read()
print(conectado, video.shape)

redimencionar = True
largura_max = 600
largura = video.shape[1]
altura = video.shape[0]


if (redimencionar & video.shape[1] > largura_max):
    proporcao = largura/altura
    video_largura = largura_max
    video_altura = int(video_largura/proporcao)
else:
    video_altura = altura
    video_largura = largura


nome_video = 'video_analisado.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 24
saida_video = cv2.VideoWriter(nome_video, fourcc, fps, (video_largura, video_altura))


haarcascade_faces = './haarcascade_frontalface_default.xml'
fonte_pequena, fonte_media = 0.4, 0.7
fonte = cv2.FONT_HERSHEY_SIMPLEX
expressoes = ['raiva', 'nojo', 'medo', 'feliz', 'triste', 'surpreso', 'neutro']


while (cv2.waitKey(1) < 0 ):
    conectado, frame = cap.read()
    if not conectado:
        break
    
    t = time.time()

    if redimencionar:
        frame = cv2.resize(frame, (video_largura, video_altura))
    
    face_cascade = cv2.CascadeClassifier(haarcascade_faces)
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_cinza, 1.3, 5)

    if len(faces) > 0:
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (255,50,50), 2)
            roi_cinza = frame_cinza[y:y+h, x:x+w]
            roi_cinza = cv2.resize(roi_cinza, (48,48), interpolation=cv2.INTER_AREA)
            roi_cinza = roi_cinza.astype('float')/255.0
            roi_cinza = img_to_array(roi_cinza)
            roi_cinza = np.expand_dims(roi_cinza, axis=0)
            preds = model.predict(roi_cinza)[0]
            print(preds)
    if preds is not None:
        resultado = np.argmax(preds)
        cv2.putText(frame, expressoes[resultado], (x, y-10), fonte, fonte_media, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.putText(frame, 'Tempo: {:.2f}'.format(time.time()-t), (10, 20), fonte, fonte_pequena, (255,255,255), 1, cv2.LINE_AA)
    cv2.imshow('Video', frame)
    saida_video.write(frame)
    print(time.time()-t)

