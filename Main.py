# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:52:47 2020

@author: jaime calderon
"""

import numpy as np 
from skimage import io
import matplotlib.pyplot as plt 
import cv2
import tensorflow as tf
from tensorflow import keras
import h5py
import Mandar as M
global rec
# ------------------------ Se obtienen los patrones  --------------------------
'''
Letras = cv2.cvtColor(io.imread('Letras.bmp'), cv2.COLOR_BGR2GRAY)
Letras = 255 - Letras
Letras = Letras / np.max(Letras)
X, Y = Letras.shape

plt.figure(0)
plt.imshow(Letras, cmap = 'gray')
plt.show()

Vertical = []
for i in range (1, X):
    if sum(Letras[i, :]) != 0:
        if sum(Letras[i - 1, :]) == 0 or sum(Letras[i + 1, :]) == 0:
            Vertical.append(i)


Guardar = []
for i in range (1, Y):
    if sum(Letras[:, i]) != 0:
        if sum(Letras[:, i-1]) == 0 or sum(Letras[:, i+1]) == 0:
            Guardar.append(i)

Longitud_l = []
for i in range (0,len(Guardar),2):
    Datos = Letras[Vertical[0]:Vertical[1],Guardar[i]:Guardar[i+1]]
    A = np.reshape(Datos, [Datos.shape[0] * Datos.shape[1],1])
    Longitud_l.append(len(A))
    
C = max(Longitud_l)   #  323

suma = 0
Ceros = np.zeros((C,int(len(Guardar)/2))) 
for i in range (0, len(Guardar),2):
    a = Letras[Vertical[0]:Vertical[1],Guardar[i]:Guardar[i+1]]
    a = np.reshape(a, [a.shape[0] * a.shape[1],1])
    Ceros[0:len(a), suma] = a[0:len(a), 0]
    suma += 1
 
BDF_s = Ceros.T

np.save('Patrones', BDF_s)

Patrones = np.load('Patrones.npy')

# ----------------------------------  Keras -----------------------------------

Target = np.tile(np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,
                         18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,
                         38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,
                         58]),(1,1)).T

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(160,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(59,activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(Patrones,Target,epochs=500)  #train
loss, accuracy = model.evaluate(Patrones, Target)
print('Accuracy: %.2f' % accuracy)
print('Loss: %.2f' % loss)
model.save('IA_Model.h5')
model.save_weights('Pesos_FM.h5')

# ---------------------------  Prueba con imagen ---------------------------------

'''
Patrones = np.load('Patrones.npy')
TrainedM = tf.keras.models.load_model('IA_Model.h5')
Poema = cv2.cvtColor(io.imread('Poema.png'), cv2.COLOR_BGR2GRAY)  
Poema = 255 - Poema
Poema = Poema / np.max(Poema)

Vertical = []
for i in range (1, Poema.shape[0] - 2):
    if sum(Poema[i, :]) != 0:
        if sum(Poema[i - 1, :]) == 0 or sum(Poema[i + 1, :]) == 0:
            Vertical.append(i)

for k in range(0,len(Vertical),2):

    Guardar2 = []
    for i in range (1, Poema.shape[1]):
        if sum(Poema[Vertical[k]:Vertical[k + 1], i]) != 0:
            if sum(Poema[Vertical[k]:Vertical[k + 1], i-1]) == 0 or sum(Poema[Vertical[k]:Vertical[k + 1], i+1]) == 0:
                Guardar2.append(i)
      
    Dato = 0
    Ceros = np.zeros((int(len(Guardar2)/2),Patrones.shape[1])) 
    for i in range (0,len(Guardar2),2):
        C = Poema[Vertical[k]:Vertical[k + 1], Guardar2[i]:Guardar2[i+1]]
        Palabras = np.reshape(C, [1,C.shape[0] * C.shape[1]])
        Ceros[Dato, 0:Palabras.shape[1]] = Palabras[0, 0:Palabras.shape[1]]
        Dato += 1
       
    
    M.Hablar(Ceros)



