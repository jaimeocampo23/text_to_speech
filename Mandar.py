# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 02:07:17 2020

@author: jaime calderon
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import h5py
import pyttsx3

global rec
TrainedM = tf.keras.models.load_model('IA_Model.h5')
def Hablar(Ceros):
    rec = ' '
    for i in range (Ceros.shape[0]):
        Respu=TrainedM.predict(np.asmatrix(Ceros[i, :]))
        if (np.argmax(Respu) == 0):
            rec += 'A' 
        elif(np.argmax(Respu) == 1):
            rec += 'B' 
        elif(np.argmax(Respu) == 2):
            rec += 'C' 
        elif(np.argmax(Respu) == 3):
            rec += 'D' 
        elif(np.argmax(Respu) == 4):
            rec += 'E'
        elif(np.argmax(Respu) == 5):
            rec += 'F' 
        elif(np.argmax(Respu) == 6):
            rec += 'G' 
        elif(np.argmax(Respu) == 7):
            rec += 'H' 
        elif(np.argmax(Respu) == 8):
            rec += 'I' 
        elif(np.argmax(Respu) == 9):
            rec += 'J' 
        elif(np.argmax(Respu) == 10):
            rec += 'K'
        elif(np.argmax(Respu) == 11):
            rec += 'L' 
        elif(np.argmax(Respu) == 12):
            rec += 'M' 
        elif(np.argmax(Respu) == 13):
            rec += 'N' 
        elif(np.argmax(Respu) == 14):
            rec += 'O' 
        elif(np.argmax(Respu) == 15):
            rec += 'P'
        elif(np.argmax(Respu) == 16):
            rec += 'Q' 
        elif(np.argmax(Respu) == 17):
            rec += 'R' 
        elif(np.argmax(Respu) == 18):
            rec += 'S' 
        elif(np.argmax(Respu) == 19):
            rec += 'T' 
        elif(np.argmax(Respu) == 20):
            rec += 'U' 
        elif(np.argmax(Respu) == 21):
            rec += 'V'
        elif(np.argmax(Respu) == 22):
            rec += 'W' 
        elif(np.argmax(Respu) == 23):
            rec += 'X' 
        elif(np.argmax(Respu) == 24):
            rec += 'Y' 
        elif(np.argmax(Respu) == 25):
            rec += 'Z' 
        elif(np.argmax(Respu) == 26):
            rec += 'a'
        elif(np.argmax(Respu) == 27):
            rec += 'b' 
        elif(np.argmax(Respu) == 28):
            rec += 'c' 
        elif(np.argmax(Respu) == 29):
            rec += 'd' 
        elif(np.argmax(Respu) == 30):
            rec += 'e' 
        elif(np.argmax(Respu) == 31):
            rec += 'f' 
        elif(np.argmax(Respu) == 32):
            rec += 'g'
        elif(np.argmax(Respu) == 33):
            rec += 'h'
        elif(np.argmax(Respu) == 34):
            rec += 'i' 
        elif(np.argmax(Respu) == 35):
            rec += 'j' 
        elif(np.argmax(Respu) == 36):
            rec += 'k' 
        elif(np.argmax(Respu) == 37):
            rec += 'l'
        elif(np.argmax(Respu) == 38):
            rec += 'm' 
        elif(np.argmax(Respu) == 39):
            rec += 'n' 
        elif(np.argmax(Respu) == 40):
            rec += 'o' 
        elif(np.argmax(Respu) == 41):
            rec += 'p' 
        elif(np.argmax(Respu) == 42):
            rec += 'q' 
        elif(np.argmax(Respu) == 43):
            rec += 'r'
        elif(np.argmax(Respu) == 44):
            rec += 's'
        elif(np.argmax(Respu) == 45):
            rec += 't' 
        elif(np.argmax(Respu) == 46):
            rec += 'u' 
        elif(np.argmax(Respu) == 47):
            rec += 'v' 
        elif(np.argmax(Respu) == 48):
            rec += 'w'
        elif(np.argmax(Respu) == 49):
            rec += 'x' 
        elif(np.argmax(Respu) == 50):
            rec += 'y' 
        elif(np.argmax(Respu) == 51):
            rec += 'z' 
        elif(np.argmax(Respu) == 52):
            rec += 'á' 
        elif(np.argmax(Respu) == 53):
            rec += 'é' 
        elif(np.argmax(Respu) == 54):
            rec += 'í'
        elif(np.argmax(Respu) == 55):
            rec += 'ó'
        elif(np.argmax(Respu) == 56):
            rec += 'ú' 
        elif(np.argmax(Respu) == 57):
            rec += ', ' 
        else:
            rec += '.   ' 
        
    print(rec)
    engine = pyttsx3.init()
    engine.say(rec)
    engine.runAndWait()