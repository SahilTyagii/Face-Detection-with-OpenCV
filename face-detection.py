# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 01:04:09 2024

@author: Sahil
"""

# Importing library cv2
import cv2

# Loading cascades
face_cascade = cv2.CascadeClassifier('C:\\Users\\Sahil\\Desktop\\Projects\\Face-Detection-with-OpenCV\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\Sahil\\Desktop\\Projects\\Face-Detection-with-OpenCV\\haarcascade_eye.xml')
