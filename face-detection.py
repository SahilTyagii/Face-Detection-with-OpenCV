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

#This function will be used for detection
def detect(gray, frame):
    # Detects faces in grayscale image and returns tuple of  rectangle with x, y, w, h its cordinates and dimensions
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # 1.3 and 5 are scaleFactor and minNeighbour respectively
    for (x, y, w, h) in faces: # iterating through the faces rectangles
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Drawing rectangle around the face
        region_of_intrest_gray = gray[y:y+h, x:x+w] # creating grayscale and color subimage to detect eyes from the faces rectangle
        region_of_intrest_color= frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(region_of_intrest_gray, 1.1, 3) # detecting eyes and returning rectangles
        for (X, Y, W, H) in eyes:
            cv2.rectangle(region_of_intrest_color, (X, Y), (X+W, Y+H), (0, 255, 0), 2)
    return frame