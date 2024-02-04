# Testing

import cv2 as cv
import numpy as np
import face_recognition
from datetime import datetime

imgBabar = face_recognition.load_image_file('Images/babar azam.jpg')
imgBabar = cv.cvtColor(imgBabar, cv.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('Test Images/babar test.jpg')
imgTest = cv.cvtColor(imgTest, cv.COLOR_BGR2RGB)

cv.imshow("",imgBabar)
cv.imshow("",imgTest)
cv.waitKey(0)

faceLoc = face_recognition.face_locations(imgBabar)[0]
encodeBabar = face_recognition.face_encodings(imgBabar)[0]

print(faceLoc) # T R B L
print(encodeBabar)
cv.rectangle(imgBabar,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]), (255,0,255),2)

cv.imshow('',imgBabar)
cv.waitKey(0)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]), (255,0,255),2)

cv.imshow('',imgTest)
cv.waitKey(0)

results = face_recognition.compare_faces([encodeBabar],encodeTest)
faceDis = face_recognition.face_distance([encodeBabar],encodeTest)
print(results, faceDis)

cv.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
cv.imshow('',imgTest)
cv.waitKey(0)