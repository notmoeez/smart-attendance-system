import cv2 as cv
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Images'
images = []
classNames = []

# get all files (images) from the "Images" directory
myList = os.listdir(path)
# print(myList)

# append all saved images into images array
for cl in myList:
  curImg = cv.imread(f'{path}/{cl}')
  images.append(curImg)
  # append all class names (names of students) into classNames array
  classNames.append(os.path.splitext(cl)[0])

# print(classNames)

# function to get the encodings (128 dimensional face encodings) of all available images (student faces)
def findEncodings(images):
  encodeList = []
  for img in images:
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    encode = face_recognition.face_encodings(img)[0]
    encodeList.append(encode)
  return encodeList

# function to mark attendance (csv file -> interface) of the student whose face is detected through the webcam
def markAttendance(name):
    file_path = 'Attendance Sheet.csv'

    # 'a+' append and read mode
    with open(file_path, 'a+') as f:
        f.seek(0)  # Move the cursor to the beginning of the file
        lines = f.readlines()

        # extracting all names from the csv file
        name_list = [line.split(',')[0] for line in lines]

        # if name is not already marked names then mark the attendance
        if name not in name_list:
            now = datetime.now()
            dt_string = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{dt_string}')
            print(f"Attendance marked for {name} at {dt_string}")
        else:
            print(f"{name} already marked present")

# save all encodings of student faces from available images into an array
encodeListKnown = findEncodings(images)
# print(len(encodeListKnown))
print("Encoding Completed")

# start recognition through device webcam
cap = cv.VideoCapture(0)

while True:
    # get data frame by frame through webcam
  success, img = cap.read()
    # resize the captured image frames
  imgSm = cv.resize(img,(0,0),None, 0.25,0.25)
    # convert BGR image to RGB
  imgSm = cv.cvtColor(imgSm, cv.COLOR_BGR2RGB)

    # detect face location in the image (frame) using Histogram of Oriented Gradients (HOG)
  facesCurFrame = face_recognition.face_locations(imgSm)
    # encode the located face features
  encodesCurFrame = face_recognition.face_encodings(imgSm,facesCurFrame)

    # check each encoded values of available student faces with the current captured face
  for encodeFace, faceLoc in zip(encodesCurFrame,facesCurFrame):
      # returns list of True and False values
    matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
      # get euclidean distance of current face with all available faces based on encoded values
    faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
    print(faceDis)
      # get index of closest match (min value from face distances array)
    matchIndex = np.argmin(faceDis)

        # if match found, get name from classnames array and display through webcam with bounded box around detected face
    if matches[matchIndex]:
        name = classNames[matchIndex].upper()
        # print(name)
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = 4 * y1, 4 * x2, 4 * y2, 4 * x1
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
        cv.putText(img, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        markAttendance(name)

  cv.imshow("Web Cam",img)
  cv.waitKey(1)
