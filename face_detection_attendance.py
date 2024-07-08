import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
from datetime import datetime
import face_recognition
 
#path = r'D:\College Materials\SEM-06\Embedded System\Project Attendance\ATTENDANCE\collected images'
#path = r'D:\College Materials\SEM-06\Embedded System\Project Attendance\ATTENDANCE\datasetImages'

path = r'D:\College Materials\SEM-06\Embedded System\Project Attendance\ATTENDANCE\final_dataset'
url='http://192.168.103.109/cam-hi.jpg'
##'''cam.bmp / cam-lo.jpg /cam-hi.jpg / cam.mjpeg '''

if 'Attendance.csv' in os.listdir(os.path.join(os.getcwd(),'attendance')):
    print("there iss..")
    os.remove("Attendance.csv")
else:
    df=pd.DataFrame(list())
    df.to_csv("Attendance.csv")
    
 
images = []
classNames = []     # list of student's name_regnum ...
myList = os.listdir(path)
#print(myList)              # list of file address..
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')  # fetching image from the folder ...
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0]) ##splitting the name from .jpg ....

print("\nTotal students registered in the class: ",len(classNames) )
print("\nRegistered Students are :\n")
for i in classNames:
    print("\t",i)

print("\nPlease wait. ENCODING the images...")

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
 
def markAttendance(name):
    with open("Attendance.csv", 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name.split("_")[0]},{name.split("_")[1]},{dtString}')
                
 
 
encodeListKnown = findEncodings(images)
print('\nEncoding Complete')  # Encoding the fetched images ...

print("-------------------------------------------------------------------")
#cap = cv2.VideoCapture(0)

present_student = [] # list of present student...
print("\nPresentees are :")
while True:
    #success, img = cap.read()
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgnp,-1)
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
 
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDis)
        matchIndex = np.argmin(faceDis)
 
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            if name not in present_student:
                print("\t",name)
                present_student.append(name)
                markAttendance(name)
 
    cv2.imshow('Webcam', img)
    key=cv2.waitKey(5)
    if key==ord('q'):
        # print(present_student)
        break
    
cv2.destroyAllWindows()
cv2.imread
