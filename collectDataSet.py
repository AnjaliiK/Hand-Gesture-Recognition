import cv2
import os
import mediapipe as mp
import os
from gtts import gTTS



def createAudio(classes):
    for class_ in classes:

        tts = gTTS(text=class_, lang='en')
        tts.save(class_+".mp3")
        
DATA_DIR = r'C:\Users\kansu\OneDrive\Desktop\Custom-HandGesture-Recognition\data'


class_labels = ["Dislike", "Goodjob", "Hello", "Iloveyou", "OK", "peace"]

createAudio(class_labels)

if len(os.listdir(DATA_DIR)) == 0:

        os.makedirs(os.path.join(DATA_DIR, class_labels))

classFolders = os.listdir(DATA_DIR)

cap = cv2.VideoCapture(0)
numberOfImages = 500

for classFolder in classFolders:
    imgCounter = 0
    while True:
        success, frame = cap.read()
        cv2.putText(frame, f'Press S to start capturing images to {classFolder} folder', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Press Esc to exit', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('Frame', frame)
        k = cv2.waitKey(25)
        if k == ord('s') or k == 27:
            break

    if k == 27:     
        break
    
    while imgCounter < numberOfImages:
        success, frame = cap.read()
        cv2.imshow('Frame', frame)
        k = cv2.waitKey(25)
        imgSavePath = os.path.join(DATA_DIR, classFolder, f'{classFolder}_{imgCounter}.jpg')
        cv2.imwrite(imgSavePath, frame)
        print(imgCounter)
        imgCounter += 1


cap.release()
cv2.destroyAllWindows()