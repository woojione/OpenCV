import cv2
import numpy as np
from PIL import Image
import os

path = 'userdata'
detector = cv2.CascadeClassifier("opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create() 

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,file) for file in os.listdir(path)]
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') 
        img_numpy = np.array(PIL_img,'uint8')
        id = int(os.path.split(imagePath)[-1].split("_")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples, ids
print("\nPlease wait for a second...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

recognizer.write('trainer/trainer.yml')
print("\n {0} faces trained.\n".format(len(np.unique(ids))))