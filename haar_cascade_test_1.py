import numpy as np
import cv2

user_name = input("Please write your name : ")
user_id = input("Please write your id : ")
detected_data = cv2.CascadeClassifier('opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

count = 0
interrupt_flag = 0 
while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1) #좌우반전
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detected_data.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=9, 
        minSize=(20, 20)
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
        count+=1 
        cv2.imwrite("userdata/User_"+str(user_id)+'_'+str(user_name)+'_'+str(count)+'.jpg', gray[y:y+h,x:x+w])
        cv2.imshow('image', img) 
    k = cv2.waitKey(50) & 0xff 
    if k == 27: 
        interrupt_flag = 1
        break
    elif count >= 200: 
        break

if interrupt_flag == 1:
    print("\nFinish by interrupt ESC.\n")
else :
    print("\nComplete to save data.\n")
cap.release()
cv2.destroyAllWindows()