import numpy as np
import dlib
import cv2
from functools import wraps
from scipy.spatial import distance
import time
import timeit

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 68))
EYES = list(range(36, 48))
lib = [60,61,63,64,65,67]

def calculate_EAR(eye): # 눈 거리 계산
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

def calculate_EAR_lib(lib): # 입 거리 계산
	A = distance.euclidean(lib[1], lib[5])
	B = distance.euclidean(lib[2], lib[4])
	C = distance.euclidean(lib[0], lib[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

# 카메라 셋팅
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# dlib 인식 모델 정의
hog_face_detector = dlib.get_frontal_face_detector()
predictor_file = 'opencv-master\data\shape_predictor_68_face_landmarks\shape_predictor_68_face_landmarks.dat'
dlib_facelandmark = dlib.shape_predictor(predictor_file)
lastsave = 0


def counter(func):
    @wraps(func)
    def tmp(*args, **kwargs):
        tmp.count += 1
        time.sleep(0.05)
        global lastsave
        if time.time() - lastsave > 5:
            lastsave = time.time()
            tmp.count = 0
        return func(*args, **kwargs)
    tmp.count = 0
    return tmp

@counter
def close():
    cv2.putText(frame,"DROWSY",(20,100), cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret is True:
        # 알고리즘 시작 시점
        start_t = timeit.default_timer()
        
        """ 알고리즘 연산 """
        
        
        """ 알고리즘 연산 """
        
        # 알고리즘 종료 시점
        terminate_t = timeit.default_timer()
        
        FPS = int(1./(terminate_t - start_t ))

       

         # 프레임 수를 문자열에 저장
        str = "FPS : %0.1f" % FPS
    


    faces = hog_face_detector(gray)
    
    face_num = len(faces)
    if face_num < 1:
            close()
            print(f'close count : {close.count}')
            if close.count == 15:
                print("Driver is distraction")
    

    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []
        lib_11=[]

        for n in range(36,42): # 오른쪽 눈 감지
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in range(42,48): # 왼쪽 눈 감지
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

        for n in lib: # 입 감지
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            lib_11.append((x,y))
            next_point = n+1
            if n == 67:
                next_point = 60
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)    


        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        lib_ear = calculate_EAR_lib(lib_11)

        EAR = (left_ear+right_ear)/2
        EAR = round(EAR,2)
        
        if EAR<0.19:
            close()
            print(f'close count : {close.count}')
            if close.count == 15:
                print("Driver is sleeping")
        # print(EAR)
        
        if lib_ear>0.27:
            close()
            print(f'close count : {close.count}')
            if close.count == 15:
                print("Driver is sleeping")
        print(lib_ear)
        
        
        
    cv2.putText(frame,str,(0,100),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0))
    cv2.imshow("Are you Sleepy", frame)
    

    key = cv2.waitKey(30)
    if key == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
# https://wjh2307.tistory.com/21