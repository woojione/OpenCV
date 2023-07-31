import numpy as np
import cv2
import dlib

faceCascade=cv2.CascadeClassifier('opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
predictor=dlib.shape_predictor('opencv-master\data\shape_predictor_68_face_landmarks\shape_predictor_68_face_landmarks.dat')

#얼굴 각 구역의 포인트들을 구분
JAWLINE_POINTS=list(range(8,17))
RIGHT_EYEBROW_POINTS=list(range(17,22))
LEFTS_EYEBROW_POINTS=list(range(22,27))
NOSE_POINTS=list(range(27,36))
RIGHT_EYE_POINTS=list(range(36,42))
LEFT_EYE_POINTS=list(range(42,48))
MOUTH_OUTLINE_POINTS=list(range(48,61))
MOUTH_INNER_POINTS=list(range(61,68))

def detect(gray,frame):
    #cascade를 이용하여 얼굴 찾음
    faces=faceCascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5,minSize=(100,100),flags=cv2.CASCADE_SCALE_IMAGE)
    
    #얼굴에서 landmark 찾기
    for(x,y,w,h) in faces:
        #opencv이미지->dlib용 사각형으로 변환
        dlib_rect=dlib.rectangel(int(x),int(y),int(x+w),int(y+h))
        #landmark point 지정
        landmarks=np.martrix([[p.x,p.y] for p in predictor(frame,dlib_rect).parts()])
        #원하는 포인트 지정
        landmarks_display=landmarks[0:68] #0~68=all

        #landmark point 출력
        for idx, point in enumerate(landmarks_display):
            pos=(point[0,0],point[0,1])
            cv2.circle(frame, pos, 2, color=(0,255,255), thickness=-1)
           
    return frame

video_capture=cv2.VideoCapture(0)

while True:
    #웹캠이키지를 프레임으로 자름
    _,frame=video_capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    canvas=detect(gray,frame)
    cv2.imshow("smile",canvas)

    #q를 누르면 종료
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()