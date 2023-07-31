import numpy as np
import cv2
import dlib
import timeit

#혹시 작동 안되면 ctrl+shift+p 해서 인터프리터가 base로 되어있는지 확인!

faceCascade = cv2.CascadeClassifier('opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor('opencv-master\data\shape_predictor_68_face_landmarks\shape_predictor_68_face_landmarks.dat')

# 얼굴의 각 구역의 포인트들을 구분해 놓기
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))


def detect(gray,frame):
    # 일단, 등록한 Cascade classifier 를 이용 얼굴을 찾음
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)  #sclaFactor=?

    # 얼굴에서 랜드마크를 찾자
    for (x, y, w, h) in faces:
        # 오픈 CV 이미지를 dlib용 사각형으로 변환하고
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # 랜드마크 포인트들 지정
        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
        # 원하는 포인트들을 넣는다, 지금은 전부
        landmarks_display = landmarks[0:68]
        # 눈만 = landmarks_display = landmarks[RIGHT_EYE_POINTS, LEFT_EYE_POINTS]

        # 포인트 출력
        for idx, point in enumerate(landmarks_display):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)

    return frame

# 웹캠에서 이미지 가져오기
video_capture = cv2.VideoCapture(0)

while True:
    # 웹캠 이미지를 프레임으로 자름
    ret, frame = video_capture.read()
    frame=cv2.flip(frame,1) #반전
    if ret is True:
        # 알고리즘 시작 시점
        start_t = timeit.default_timer()
        
        """ 알고리즘 연산 """
        # 그리고 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 만들어준 얼굴 눈 찾기
        canvas = detect(gray, frame)
        """ 알고리즘 연산 """
        
        # 알고리즘 종료 시점
        terminate_t = timeit.default_timer()
        
        FPS = int(1./(terminate_t - start_t ))
        #avg_frame_rate = (sum(FPS) / len(FPS))
       

         # 프레임 수를 문자열에 저장
        str = "FPS : %0.1f" % FPS
        
         # 프레임 속도 표시
        cv2.putText(frame, str, (0, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0))
       
        cv2.imshow("haha", canvas)
        print(FPS)   


    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 끝
video_capture.release()
cv2.destroyAllWindows()

