import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=10, #탐지할 최대 얼굴 수
    refine_landmarks=True, #눈동자 주변의 랜드마크를 추가로 출력할지 여부. 기본값 false
    min_detection_confidence=0.5, #탐지가 성공한 것으로 간주되는 얼굴 탐지 모델의 최소 신뢰값[0.0,1.0] 기본값 0.5
    min_tracking_confidence=0.5) as face_mesh: #랜드마크 추적 모델의 최소 신뢰 값[0,0.1.0]이 성공적으로 추적된 것으로 간주되거나, 그렇지 않으면 다음 입력 영상에서 자동으로 얼굴 감지가 호출됨.
    #높은 값으로 설정하면 지연 시간이 길어지는 대신 솔루션의 정확성을 높일 수 있음. 기본값 0.5
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = True #기본값 False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1)) # 이미지 좌우반전
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()