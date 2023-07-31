import cv2
import dlib
import sys
import numpy as np
#import matplotlib.pyplot as plt

scaler = 0.3

detector = dlib.get_frontal_face_detector()

# 머신 러닝으로 학습된 모델
predictor = dlib.shape_predictor('opencv-master\data\shape_predictor_68_face_landmarks\shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture('people.mp4')

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(
        img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))

    ori = img.copy()

    # detect faces
    faces = detector(img)
    face = faces[0]

    dlib_shape = predictor(img, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # compute center of face
    # 얼굴의 좌상단
    top_left = np.min(shape_2d, axis=0)
    # 얼굴의 우하단
    bottom_right = np.max(shape_2d, axis=0)
    # 얼굴의 중심점
    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

    # visualize
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(
    ), face.bottom()), color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 얼굴 특징점 표시 (marked on face)
    for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius=1, color=(
            255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # 얼굴의 좌상단 우하단에 파랑색 특징점 표시
    cv2.circle(img, center=tuple(top_left), radius=1, color=(
        255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.circle(img, center=tuple(bottom_right), radius=1,
               color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

    # 얼굴의 중심에 빨간점 표시
    cv2.circle(img, center=tuple((center_x, center_y)), radius=1,
               color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('img', img)
    cv2.waitKey(1)