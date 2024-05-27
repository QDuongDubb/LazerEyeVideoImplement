#!/usr/bin/python3
# -*- coding:utf-8 -*-

from service.head_pose import HeadPoseEstimator
from service.face_alignment import CoordinateAlignmentModel
from service.face_detector import MxnetDetectionModel
from service.iris_localization import IrisLocalizationModel
import cv2
import numpy as np
from numpy import sin, cos, pi, arctan
from numpy.linalg import norm
import time
from queue import Queue
from threading import Thread
import sys
import math

SIN_LEFT_THETA = 2 * sin(pi / 4)
SIN_UP_THETA = sin(pi / 6)


def calculate_3d_gaze(frame, poi, scale=256):
    starts, ends, pupils, centers = poi

    eye_length = norm(starts - ends, axis=1)
    ic_distance = norm(pupils - centers, axis=1)
    zc_distance = norm(pupils - starts, axis=1)

    s0 = (starts[:, 1] - ends[:, 1]) * pupils[:, 0]
    s1 = (starts[:, 0] - ends[:, 0]) * pupils[:, 1]
    s2 = starts[:, 0] * ends[:, 1]
    s3 = starts[:, 1] * ends[:, 0]

    delta_y = (s0 - s1 + s2 - s3) / eye_length / 2
    delta_x = np.sqrt(abs(ic_distance**2 - delta_y**2))

    delta = np.array((delta_x * SIN_LEFT_THETA,
                      delta_y * SIN_UP_THETA))
    delta /= eye_length
    theta, pha = np.arcsin(delta)

    # print(f"THETA:{180 * theta / pi}, PHA:{180 * pha / pi}")
    # delta[0, abs(theta) < 0.1] = 0
    # delta[1, abs(pha) < 0.03] = 0

    inv_judge = zc_distance**2 - delta_y**2 < eye_length**2 / 4

    delta[0, inv_judge] *= -1
    theta[inv_judge] *= -1
    delta *= scale

    # cv2.circle(frame, tuple(pupil.astype(int)), 2, (0, 255, 255), -1)
    # cv2.circle(frame, tuple(center.astype(int)), 1, (0, 0, 255), -1)

    return theta, pha, delta.T


def draw_sticker(src, offset, pupils, landmarks,
                 blink_thd=0.22,
                 arrow_color=(0, 125, 255), copy=False):
    if copy:
        src = src.copy()

    left_eye_hight = landmarks[33, 1] - landmarks[40, 1]
    left_eye_width = landmarks[39, 0] - landmarks[35, 0]
    left_iris_center = (landmarks[42, 0] + landmarks[45, 0]) / 2, (landmarks[42, 1] + landmarks[45, 1]) / 2
    left_iris_radius = (landmarks[42, 0] - landmarks[45, 0]) / 2  # Assuming circular iris

    right_eye_hight = landmarks[87, 1] - landmarks[94, 1]
    right_eye_width = landmarks[93, 0] - landmarks[89, 0]
    right_iris_center = (landmarks[90, 0] + landmarks[93, 0]) / 2, (landmarks[90, 1] + landmarks[93, 1]) / 2
    right_iris_radius = (landmarks[90, 0] - landmarks[93, 0]) / 2

    for mark in landmarks.reshape(-1, 2).astype(int):
        circle = cv2.circle(src, tuple(mark), radius=1,
                   color=(0, 0, 255), thickness=-1)

    if left_eye_hight / left_eye_width > blink_thd:
            cv2.arrowedLine(src, tuple(pupils[0].astype(int)),      
                        tuple((offset+pupils[0]).astype(int)), arrow_color, 2)

    if right_eye_hight / right_eye_width > blink_thd:
            cv2.arrowedLine(src, tuple(pupils[1].astype(int)),
                            tuple((offset+pupils[1]).astype(int)), arrow_color, 2)

    return src


def main(video, gpu_ctx=-1):
    cap = cv2.VideoCapture(0)

    fd = MxnetDetectionModel('weights/16and32', 0, .6, gpu=gpu_ctx)
    fa = CoordinateAlignmentModel('weights/2d106det', 0, gpu=gpu_ctx)
    gs = IrisLocalizationModel("weights/iris_landmark.tflite")
    hp = HeadPoseEstimator("weights/object_points.npy", cap.get(3), cap.get(4))

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        bboxes = fd.detect(frame)

        for landmarks in fa.get_landmarks(frame, bboxes, calibrate=True):
            # calculate head pose
            _, euler_angle = hp.get_head_pose(landmarks)
            pitch, yaw, roll = euler_angle[:, 0]

            eye_markers = np.take(landmarks, fa.eye_bound, axis=0)
            
            eye_centers = np.average(eye_markers, axis=1)
            # eye_centers = landmarks[[34, 88]]
            
            # eye_lengths = np.linalg.norm(landmarks[[39, 93]] - landmarks[[35, 89]], axis=1)
            eye_lengths = (landmarks[[39, 93]] - landmarks[[35, 89]])[:, 0]
            left_eye_hight = landmarks[33, 1] - landmarks[40, 1]
            left_eye_width = landmarks[39, 0] - landmarks[35, 0]

            right_eye_hight = landmarks[87, 1] - landmarks[94, 1]
            right_eye_width = landmarks[93, 0] - landmarks[89, 0]

            S_left_eye = math.pi * left_eye_hight * left_eye_width
            s_right_eye = math.pi * right_eye_hight * right_eye_width

            eye_centers_left = left_eye_width / 2
            eye_center_right = right_eye_width / 2


            iris_left = gs.get_mesh(frame, eye_lengths[0], eye_centers[0])
            pupil_left, _ = gs.draw_pupil(iris_left, frame, thickness=1)

            iris_right = gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
            pupil_right, _ = gs.draw_pupil(iris_right, frame, thickness=1)

            pupils = np.array([pupil_left, pupil_right])

            poi = landmarks[[35, 89]], landmarks[[39, 93]], pupils, eye_centers
            theta, pha, delta = calculate_3d_gaze(frame, poi)

            if yaw > 30:
                end_mean = delta[0]
            elif yaw < -30:
                end_mean = delta[1]
            else:
                end_mean = np.average(delta, axis=0)

            if end_mean[0] < 0:
                zeta = arctan(end_mean[1] / end_mean[0]) + pi
            else:
                zeta = arctan(end_mean[1] / (end_mean[0] + 1e-7))

            # print(zeta * 180 / pi)
            # print(zeta)
            if roll < 0:
                roll += 180
            else:
                roll -= 180

            real_angle = zeta + roll * pi / 180
            # real_angle = zeta

            # print("end mean:", end_mean)
            # print(roll, real_angle * 180 / pi)

            R = norm(end_mean)
            offset = R * cos(real_angle), R * sin(real_angle)

            landmarks[[38, 92]] = landmarks[[34, 88]] = eye_centers

            # gs.draw_eye_markers(eye_markers, frame, thickness=1)

            draw_sticker(frame, offset, pupils, landmarks)

        cv2.imshow('res', cv2.resize(frame, (960, 540)))
        #Show landmark numbers
#   for i, landmark in enumerate(landmarks):
    #      cv2.putText(frame, str(i), tuple(landmark.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Show landmark numbers for iris left, iris right, pupil left, pupil right, eye length, and gaze

        cv2.putText(frame, "Left Eye S: " + str(S_left_eye), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "Right Eye S: " + str(s_right_eye), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "Eye Length: " + str(eye_lengths), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, "Gaze: " + str(delta.T), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('res', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main(sys.argv[0])
