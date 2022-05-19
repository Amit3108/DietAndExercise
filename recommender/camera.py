
from imutils.video import VideoStream
import imutils
import cv2
import os
import urllib.request
import numpy as np
from django.conf import settings
#####
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # def get_frame(self):
    # 	success, image = self.video.read()
    # 	# We are using Motion JPEG, but OpenCV defaults to capture raw images,
    # 	# so we must encode it into JPEG in order to correctly display the
    # 	# video stream.

    # 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 	faces_detected = face_detection_videocam.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    # 	for (x, y, w, h) in faces_detected:
    # 		cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
    # 	frame_flip = cv2.flip(image,1)
    # 	ret, jpeg = cv2.imencode('.jpg', frame_flip)
    # 	return jpeg.tobytes()
############################################

    def calculate_angle(a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
            np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180.0:
            angle = 360-angle

        return angle


###############################################


    def get_frame(self):
        def calculate_angle(a, b, c):
            a = np.array(a)  # First
            b = np.array(b)  # Mid
            c = np.array(c)  # End

            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
                np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)

            if angle > 180.0:
                angle = 360-angle

            return angle

        counter = 0
        stage = None
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while self.video.isOpened():
                ret, frame = self.video.read()

                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detection
                results = pose.process(image)

                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Render detections
                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get coordinates
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle

                    angle = calculate_angle(shoulder, elbow, wrist)

                    # Visualize angle
                    # cv2.putText(image, str(angle),
                    #             tuple(np.multiply(
                    #                 elbow, [640, 480]).astype(int)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                    #                 255, 255, 255), 2, cv2.LINE_AA
                    #             )

                    # Curl counter logic

                    if angle > 160:
                        stage = "down"
                    if angle < 30 and stage == 'down':
                        stage = "up"
                        counter += 1
                        print(counter)

                    # if counter == 15:
                    #     counter = ""
                    #     stage = "Done"
                    #     print("Task Completed")
                        # cap.release()

                except:
                    pass

                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(
                                              color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(
                                              color=(245, 66, 230), thickness=2, circle_radius=2)
                                          )
##########################################################
                frame_flip = cv2.flip(image, 1)
                cv2.putText(frame_flip, str(angle),
                            tuple(np.multiply(elbow, [200, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (
                                255, 255, 255), 2, cv2.LINE_AA
                            )
                cv2.rectangle(frame_flip, (0, 0),
                              (225, 73), (245, 117, 16), -1)
                cv2.putText(frame_flip, 'REPS', (15, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame_flip, str(counter),
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)


                # Stage data
                cv2.putText(frame_flip, 'STAGE', (65, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(frame_flip, stage,
                            (60, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
                ret, jpeg = cv2.imencode('.jpg', frame_flip)
                return jpeg.tobytes()
