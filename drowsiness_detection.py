import cv2
import dlib
import pyttsx3
from scipy.spatial import distance
import numpy as np
import os

# INITIALIZING THE pyttsx3 FOR AUDIO ALERTS
engine = pyttsx3.init()

# SETTING UP CAMERA (0 for default camera, 1 for external)
cap = cv2.VideoCapture(0)

# FACE DETECTION USING DLIB
face_detector = dlib.get_frontal_face_detector()

# LOADING AGE PREDICTION MODEL
prototxt_path = "C:\\Users\\Om Sai\\Desktop\\Machine Learning Internship\\Project 2\\models\\age_deploy.prototxt"
caffemodel_path = "C:\\Users\\Om Sai\\Desktop\\Machine Learning Internship\\Project 2\\models\\age_net.caffemodel"

if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
    age_model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    print("Age prediction model loaded successfully.")
else:
    print("Error: Age prediction model files not found.")
    age_model = None  # Set to None to skip age prediction if files are missing

# LOADING THE FACIAL LANDMARK DETECTOR
landmark_model_path = "C:\\Users\\Om Sai\\Desktop\\Machine Learning Internship\\Project 2\\shape_predictor_68_face_landmarks.dat"
if os.path.exists(landmark_model_path):
    dlib_facelandmark = dlib.shape_predictor(landmark_model_path)
else:
    print("Error: Facial landmark model not found.")
    exit()

# FUNCTION TO CALCULATE THE ASPECT RATIO OF THE EYE
def Detect_Eye(eye):
    poi_A = distance.euclidean(eye[1], eye[5])
    poi_B = distance.euclidean(eye[2], eye[4])
    poi_C = distance.euclidean(eye[0], eye[3])
    aspect_ratio_Eye = (poi_A + poi_B) / (2 * poi_C)
    return aspect_ratio_Eye

# Age categories and mean values for pre-processing
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

time_stamp = 7
flag = 0

# MAIN LOOP FOR DROWSINESS DETECTION AND AGE PREDICTION
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame. Exiting.")
        break

    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_scale)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray_scale, face)
        leftEye = []
        rightEye = []

        # EYE DETECTION
        for n in range(42, 48):  # Right eye
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1 if n != 47 else 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(36, 42):  # Left eye
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1 if n != 41 else 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

        # EYE ASPECT RATIO CALCULATION
        right_Eye = Detect_Eye(rightEye)
        left_Eye = Detect_Eye(leftEye)
        Eye_Rat = (left_Eye + right_Eye) / 2
        Eye_Rat = round(Eye_Rat, 2)

        # DROWSINESS DETECTION
        if Eye_Rat < 0.3:
            flag += 1
            if flag>=time_stamp:

                cv2.putText(frame, "DROWSINESS DETECTED", (50, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 210), 3)
                cv2.putText(frame, "Alert!!!! WAKE UP DUDE", (50, 450),
                        cv2.FONT_HERSHEY_PLAIN, 2, (21, 56, 212), 3)
                engine.say("Alert!!!! WAKE UP DUDE")
                engine.runAndWait()
                flag = 0

        else:
            flag = 0

        # AGE PREDICTION
        if age_model is not None:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            face_img = frame[y1:y2, x1:x2]

            if face_img.size > 0:
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), model_mean, swapRB=False)
                age_model.setInput(blob)
                age_preds = age_model.forward()
                age = ageList[age_preds[0].argmax()]

                cv2.putText(frame, f"Age: {age}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Drowsiness and Age Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CLEANUP: Release capture and close window
cap.release()
cv2.destroyAllWindows()
