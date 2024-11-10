import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import dlib
from scipy.spatial import distance
import pyttsx3
import os

# Paths for age model and landmark predictor
prototxt_path = "C:\\Users\\Om Sai\\Desktop\\Machine Learning Internship\\Project 2\\models\\age_deploy.prototxt"
caffemodel_path = "C:\\Users\\Om Sai\\Desktop\\Machine Learning Internship\\Project 2\\models\\age_net.caffemodel"
landmark_model_path = "C:\\Users\\Om Sai\\Desktop\\Machine Learning Internship\\Project 2\\shape_predictor_68_face_landmarks.dat"

# Initialize models
age_model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path) if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path) else None
face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor(landmark_model_path) if os.path.exists(landmark_model_path) else None
engine = pyttsx3.init()

# GUI Setup
root = tk.Tk()
root.title("Drowsiness Detection GUI")
root.geometry("600x500")
image_path = None

# Function to calculate EAR
def calculate_eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Define age categories
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)
time_stamp = 5
flag = 0

# Load image and detect drowsiness function
def load_image_and_detect_drowsiness():
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if image_path:
        img = Image.open(image_path).resize((200, 200), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        detect_drowsiness_in_image(cv2.imread(image_path))

# Detect drowsiness in an image
def detect_drowsiness_in_image(img):
    global flag
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        left_eye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(36, 42)]
        right_eye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(42, 48)]

        # Calculate EAR for both eyes
        left_EAR = calculate_eye_aspect_ratio(left_eye)
        right_EAR = calculate_eye_aspect_ratio(right_eye)
        ear = (left_EAR + right_EAR) / 2.0

        if ear < 0.2:
            result_label.config(text="Drowsiness detected in the image!")
            engine.say("Drowsiness detected in the image!")
            engine.runAndWait()
        else:
            result_label.config(text="No drowsiness detected in the image.")

        # Perform age prediction
        if age_model is not None:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            face_img = img[y1:y2, x1:x2]
            
            if face_img.size > 0:
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), model_mean, swapRB=False)
                age_model.setInput(blob)
                age_preds = age_model.forward()
                age = ageList[age_preds[0].argmax()]
                result_label.config(text=f"Age: {age}, Drowsiness: {'Detected' if ear < 0.2 else 'Not Detected'}")

# Start video detection
def start_video_detection():
    global flag
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        
        for face in faces:
            face_landmarks = dlib_facelandmark(gray, face)
            left_eye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(36, 42)]
            right_eye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(42, 48)]

            # Calculate EAR for both eyes
            left_EAR = calculate_eye_aspect_ratio(left_eye)
            right_EAR = calculate_eye_aspect_ratio(right_eye)
            ear = (left_EAR + right_EAR) / 2.0

            if ear < 0.2:
                flag += 1
                if flag >= time_stamp:
                    engine.say("Alert! Wake up, please!")
                    engine.runAndWait()
                    flag = 0  # Reset after alert
            else:
                flag = 0

            if age_model is not None:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0:
                    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), model_mean, swapRB=False)
                    age_model.setInput(blob)
                    age_preds = age_model.forward()
                    age = ageList[age_preds[0].argmax()]
                    cv2.putText(frame, f"Age: {age}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
        cv2.imshow("Drowsiness Detection - Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# GUI components
image_label = tk.Label(root)
image_label.pack()

result_label = tk.Label(root, text="Result: ", font=('Helvetica', 16))
result_label.pack()

load_button = tk.Button(root, text="Load Image and Detect Drowsiness", command=load_image_and_detect_drowsiness)
load_button.pack()

start_video_button = tk.Button(root, text="Start Video Drowsiness Detection", command=start_video_detection)
start_video_button.pack()

root.mainloop()
