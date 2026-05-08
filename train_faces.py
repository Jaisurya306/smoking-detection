import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_label = 0

dataset_path = "faces"

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        detected_faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            faces.append(img[y:y+h, x:x+w])
            labels.append(current_label)

    current_label += 1

recognizer.train(faces, np.array(labels))
recognizer.save("face_model.yml")

np.save("labels.npy", label_map)

print("✅ Face model trained successfully")
