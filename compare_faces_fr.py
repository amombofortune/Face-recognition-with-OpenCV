"""
This script makes use of face_recognition package to calculate the 128D descriptor to be used for face recognition
and compare the faces using some distance metrics
"""
import face_recognition

#Load known images. These images are loaded in RGB format
known_image_1 = face_recognition.load_image_file("/Users/fortuneamombo/Desktop/Projects/OpenCV/face_recognition/jared_1.jpg")
known_image_2 = face_recognition.load_image_file("/Users/fortuneamombo/Desktop/Projects/OpenCV/face_recognition/jared_2.jpg")
known_image_3 = face_recognition.load_image_file("/Users/fortuneamombo/Desktop/Projects/OpenCV/face_recognition/jared_3.jpg")
known_image_4 = face_recognition.load_image_file("/Users/fortuneamombo/Desktop/Projects/OpenCV/face_recognition/obama.jpg")

#Create names for each loaded image
names = ["Jayden_1.jpg", "Jayden_2.jpg", "Jayden_3.jpg", "Obama.jpg"]

#Load unknown images
unknown_image = face_recognition.load_image_file("/Users/fortuneamombo/Desktop/Projects/OpenCV/face_recognition/jared_4.jpg")

#Calculate the encodings for every of the image
known_image_1_encoding = face_recognition.face_encodings(known_image_1)[0]
known_image_2_encoding = face_recognition.face_encodings(known_image_2)[0]
known_image_3_encoding = face_recognition.face_encodings(known_image_3)[0]
known_image_4_encoding = face_recognition.face_encodings(known_image_4)[0]
known_encodings = [known_image_1_encoding, known_image_2_encoding, known_image_3_encoding, known_image_4_encoding]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

#Compare the faces
results = face_recognition.compare_faces(known_encodings, unknown_encoding)

#Print results
print(results)