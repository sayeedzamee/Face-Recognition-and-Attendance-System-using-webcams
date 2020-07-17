import os               # This library is used to get access to your Operating System
import numpy as np      # This library is used to manipulate & claculate with Vectors & Arrays
import scipy.io as sio  # This library is used to save Dataset variables as MATLAB Matrix format (*.mat) & Read them again from *.mat file
import cv2              # This library is used in this program to Recognize faces & get access to your Webcam.
import face_recognition # This library is used to Face Image Reading & all the manipulation required to recognaize faces.
import datetime         # This library is used to get the TimeStamp in order to store Matched faces name with Matched Date-Time

Saved_Dataset = sio.loadmat('Trained_Dataset.mat')
known_faces = Saved_Dataset['Known_Faces_Name']
known_face_encoding_dictionary = Saved_Dataset['Known_Faces_Encoding_Dictionary']

webcam = cv2.VideoCapture(0)
previous_name = {}

while True:
    ret, frame = webcam.read()  #capture frame by frame of video
    rgb_frame = frame[:, :, ::-1]  #converting the frame from OpenCV's BGR format to the RGB format

    #finds the face locations and encodings in each frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    #loops through each face in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        #checks if the face is a match for known faces
        matches = face_recognition.compare_faces(known_face_encoding_dictionary, face_encoding)
        #if not, labelled as Unknown
        name = 'Unknown'

        # Given a list of face encodings, compare them to a known face encoding and get a euclidean distance for each comparison face. The distance tells you how similar the faces are
        face_distances = face_recognition.face_distance(known_face_encoding_dictionary, face_encoding)
        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = known_faces[best_match]

            # Save Attendance Record
            if previous_name != name:
                datetime_object = str(datetime.datetime.now())
                Attendance = str([datetime_object, name])
                with open('TimeStamp_MatchedFaces.txt', 'a') as txt_File:
                    txt_File.write(Attendance)
                previous_name = name


        #draws a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        #draws a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX;
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    #displays the webcam video on screen
    cv2.imshow('Video', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

webcam.release()
cv2.destroyAllWindows()
