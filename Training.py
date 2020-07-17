import os               # This library is used to get access to your Operating System
import numpy as np      # This library is used to manipulate & claculate with Vectors & Arrays
import scipy.io as sio  # This library is used to save Dataset variables as MATLAB Matrix format (*.mat) & Read them again from *.mat file
import cv2              # This library is used in this program to Recognize faces & get access to your Webcam.
import face_recognition # This library is used to Face Image Reading & all the manipulation required to recognaize faces.
import datetime         # This library is used to get the TimeStamp in order to store Matched faces name with Matched Date-Time
from os import walk

mypath = './known_faces/'
known_faces = []
for (dirpath, dirnames, filenames) in walk(mypath):
    known_faces.extend(filenames)
    break

image_list = []    # Make an empty List

for i in range(len(known_faces)):
    im = face_recognition.load_image_file('./known_faces/' + known_faces[i])
    image_list.append(im)

for i in range(len(known_faces)):
    known_faces[i] = known_faces[i].replace('.jpg','')
    known_faces[i] = known_faces[i].replace('.JPG','')

known_face_encoding_dictionary = []    # Make an empty List

print("\nPlease wait..." + "\n\tTraining Dataset is being loaded...\n")
print("\nSL no. \t\t Image_name \t Image_pixel_shape\n")

for i in range(len(known_faces)):
    image_name, image = known_faces[i], image_list[i]  # Converting List string into variable
    print(i+1, 'of', len(known_faces), '\t ', image_name, '\t ', image.shape)
    face_encoded = face_recognition.face_encodings(image)[0]
    known_face_encoding_dictionary.append(face_encoded)

Saved_Dataset = {
                    'Known_Faces_Name': known_faces,
                    'Known_Faces_Encoding_Dictionary': known_face_encoding_dictionary
                }

sio.savemat('Trained_Dataset.mat', Saved_Dataset)
