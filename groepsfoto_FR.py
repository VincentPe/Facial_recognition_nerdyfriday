# Import required packages
import face_recognition
import cv2
import numpy as np
import scipy.misc
import os

# Get filenames from all pictures in the groepsfoto directory
image_file_names = os.listdir('./groepsfoto/')

# Make a dictionay with the names of the people and their respective picture
pic_name_dict = {}
for img in image_file_names:
    pic_name_dict[img[:-5]] = img

# Initiate empty dictionary for face encodings
face_encodings_dict = {}

# Encode known faces
for name in pic_name_dict:
    image = face_recognition.load_image_file("groepsfoto/" + pic_name_dict[name])
    face_encodings_dict[name] = face_recognition.face_encodings(image)[0]

# Save face encodings as a list
known_faces = [face_encodings_dict[name] for name in face_encodings_dict]

# Save names as list
name_list = list(pic_name_dict.keys())

# Import group picture
frame = face_recognition.load_image_file("groepsfoto.jpg")

# Find all the faces and face encodings in the current frame of video
face_locations = face_recognition.face_locations(frame)
face_encodings = face_recognition.face_encodings(frame, face_locations, num_jitters=5)

# Add face names to the right locations based on matching with our pictures
face_names = []
for face_encoding in face_encodings:
    match_scores = face_recognition.face_distance(known_faces, face_encoding)

    name = None
    if min(match_scores) <= 0.5: # Threshold for match
        name = name_list[np.argmin(match_scores)]

    face_names.append(name)

# Label the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    if not name:
        continue

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

# Display the resulting frame
scipy.misc.toimage(frame).show()

# Save resulting frame when happy
scipy.misc.imsave('groepsfoto_FRed.png', frame)








