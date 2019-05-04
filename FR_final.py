# Import required packages
import face_recognition
import cv2
import numpy as np
import subprocess
import os

# Import video
input_video = cv2.VideoCapture("Video/test_movie.mov")

# Get number of frames from the video
length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

# Set format for output
fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
output_video = cv2.VideoWriter('output_NF.avi', # Output name
                               fourcc, # data format
                               30, # frames per second
                               (480, 360)) # size of the frames

# Create name and photo dictionary
# The name becomes the name showed in the video
# The photo must be the name of how the picture is saved
image_file_names = os.listdir('./Fotos/')

pic_name_dict = {}
for img in image_file_names:
    pic_name_dict[img[:-4]] = img

# Initiate empty dictionary for face encodings
face_encodings_dict = {}

# Encode known faces
for name in pic_name_dict:
    image = face_recognition.load_image_file("Fotos/" + pic_name_dict[name])
    face_location = face_recognition.face_locations(image)
    face_encodings_dict[name] = face_recognition.face_encodings(image, face_location, num_jitters=5)[0]

# Save face encodings as a list
known_faces = [face_encodings_dict[name] for name in face_encodings_dict]

# Save names as list
name_list = list(pic_name_dict.keys())

# Initialize frame counter
frame_number = 0

# Loop through each frame
while True:  
    ret, frame = input_video.read() # read in the current frame
    frame_number += 1 # add 1 frame to the frame count

    # Quit when the input video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=2)

    # See if the face is a match for the known face(s)
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

        print_name = name[:-3]
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, print_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_video.write(frame)

# Add sound to clip (make sure to remove old file or rename before running again)
cmd = 'ffmpeg -i "output_NF.avi" -i "Video/test_movie_audio.mp3" -qscale 0 output_complete.avi'
subprocess.call(cmd, shell=True)