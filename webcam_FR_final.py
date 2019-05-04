import cv2
import sys
import face_recognition

# Encode known faces
vincent_image = face_recognition.load_image_file("groepsfoto/Vincent.jpeg")
vincent_face_encoding = face_recognition.face_encodings(vincent_image)[0]

kylie_image = face_recognition.load_image_file("groepsfoto/Kylie.jpeg")
kylie_face_encoding = face_recognition.face_encodings(kylie_image)[0]

known_faces = [
    vincent_face_encoding,
    kylie_face_encoding
]

# Connect to webcam
cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)

# Loop through every frame comming in, locate and recognize faces
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Draw a box around the face
    for face in face_locations:
        top = face[0]
        right = face[1]
        bottom = face[2]
        left = face[3]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Check if there is a match 
    face_names = []
    for person in face_encodings:
        match = face_recognition.compare_faces(known_faces, person, tolerance=1)

        name = None
        if match[0]:
            name = "Vincent"
        elif match[1]:
            name = "Emma"

        face_names.append(name)

    # Draw a label with a name below the face
    for name in face_names:
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()