import cv2
import face_recognition


def facial_authentication():
    # Load the user's reference image
    user_image_path = 'fahim_front.jpg'
    user_image = face_recognition.load_image_file(user_image_path)
    user_face_encoding = face_recognition.face_encodings(user_image)[0]

    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the webcam
        ret, frame = video_capture.read()

        # Find face locations and encodings in the captured frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare the face encoding in the captured frame with the user's face encoding
            match = face_recognition.compare_faces(
                [user_face_encoding], face_encoding)[0]

            # Draw a box around the face
            color = (0, 255, 0) if match else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Display the result
            label = "User Authenticated" if match else "Unauthorized"
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return 1 if match else 0

        # Display the captured frame
        cv2.imshow('Video', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    video_capture.release()
    cv2.destroyAllWindows()
