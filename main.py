import cv2
from recognition import FaceRecognition

if __name__ == "__main__":
    CAMERA_DEVICE_INDEX = 0
    DATASET = "dataset"

    # create FaceRecognition object
    fr = FaceRecognition()

    # generate face embeddings for dataset directory
    fr.generate_face_embeddings(DATASET)

    # Initialize the video stream and allow the camera sensor to warm up
    cap = cv2.VideoCapture(CAMERA_DEVICE_INDEX)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # identify faces in frame
        face_bb, identified_classes = fr.identify_faces(frame)

        if face_bb is not None and identified_classes is not None:
            # loop over each face bounding box and identified class
            for face_loc, name in zip(face_bb, identified_classes):
                # get (x, y) coordinates of the face bounding box
                y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                # write name of identified class on the face of identified person
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 4)

                # draw a rectangle around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)


        # Display the resulting frame
        cv2.imshow("Frame", frame)

        # Press ESC on keyboard to  exit
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Exiting...")
