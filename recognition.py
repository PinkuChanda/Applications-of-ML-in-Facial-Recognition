import face_recognition
import os
import glob
import numpy as np
import cv2


class FaceRecognition:
    def __init__(self):
        self.dataset_encodings = []
        self.dataset_classes = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def generate_face_embeddings(self, dataset_dir):
        """
        generate and load face embeddings for each class image in dataset directory
        :param dataset_dir: directory of images
        :return: None
        """
        # Load Images using glob
        images_path = glob.glob(os.path.join(dataset_dir, "*.*"))
        print("Total Classes in Dataset:{}".format(len(images_path)))

        # loop over images and generate embeddings
        for img_path in images_path:
            # read image
            img = cv2.imread(img_path)
            # convert to rgb
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            # Get encoding/embeddings
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # store image encoding/embeddings
            self.dataset_encodings.append(img_encoding)

            # store class name for generated embeddings
            self.dataset_classes.append(filename)

        print("Encodings generated for {} classes".format(len(self.dataset_classes)))

    def identify_faces(self, frame):
        # resize frame for faster speed
        resized_frame = cv2.resize(frame, None, fx=self.frame_resizing, fy=self.frame_resizing)

        # convert to rgb color
        rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Identy
        face_locations = face_recognition.face_locations(rgb_resized_frame)

        # check if any face is detected
        if len(face_locations) > 0:
            identified_classes = []

            # get face encodings for the detected faces in frame
            face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations)

            # loop over each face encoding in case of multiple faces in frame
            for face_encoding in face_encodings:
                # match face with known face encodings
                matches = face_recognition.compare_faces(self.dataset_encodings, face_encoding)
                class_name = "Unknown"
                # find known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.dataset_encodings, face_encoding)

                # get index of the smallest distance
                best_match_index = np.argmin(face_distances)

                # check if the face is a match for any known face
                if matches[best_match_index]:
                    class_name = self.dataset_classes[best_match_index]

                identified_classes.append(class_name)

            # Convert to numpy array to adjust coordinates with frame resizing quickly
            face_bounding_boxes = np.array(face_locations)
            face_bounding_boxes = face_bounding_boxes / self.frame_resizing
            # return identified classes and face bounding boxes
            return face_bounding_boxes.astype(int), identified_classes
        else:
            return None, None
