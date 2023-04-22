import cv2
import os

class ImageDetector:
    def __init__(self, knownDir):
        # Load the known faces and their names from the folder
        self.known_faces_dir = knownDir
        self.known_faces = []
        self.known_names = []

        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        for filename in os.listdir(self.known_faces_dir):
            image = cv2.imread(os.path.join(self.known_faces_dir, filename))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (128, 128))
                self.known_faces.append(face_roi)
                self.known_names.append(os.path.splitext(filename)[0])

    def detect(self, img):
        # Load the image to search for faces
        image = img
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find all the faces in the image
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Loop through each face found in the image
        for (x, y, w, h) in faces:
            # Resize and preprocess the face
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (128, 128))

            # Compare the face to the known faces
            match = None
            for i, known_face in enumerate(self.known_faces):
                similarity = cv2.compareHist(cv2.calcHist([known_face], [0], None, [256], [0, 256]),
                                            cv2.calcHist([face_roi], [0], None, [256], [0, 256]),
                                            cv2.HISTCMP_CORREL)
                if similarity > 0.8:
                    match = i
                    break

            # Draw a box around the face
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Write the name of the face found
            font = cv2.FONT_HERSHEY_SIMPLEX
            if match is None:
                cv2.putText(image, "Unknown", (x, y-10), font, 1, (0, 0, 255), 2)
            else:
                name = self.known_names[match]
                cv2.putText(image, name, (x, y-10), font, 1, (0, 255, 0), 2)
                match_path = os.path.join(self.known_faces_dir, name + '.jpg')
                print(f"Match found with {match_path}")

        # Display the final image with names and matches
        cv2.imshow('Image', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
