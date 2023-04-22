import cv2
import os

class FaceExtractor:
    def __init__(self):
        # Load the input image
        self.input_image_path = 'ClassUpload/SwimTeam.jpeg'
        self.input_image = cv2.imread(self.input_image_path)
        if self.input_image is None:
            print('Error: could not load image')
            exit()

        # Load the face detection classifier
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def extract(self):
        # Detect faces in the image
        self.faces = self.face_cascade.detectMultiScale(self.input_image, scaleFactor=1.1, minNeighbors=5)

        # Create a directory to store the cropped face images
        self.output_dir = 'Classes'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Crop and save each face as a separate image
        for i, (x, y, w, h) in enumerate(self.faces):
            self.face_image = self.input_image[y:y+h, x:x+w]
            self.output_image_path = os.path.join(self.output_dir, f'face_{i}.jpg')
            cv2.imwrite(self.output_image_path, self.face_image)
