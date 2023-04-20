import cv2
import os

class FaceDetector:
    def __init__(self):
        # Load the classifier for face detection
        self.face_cascade = cv2.CascadeClassifier('env/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml')

        if self.face_cascade.empty():
            print("Error loading face detection classifier file!")


        # Load the known faces from the directory
        self.known_faces_dir = 'Classes'
        self.known_faces = {}
        for filename in os.listdir(self.known_faces_dir):
            self.image = cv2.imread(os.path.join(self.known_faces_dir, filename))
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.faces = self.face_cascade.detectMultiScale(self.gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(self.faces) > 0:
                x, y, w, h = self.faces[0]
                self.known_faces[filename] = self.gray[y:y+h, x:x+w]

        # Initialize the video capture device
        self.video_capture = cv2.VideoCapture(0)

    def detect(self):
        # Loop over frames from the video feed
        while True:
            # Read a frame from the video feed
            self.ret, self.frame = self.video_capture.read()

            # Convert the frame to grayscale for face detection
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the grayscale image
            self.faces = self.face_cascade.detectMultiScale(self.gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Loop over the detected faces
            for (x, y, w, h) in self.faces:
                # Extract the face region from the grayscale image
                self.face = self.gray[y:y+h, x:x+w]
                
                # Compare the face to the known faces
                for filename, known_face in self.known_faces.items():
                    # Resize the known face to match the size of the detected face
                    resized_known_face = cv2.resize(known_face, (w, h))
                    
                    # Calculate the absolute difference between the detected face and the known face
                    self.diff = cv2.absdiff(self.face, resized_known_face)
                    
                    # Calculate the sum of squared differences between the detected face and the known face
                    sse = (self.diff ** 2).sum()
                    
                    # If the sum of squared differences is below a threshold, the face is a match
                    if sse < 100000:
                        print('Match:', filename)
                
                # Draw a rectangle around the detected face
                cv2.rectangle(self.frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display the resulting image
            cv2.imshow('Video', self.frame)
            
            # Exit the program if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture device and close the window
        self.video_capture.release()
        cv2.destroyAllWindows()
