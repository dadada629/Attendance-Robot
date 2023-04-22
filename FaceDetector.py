import cv2
from ImageDetector import ImageDetector

# Create a video capture object
cap = cv2.VideoCapture(0)
detector = ImageDetector("Classes")

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Start an infinite loop
while True:
    # Capture the video frame
    ret, frame = cap.read()

    detector.detect(frame)

    # Display the resulting frame
    #cv2.imshow("Webcam", frame)

    # The 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()
