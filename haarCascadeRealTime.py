import cv2

# To get the current time
from datetime import datetime

# for efficiency of time testing
start = datetime.now()
print("Starting Time =", start)
###########################################################

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:

    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(image, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow('img', image)

    ##################################################
    end = datetime.now()
    print("End Time =", end)
    duration = end - start
    duration_in_s = duration.total_seconds()
    print(duration_in_s)
    ##################################################
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
