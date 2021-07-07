import cv2

# To get the current time
from datetime import datetime

# for efficiency of time testing
start = datetime.now()
print("Starting Time =", start)
###########################################################

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread("images/face.jpeg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.putText(img, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('img', img)

##################################################
end = datetime.now()
print("End Time =", end)
duration = end - start
duration_in_s = duration.total_seconds()
print(duration_in_s)
##################################################

cv2.waitKey(0)
