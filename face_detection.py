import cv2

faceCascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("haarcascades/haarcascade_smile.xml")

video = cv2.VideoCapture(0)

# img = cv2.imread("images/lena.png")
check, frame = video.read()

img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(img_gray, 1.1, 4)
eyes = eyesCascade.detectMultiScale(img_gray, 1.1, 4)
smile = smileCascade.detectMultiScale(img_gray, 1.7, 10)

for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

for (x, y, w, h) in eyes:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
for (x, y, w, h) in smile:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow("Face Detection", frame)


a = 1

while 1:
    a = a + 1
    check, frame = video.read()
    print(frame)
    
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Capturing", gray)
    check, frame = video.read()

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(img_gray, 1.1, 4)
    eyes = eyesCascade.detectMultiScale(img_gray, 1.1, 4)
    smile = smileCascade.detectMultiScale(img_gray, 1.7, 10)

    for (x, y, w, h) in faces:
        cv2.putText(frame, "Person", (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    for (x, y, w, h) in smile:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Face Detection", frame)
        
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

print(a)
video.release()
cv2.destroyAllWindows