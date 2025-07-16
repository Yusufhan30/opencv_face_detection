import cv2


cizimxml = "haarcascade_frontalface_default.xml"
yuzCascadeClassifier = cv2.CascadeClassifier(cizimxml)


video_capture = cv2.VideoCapture(0)


while True:    
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    faceDetected = yuzCascadeClassifier.detectMultiScale(
        gray,         
        minSize=(35, 35)
    )
    
    
    for (x, y, w, h) in faceDetected:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x+50 , y+120 ), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

    
    cv2.imshow('Python Yüz Tanıma - Image Processing', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()