import cv2;

camera = cv2.VideoCapture(0);

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml");
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml");

while True:
    ret, frame = camera.read();

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);

    face_coordinates = face_cascade.detectMultiScale(gray_frame,
                                                     1.2,
                                                     5,
                                                     minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE);

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (69, 69, 69));

    cv2.imshow("Video Frame", frame);

    if cv2.waitKey(1) == ord('q'):
        break;

camera.realease();
cv2.destroyAllWindows();