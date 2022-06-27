import numpy as np
import cv2
import tensorflow as tf
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

# mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color, (96, 96))
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            img_array = tf.keras.utils.img_to_array(roi_color)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            text = "{} : {:.2f}".format(classes[np.argmax(score)], 100 * np.max(score))
            img = cv2.putText(img, text, (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


    cv2.imshow('img',img)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break


cv2.destroyAllWindows()
cv2.waitKey(1)
cap.release()