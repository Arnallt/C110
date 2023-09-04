import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('keras_model.h5')

video = cv2.VideoCapture(0)

while True:

    check,frame = video.read()
    global img
    #Modeify the input data by
    #1.Resizing the image
    try:
        img = cv2.resize(frame,(224,224))
    except Exception as e:
        print(str(e))
    
    test_image = np.array(img, dtype=np.float32)
    test_image = np.expand_dims(test_image, axis = 0)

    normalised_image = test_image/255.0

    prediction = model.predict(normalised_image)

    print("Predicition : " , prediction)

    cv2.imshow("Result", frame)

    key = cv2.waitKey(1)

    if key == 32:
        print("Closing")
        break

video.release()