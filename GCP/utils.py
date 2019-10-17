import cv2
import json
import numpy as np

def from_image_to_json(image_directory):

    image=cv2.imread(image_directory)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(100, 100)
    ) 
    
    try:
        if faces.shape[0]>1:
            return print('too many faces on the picture')
        
    except:
        return print('no face detected in the picture')
        
    for (x, y, w, h) in faces:
        crop_img = gray[y:y+h, x:x+w]
        resized_img=cv2.resize(crop_img,(28,28))
        data=np.reshape((resized_img), (-1,28,28,1))
        data_serializable=data.tolist()
    with open('image.json', 'w') as f:
        json.dump({'x' : data_serializable}, f)