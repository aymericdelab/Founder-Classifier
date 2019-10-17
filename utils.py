from sklearn.model_selection import train_test_split
import cv2
import json
import numpy as np


def images_to_json(prefix):
    
    founders=[
        {'name': 'Bill Gates',
        'label': 0},
        {'name': 'Jeff Bezos', #order encoding
        'label': 1},
        {'name': 'Larry Page',
        'label': 2}
       ]
    
    X=[]
    y=[]
    
    for founder in founders:

        path=os.path.join(prefix,founder['name'] +'/faces_28x28')
        for image in os.listdir(path):
            img=cv2.imread(os.path.join(path,image),cv2.IMREAD_GRAYSCALE)
            X.append(img)
            y.append(founder['label'])

    X_array=np.array(X)
    y_array=np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, test_size=0.2, random_state=42)

    train={'images':X_train.tolist(),
          'labels': y_train.tolist(),
          'decode': founders}

    test={'images':X_test.tolist(),
          'labels':y_test.tolist(),
          'decode': founders}

    with open('./data/train.json', 'w',encoding='utf-8') as f:
        json.dump(train,f)

    with open('./data/test.json', 'w', encoding='utf-8') as f:
        json.dump(test, f)