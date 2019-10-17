# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:25:04 2019

@author: Aymeric
"""

import numpy as np
from tensorflow.contrib import predictor
from azureml.core.model import Model
import json


def init():
    global loaded_model
    
    saved_model_location=Model.get_model_path('founder-classifier-test')
    loaded_model = predictor.from_saved_model(saved_model_location)

def run(raw_data):
    
    founders=['Bill Gates', 'Jeff Bezos', 'Larry Page']
    
    raw_data = np.array(json.loads(raw_data)['data'])
    data=np.reshape((raw_data), (-1,28,28,1))
    predictions = loaded_model({'x': data})['classes']
    founder=founders[predictions[0]]
    
    #return a serializable object 
    return str(founder)