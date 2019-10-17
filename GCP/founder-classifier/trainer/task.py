# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:17:14 2019

@author: Aymeric
"""

import tensorflow as tf
import os
import argparse
import subprocess
from . import model
import sys

def main():
    
    ##ARGPARSE
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, dest='data_folder', help='address to the google storage bucket')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')
    parser.add_argument('--steps', type=int, dest='steps', help = 'number of steps')
    
    args = parser.parse_args()
    
    data_folder = args.data_folder
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    steps = args.steps

    print('training dataset is stored here:', data_folder)
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    data_files= data_folder + '/*.json'
    
    training_dir='data'
    
    os.makedirs(training_dir)
    
    subprocess.check_call(['gsutil', 'cp', data_files , training_dir], stderr=sys.stdout)

    output_dir='outputs/model'
    
    os.makedirs(output_dir)
    
    params={'learning_rate' : learning_rate,
    			'batch_size' : batch_size}
    
    
    estimator = estimator=tf.estimator.Estimator(model_fn = model.model_fn,
                                                 model_dir = output_dir,
                                                 params = params)
       
    tensors_to_log = {"probabilities": "softmax_tensor"}
    
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log,
            every_n_iter=50)
    	
    
    estimator.train(
            input_fn=model.train_input_fn(training_dir, params),
            steps=steps,
            hooks=[logging_hook])
    
    estimator.export_saved_model(output_dir, model.serving_input_fn)
    
    subprocess.check_call(['gsutil', '-m', 'cp', '-R' ,output_dir, data_folder], stderr=sys.stdout)

if __name__ == "__main__":
    main()