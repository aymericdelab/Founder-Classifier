# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 21:12:10 2019

@author: Aymeric
"""

import numpy as np
import tensorflow as tf
import json
import os
import argparse

def model_fn(features, labels, mode, params):
    
    learning_rate=params.get('learning_rate',0.001)
    
    print('learning rate is : ' + str(learning_rate))

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    
    print('reshaped input')

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    
    print('log 1 ok')
    
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.layers.flatten(inputs=pool2)
    
    print('log 2 ok')
    
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=3)
    
    print('log 3 ok')
    
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    
    print('log 4 ok')

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar('loss',loss)
    
    print('loss is fine')

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
def train_input_fn(training_dir, params):
    
    path=os.path.join(training_dir,'train.json')
    
    with open(path,'rb') as f:
        train=json.load(f)
    X_train=np.array(train['images'],dtype=np.float64)
    y_train=np.array(train['labels'],dtype=np.int64)    
    
    print('X_train has shape : ' + str(X_train.shape))
    
    batch_size=params.get('batch_size',32)
    
    return tf.estimator.inputs.numpy_input_fn(
        x={"x": X_train},
        y=y_train,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)

def eval_input_fn(training_dir):
    
    path=os.path.join(training_dir,'test.json')
                      
    with open(path, 'rb') as f:
        test=json.load(f)
    X_test=np.array(test['images'],dtype=np.float64)
    y_test=np.array(test['labels'],dtype=np.int64)
    
    return tf.estimator.inputs.numpy_input_fn(
        x={"x": X_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
 

def serving_input_fn():
    inputs = {
        'x': tf.placeholder(tf.float64, [None, 28, 28, 1])
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


#######################################################################################################################


def main():
    
    ##ARGPARSE
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=50, help='mini batch size for training')
    parser.add_argument('--learning-rate', type=float, dest='learning_rate', default=0.001, help='learning rate')
    parser.add_argument('--prefix', type=str, dest='prefix', help='target path when uploading data to Azure storage')
    parser.add_argument('--steps', type=int, dest='steps', help = 'number of steps')
    
    args = parser.parse_args()
    
    data_folder = args.data_folder
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    prefix = args.prefix
    steps = args.steps

    print('training dataset is stored here:', data_folder)
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    training_dir = os.path.join(data_folder,prefix)

    output_dir=os.path.join(training_dir,'./outputs/model')
    
    os.makedirs(output_dir, exist_ok=True)
    
    params={'learning_rate' : learning_rate,
    			'batch_size' : batch_size}
    
    
    estimator = estimator=tf.estimator.Estimator(model_fn = model_fn,
                                                 model_dir = output_dir,
                                                 params = params)
       
        
    tensors_to_log = {"probabilities": "softmax_tensor"}
    
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log,
            every_n_iter=50)
    	
    
    estimator.train(
            input_fn=train_input_fn(training_dir, params),
            steps=steps,
            hooks=[logging_hook])
    
    estimator.export_saved_model(output_dir, serving_input_fn)

if __name__ == "__main__":
    main()


'''
    callable_train_input_fn = lambda: train_input_fn(training_dir, hyperparameters)
    callable_eval_input_fn = lambda: eval_input_fn(training_dir)    
    ''' 
