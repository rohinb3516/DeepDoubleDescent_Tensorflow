from tensorflow.python.lib.io import tf_record
from tensorflow.core.util import event_pb2

import matplotlib.pyplot as plt
import numpy as np

def extract_model_metrics(path):
    ''' 
        Retrieves the Model Metrics written by a tf.summary.SummaryWriter located at the given path.
        
        Assumes the summary tags are "Train/Test Accuracy" and "Train/Test Loss"
        Returns a dictionary of 4 numpy arrays containing train/test loss and accuracy.
    '''
    
    summary_iter = tf_record.tf_record_iterator(path)

    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []

    for log_item in summary_iter:
        log_item = event_pb2.Event.FromString(log_item)
        if log_item.step < 1:
            continue

        val = tf.make_ndarray(log_item.summary.value[0].tensor)
        tag = log_item.summary.value[0].tag

        if tag == "Train Accuracy":
            train_acc.append(val)
        elif tag == "Train Loss":
            train_loss.append(val)
        elif tag == "Test Accuracy":
            test_acc.append(val)
        elif tag == "Test Loss":
            test_loss.append(val)
        
    train_acc = np.array(train_acc)
    train_loss = np.array(train_loss)
    test_acc = np.array(test_acc)
    test_loss = np.array(test_loss)
    
    return {
        'train_accuracy': train_acc,
        'train_loss': train_loss,
        'test_accuracy': test_acc,
        'test_loss': test_loss        
    }