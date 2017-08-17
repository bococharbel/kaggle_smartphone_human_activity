from __future__ import division, print_function, absolute_import

import time
import os
import sys
import re
import csv
import codecs
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv
import tensorflow as tf
import glob
from datetime import datetime
from filelogger import FileLogger

#import tflearn
#from tflearn.data_utils import to_categorical, pad_sequences
#from tflearn.datasets import imdb
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.embedding_ops import embedding
#from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
#from tflearn.layers.estimator import regression
#from tflearn.layers.normalization import local_response_normalization
#from tflearn.layers.merge_ops import merge

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder


sys.path.append(os.path.abspath('..'))
#reload(sys)
#sys.setdefaultencoding('utf-8')

BASE_DIR = './'
TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
TEST_DATA_FILE = BASE_DIR + 'test.csv'
num_features = 561
mlearning_rate = 0.0025
lambda_loss_amount = 0.0015
num_epochs = 1502
batch_size = 10
num_hidden = 561
num_classes = 6
max_input_length=1
num_layers= 2
num_train_examples=7300
num_test_examples=2947
num_batches_per_epoch=100
CLASSES=["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","SITTING","STANDING","LAYING"]
save_step=num_epochs/3

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

def onehot2(yvalues, nbclasses):
    yvalues = yvalues.reshape(len(yvalues))
    return np.eye(nbclasses)[np.array(yvalues, dtype=np.float32)]  

def onehot3(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

def read_data_by_subject(train=True):
    global num_train_examples
    global num_test_examples
    if train:
        df = read_csv(os.path.expanduser(TRAIN_DATA_FILE))  # load pandas dataframe
        labelsstr  = df["Activity"].values #np.vstack
        num_train_examples=len(labelsstr)
    else:
        df = read_csv(os.path.expanduser(TEST_DATA_FILE))  # load pandas dataframe
        labelsstr  = df["Activity"].values #np.vstack
        num_test_examples=len(labelsstr)
    subject = df["subject"].values
    ffeatures = df[df.columns[:-2]].values
    num_batches_per_epoch=len(ffeatures)
    ffeatures= ffeatures.astype(np.float32)
    ffeatures= np.asarray(ffeatures[np.newaxis, :])
    ffeatures = tf.transpose(np.array(ffeatures), [1, 0, 2]).eval()
    ffeatures = np.array(ffeatures)
    classes= CLASSES 
    subjectindex=[]
    mindex="-1"
     
    for i,k in enumerate(subject):
        if i==len(subject) -1:
            subjectindex.append(i+1)
        elif mindex!=k :
            mindex=k
            if i>0:
                subjectindex.append(i)

    labels=[CLASSES.index(key) for key in [k for i,k in enumerate(labelsstr)]]
    labels= np.array(labels)
    labels=tf.one_hot(labels, len(CLASSES)).eval()
    labels=np.array(labels)
    return ffeatures, subject, subjectindex, labels, classes
    
def read_data(train=True):
    global num_train_examples
    global num_test_examples
    if train:
        df = read_csv(os.path.expanduser(TRAIN_DATA_FILE))  # load pandas dataframe
        labelsstr  = df["Activity"].values #np.vstack
        num_train_examples=len(labelsstr)
    else:
        df = read_csv(os.path.expanduser(TEST_DATA_FILE))  # load pandas dataframe
        labelsstr  = df["Activity"].values #np.vstack
        num_test_examples=len(labelsstr)
    subject = df["subject"].values
    ffeatures = df[df.columns[:-2]].values 
    num_batches_per_epoch=len(ffeatures)    
    ffeatures= ffeatures.astype(np.float32)
    ffeatures= np.asarray(ffeatures[np.newaxis, :])
    ffeatures = tf.transpose(np.array(ffeatures), [1, 0, 2]).eval()
    ffeatures = np.array(ffeatures)
  	    
    if True:#train:
        #le = LabelEncoder().fit(labelsstr) 
        #labels = le.transform(labelsstr)           # encode species strings
        #classes = list(le.classes_)
        #labels= np.array(labels)
        classes= CLASSES 
        labels=[CLASSES.index(key) for key in [k for i,k in enumerate(labelsstr)]]
        labels=tf.one_hot(labels, len(CLASSES)).eval()
        labels=np.array(labels)
        print("read_data features shapes {} ".format(ffeatures.shape)+"labelsss shapes {}".format(labels.shape))
        #print(labels[0:100])
        return ffeatures, subject, labels, classes
    else: 
        return ffeatures, subject

def next_training_batch_subject(ffeatures, subject, subjectindex, labels, batch):
    import random
    global num_train_examples
    global num_test_examples
    num_examples=num_train_examples
    #print("next_training_batch_subject sfeaturess shape {} ".format(ffeatures.shape)+" features type {}".format(ffeatures.dtype))
    debut=batch * batch_size % num_examples
    fin=(batch + 1) * batch_size % num_examples
    if fin < debut:
        fin=num_examples
    for i,k in subjectindex:
        if debut<k:
            if fin>k:
                fin=k
            break
    features_batch = ffeatures[debut:fin]
    subject_batch = subject[debut:fin]
    labels_batch = labels[debut:fin]
    return features_batch, subject_batch, labels_batch

def next_training_batch(ffeatures, subject, labels, batch):
    import random

    #print("next_training_batch features shape {} ".format(ffeatures.shape)+" features type {}".format(ffeatures.dtype))    
    global num_train_examples
    global num_test_examples
    num_examples=num_train_examples
    debut= (batch * batch_size) % num_examples
    fin =  ((batch + 1) * batch_size) % num_examples
    if fin< debut:
        fin=num_examples
    #num_examples=  len(labels)
    #random_index = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
    features_batch=np.zeros((batch_size, max_input_length ,num_hidden))
    features_batch[0:fin-debut] = ffeatures[debut:fin]#ffeatures[batch % num_examples]
    subject_batch=np.zeros((batch_size))
    subject_batch[0:fin-debut] = subject[debut:fin]#subject[batch % num_examples]
    labels_batch=np.zeros((batch_size, num_classes))
    labels_batch[0:fin-debut] = labels[debut:fin]#labels[batch % num_examples]
    add=0
    while (add+fin-debut)<batch_size:
        #faddcol= ffeatures[fin-1:fin]
        #features_batch=np.vstack((features_batch, np.asarray( faddcol[np.newaxis, :])))#ffeatures[batch % num_examples]
        #saddcol=subject[fin-1:fin]
        #subject_batch=np.vstack((subject_batch, np.asarray( saddcol[np.newaxis, :])))#subject[batch % num_examples]
        #laddcol=labels[fin-1:fin]
        #labels_batch=np.vstack((labels_batch, np.asarray( laddcol[np.newaxis, :])))#labels[batch % num_examples]
        features_batch[add+fin-debut] = ffeatures[fin-1]#ffeatures[batch % num_examples]
        subject_batch[add+fin-debut] = subject[fin-1]#subject[batch % num_examples]
        labels_batch[add+fin-debut] = labels[fin-1]#labels[batch % num_examples]
        add=add+1
    return features_batch, subject_batch, labels_batch

def next_test_batch(ffeatures, subject, labels, batch):
    import random
    global num_train_examples
    global num_test_examples
    num_examples=num_test_examples
    debut= (batch * batch_size) % num_examples
    fin =  ((batch + 1) * batch_size) % num_examples
    if fin< debut:
        fin=num_examples    #num_examples=  len(labels)
    #random_index = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
    features_batch=np.zeros((batch_size, max_input_length ,num_hidden))
    features_batch[0:fin-debut] = ffeatures[debut:fin]##ffeatures[batch % num_examples]
    #features_batch=  np.asarray(features_batch[np.newaxis, :])
    subject_batch=np.zeros((batch_size))
    subject_batch[0:fin-debut] = subject[debut:fin]#subject[batch % num_examples]
    #subject_batch=  np.asarray( subject_batch[np.newaxis, :])
    labels_batch=np.zeros((batch_size, num_classes))
    labels_batch[0:fin-debut] = labels[debut:fin]#labels[batch % num_examples]
    #labels_batch=  np.asarray(labels_batch[np.newaxis, :])
    #print(labels_batch)
    add=0
    while (add+fin-debut)<batch_size:
        #faddcol=ffeatures[fin-1:fin]
        #print(" features_batch{}  faddcol{}".format(features_batch.shape,faddcol.shape))
        #features_batch=np.vstack((features_batch,  np.asarray( faddcol[np.newaxis, :])))#ffeatures[batch % num_examples]
        #saddcol=  subject[fin-1:fin]
        #print(" subject_batch{}  saddcol{}".format(subject_batch.shape,saddcol.shape))
        #subject_batch=np.vstack(( subject_batch, np.asarray( saddcol[np.newaxis, :])))#subject[batch % num_examples]
        #laddcol= labels[fin-1:fin]
        #print(" labels_batch{}  laddcol{}".format(labels_batch.shape,laddcol.shape))
        #labels_batch=np.vstack((labels_batch, np.asarray( laddcol[np.newaxis, :])))#labels[batch % num_examples]

        features_batch[add+fin-debut] = ffeatures[fin-1]#ffeatures[batch % num_examples]
        subject_batch[add+fin-debut] = subject[fin-1]#subject[batch % num_examples]
        labels_batch[add+fin-debut] = labels[fin-1]#labels[batch % num_examples]
        add=add+1
    return features_batch, subject_batch, labels_batch

def next_continuous_test_batch(ffeatures, subject, labels, batch):
    import random
    global num_train_examples
    global num_test_examples
    num_examples=num_test_examples
    debut= (batch) % num_examples
    fin =  ((debut ) + batch_size) #% num_examples
    #debut= (batch * batch_size) % num_examples
    #fin =  ((batch + 1) * batch_size) % num_examples
    if fin< debut or fin>=num_examples:
        fin=num_examples    #num_examples=  len(labels)
    #if num_test_examples==len(ffeatures):
    #    for i in range(batch_size):
    #        ffeatures[num_examples+i]=ffeatures[num_examples]
    #        subject[num_examples+i]=subject[num_examples]
    #        labels[num_examples+i]=labels[num_examples]

    #if fin< debut:
    #    fin=num_examples    #num_examples=  len(labels)
    #random_index = [i % num_examples for i in range(batch * batch_size, (batch + 1) * batch_size)]
    features_batch=np.zeros((batch_size, max_input_length ,num_hidden))
    features_batch[0:fin-debut] = ffeatures[debut:fin]##ffeatures[batch % num_examples]
    #features_batch=  np.asarray(features_batch[np.newaxis, :])
    subject_batch=np.zeros((batch_size))
    subject_batch[0:fin-debut] = subject[debut:fin]#subject[batch % num_examples]
    #subject_batch=  np.asarray( subject_batch[np.newaxis, :])
    labels_batch=np.zeros((batch_size, num_classes))
    labels_batch[0:fin-debut] = labels[debut:fin]#labels[batch % num_examples]
    #labels_batch=  np.asarray(labels_batch[np.newaxis, :])
    #print(labels_batch)
    add=0
    while (add+fin-debut)<batch_size:
        #faddcol=ffeatures[fin-1:fin]
        #print(" features_batch{}  faddcol{}".format(features_batch.shape,faddcol.shape))
        #features_batch=np.vstack((features_batch,  np.asarray( faddcol[np.newaxis, :])))#ffeatures[batch % num_examples]
        #saddcol=  subject[fin-1:fin]
        #print(" subject_batch{}  saddcol{}".format(subject_batch.shape,saddcol.shape))
        #subject_batch=np.vstack(( subject_batch, np.asarray( saddcol[np.newaxis, :])))#subject[batch % num_examples]
        #laddcol= labels[fin-1:fin]
        #print(" labels_batch{}  laddcol{}".format(labels_batch.shape,laddcol.shape))
        #labels_batch=np.vstack((labels_batch, np.asarray( laddcol[np.newaxis, :])))#labels[batch % num_examples]

        features_batch[add+fin-debut] = ffeatures[fin-1]#ffeatures[batch % num_examples]
        subject_batch[add+fin-debut] = subject[fin-1]#subject[batch % num_examples]
        labels_batch[add+fin-debut] = labels[fin-1]#labels[batch % num_examples]
        add=add+1
    return features_batch, subject_batch, labels_batch

if __name__ == "__main__":

    #file_logger = FileLogger('out.tsv', ['curr_epoch', 'train_cost', 'val_cost', 'val_acc'])
    graph = tf.Graph()
    with graph.as_default():
        # batch_size and max_step_size can vary along each step
        #0#inputs = tf.placeholder(tf.float32, [None, None, num_features])
        #inputs=tf.placeholder(tf.float32, shape=(batch_size, num_features, max_input_length))
        inputs0=tf.placeholder(tf.float32, shape=(batch_size, max_input_length, num_features))
        # inputs = tf.transpose(inputs, [0, 2, 1]) #  inputs must be a `Tensor` of shape: `[batch_size, max_time, ...]`
        # inputs = tf.transpose(inputs, [2, 0, 1]) # [max_time, batch_size, features] to split:
        inputs = tf.transpose(inputs0, [1, 0, 2]) # [max_time, batch_size, features] to split:
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        inputs = tf.split(axis=0, num_or_size_splits=max_input_length, value=inputs)  # n_steps * (batch_size, features)
        y = tf.placeholder(tf.float32, shape=(batch_size, num_classes))  # -> seq2seq!
        cell = tf.contrib.rnn.LSTMCell(num_units=num_hidden, state_is_tuple=True)
        #cell = tf.contrib.grid_rnn.Grid2LSTMCell(num_units=num_hidden, state_is_tuple=True)
    # Forward direction cell
        #lstm_fw_cell = tf.contrib.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True, forget_bias=1.0)
    # Backward direction cell
        #lstm_bw_cell = tf.contrib.BasicLSTMCell(num_units=num_hidden, state_is_tuple=True, forget_bias=1.0)
        stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
        inputs=[tf.reshape(input_, [batch_size, num_features]) for input_ in inputs]
        outputs, _ = tf.nn.static_rnn(stack, inputs,  dtype=tf.float32)
        #try:
            ##outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
        #except Exception: # Old TensorFlow version only returns outputs not states
            ##outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,  dtype=tf.float32)

        shape = tf.shape(inputs)
        batch_s, max_time_steps = shape[0], shape[1]
        output=outputs[-1]
        weights = tf.Variable(tf.random_uniform([num_hidden, num_classes]))
        bias = tf.Variable(tf.random_uniform([num_classes]))
        y_out =  tf.matmul(output, weights) + bias#tf.nn.softmax
        #tf.contrib.layers.summarize_variables()
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=y))  # prediction, target
        #optimizer = tf.train.AdamOptimizer(mlearning_rate).minimize(cost)
        optimizer = tf.train.MomentumOptimizer(learning_rate=mlearning_rate, momentum=0.95).minimize(cost)
        eqeval = tf.equal(tf.argmax(y_out, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(eqeval, tf.float32))
        # add TensorBoard summaries for all variables
        #tf.scalar_summary('train/cost', cost)
        #tf.scalar_summary('train/accuracy', accuracy)


    with tf.Session(graph=graph) as session:
        ffeatures, subject, labels, trainclasses=read_data(True)
        testfeatures, testsubject, testlabels, testclasses=read_data(False)
        train_inputs, train_subjects, train_targets = next_training_batch(ffeatures, subject, labels, 0)
        #print("type input {}".format(type(train_inputs))+ "type target {} ".format(type(train_targets)))
                #num_classes= len(trainclasses)
        print("initialisation")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        #summaries_path = "tensorboard/%s/logs" % (timestamp)
        #summaries = tf.merge_all_summaries()
        #summarywriter = tf.train.SummaryWriter(summaries_path, sess.graph)

        try:saver = tf.train.Saver(tf.global_variables())
        except:saver = tf.train.Saver(tf.global_variables())
        checkpoint = tf.train.latest_checkpoint(checkpoint_dir="checkpoints")
        if checkpoint:
            print("LOADING " + checkpoint + " !!!")
            try:saver.restore(session, checkpoint)
            except: print("incompatible checkpoint")
        tf.global_variables_initializer().run()
        print("Begin training")
        for curr_epoch in range(num_epochs):
            train_cost = train_ler = 0
            start = time.time()

            print("epoch {}/{} start at {} for {} training and {} tests".format(curr_epoch + 1, num_epochs, start, num_train_examples, num_test_examples))
            #num_batches_per_epoch=len(ffeatures) / batch_size
            for batch in range(num_batches_per_epoch):
                train_inputs, train_subjects, train_targets = next_training_batch(ffeatures, subject, labels, batch)
                #print("type input {}".format(type(train_inputs))+ "type target {} ".format(type(train_targets)))
                #if type(train_inputs) is Tensor:
                #     train_inputs=np.array((train_inputs.eval())
                #if type(train_targets) is Tensor:
                #     train_targets=np.array((train_targets.eval())
                #fetches_train = [ cost, accuracy, summaries]
                feed = { inputs0: train_inputs, y: train_targets }

                #batch_cost, _,msumma = session.run([cost, optimizer,summaries], feed_dict=feed)
                batch_cost, _ = session.run([cost, optimizer], feed_dict=feed)
                train_cost += batch_cost * batch_size
                #summarywriter.add_summary(msumma, curr_epoch*batch)

            testfeatures_b, testsubject_b, testlabels_b=next_test_batch(testfeatures, testsubject, testlabels, curr_epoch)
            val_feed = { inputs0: testfeatures_b,
                        y: testlabels_b }
            test_pred, test_acc, test_cost = session.run([y_out, accuracy, cost], feed_dict=val_feed)
            #file_logger.write([curr_epoch + 1,train_cost,test_cost,test_acc])
            #test_pred = session.run(y_out[0], feed_dict=feed)
            #dense_decoded = tf.sparse_tensor_to_dense(test_pred, default_value=-1).eval(session=session)
            dense_decoded = test_pred[0]#tf.sparse_tensor_to_dense(test_pred, default_value=-1).eval(session=session)
            print("pred {} ".format(dense_decoded)+" Labels {} ".format(testlabels_b)+ " accuracy {} ".format(test_acc)+"  cost {} ".format(test_cost)+" train cost ".format(train_cost))
            if curr_epoch % save_step == 0 and curr_epoch > 0:
                snapshot="trainepoch{}".format(curr_epoch)
                print("SAVING snapshot %s" % snapshot)
                saver.save(session, "checkpoints/" + snapshot + ".ckpt", curr_epoch)

        testlabel=[]
        testresult=[]
        testids=[]
        testpredvalue=[]
        testlabelsvalue=[]
        mequal=0;
        for curr_epoch in range(num_test_examples):
            testfeatures_b, testsubject_b, testlabels_b=next_continuous_test_batch(testfeatures, testsubject, testlabels, curr_epoch)
            val_feed = { inputs0: testfeatures_b,
                        y: testlabels_b }
            test_pred, test_acc, test_cost = session.run([y_out, accuracy, cost], feed_dict=val_feed)
            testresult.append(test_pred[0])
            testlabel.append(testlabels_b[0])
            testids.append(curr_epoch)
            argpred=np.argmax(test_pred[0], 0)
            arglabel=np.argmax(testlabels_b[0], 0)
            if argpred==arglabel:
                mequal=mequal+1
            testpredvalue.append(argpred)
            testlabelsvalue.append(arglabel)
            
        preds_df = pd.DataFrame(testresult, columns=CLASSES)
        ids_test_df = pd.DataFrame(testids, columns=["id"])
        testlabel_df= pd.DataFrame(testlabelsvalue, columns=["label"])
        testresult_df= pd.DataFrame(testpredvalue, columns=["prediction"])
        submission = pd.concat([ids_test_df, preds_df, testresult_df, testlabel_df], axis=1)
        submission.to_csv('testresult_mlp.csv', index=False)
        print("SAVING snapshot {}".format(mequal)+"/{}".format(num_test_examples))

