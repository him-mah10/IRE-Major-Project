import tensorflow as tf
import numpy as np
import os
import datetime
import re
import time
import _pickle as cPickle
from model import Attention

tf.flags.DEFINE_integer("max_sentence_length", 30, "Max sentence length in train/test data")
tf.flags.DEFINE_integer("max_trigger_length", 7, "Max trigger length in train/test data")
tf.flags.DEFINE_integer("hidden_size", 256, "Size of LSTM hidden layer")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")

FLAGS = tf.flags.FLAGS

def get_data(sents):
    sent1_word_index = []
    sent1_dist_index = []
    trigger1_word_index = []
    trigger1_dist_index = []
    sent2_word_index = []
    sent2_dist_index = []
    trigger2_word_index = []
    trigger2_dist_index = []
    label = []
    time_diff = []

    pic_in = open(file_path, "rb")
    # test_sents = cPickle.load(pic_in)
    # embedding_matrix = cPickle.load(pic_in)
    
    for item in sents:
        sent1_word_index.append(item[0])
        sent1_dist_index.append(item[1])
        trigger1_word_index.append(item[2])
        trigger1_dist_index.append(item[3])
        sent2_word_index.append(item[4])
        sent2_dist_index.append(item[5])
        trigger2_word_index.append(item[6])
        trigger2_dist_index.append(item[7]) 
        label.append(item[8])
        time_diff.append(item[9])
    
    return sent1_word_index, sent1_dist_index, trigger1_word_index, trigger1_dist_index, sent2_word_index, sent2_dist_index, trigger2_word_index, trigger2_dist_index, label, time_diff


def make_batches(train_sent1_word_index, train_sent1_dist_index, train_trigger1_word_index, train_trigger1_dist_index, train_sent2_word_index, train_sent2_dist_index, train_trigger2_word_index, train_trigger2_dist_index, label, time_diff ):
    batches = []
    # batches.append([xtrain_batch[],ytrain_batch[]])
    n = len(label)
    for i in range(0, n, FLAGS.batch_size):
        batch = []
        batch.append(train_sent1_word_index[i:min(i+FLAGS.batch_size, n-1)])
        batch.append(train_sent1_dist_index[i:min(i+FLAGS.batch_size, n-1)])
        batch.append(train_trigger1_word_index[i:min(i+FLAGS.batch_size, n-1)])
        batch.append(train_trigger1_dist_index[i:min(i+FLAGS.batch_size, n-1)])
        batch.append(train_sent2_word_index[i:min(i+FLAGS.batch_size, n-1)])
        batch.append(train_sent2_dist_index[i:min(i+FLAGS.batch_size, n-1)])
        batch.append(train_trigger2_word_index[i:min(i+FLAGS.batch_size, n-1)])
        batch.append(train_trigger2_dist_index[i:min(i+FLAGS.batch_size, n-1)])
        batch.append(label[i:min(i+FLAGS.batch_size, n-1)])
        batch.append(time_diff[i:min(i+FLAGS.batch_size, n-1)])
    
        batches.append(batch)

    return batches

def train(file_path):
    pic_in = open(file_path, "rb")
    train_sents = cPickle.load(pic_in)
    test_sents = cPickle.load(pic_in)
    # embedding_matrix = cPickle.load(pic_in)
    
    train_sent1_word_index, train_sent1_dist_index, train_trigger1_word_index, train_trigger1_dist_index, train_sent2_word_index, train_sent2_dist_index, train_trigger2_word_index, train_trigger2_dist_index, train_label, train_time_diff = get_data(train_sents)
    test_sent1_word_index, test_sent1_dist_index, test_trigger1_word_index, test_trigger1_dist_index, test_sent2_word_index, test_sent2_dist_index, test_trigger2_word_index, test_trigger2_dist_index, test_label, test_time_diff = get_data(test_sents)

    # vocal_count = embedding_matrix.shape[0]

##------------------------------------------------------------------------------------------------
    ## PADDING DATA

    for i in range(len(train_sent1_word_index)):
        train_sent1_word_index[i] = np.pad(train_sent1_word_index[i], pad_width=(0,FLAGS.max_sentence_length - len(train_sent1_word_index[i])), mode='constant', constant_values=(0, FLAGS.max_sentence_length))

    for i in range(len(train_sent1_dist_index)):
        train_sent1_dist_index[i] = np.pad(train_sent1_dist_index[i], pad_width=(0,FLAGS.max_sentence_length - len(train_sent1_dist_index[i])), mode='constant', constant_values=(0, FLAGS.max_sentence_length))

    for i in range(len(train_trigger1_word_index)):
        train_trigger1_word_index[i] = np.pad(train_trigger1_word_index[i], pad_width=(0,FLAGS.max_trigger_length - len(train_trigger1_word_index[i])), mode='constant', constant_values=(0, FLAGS.max_trigger_length))

    for i in range(len(train_trigger1_word_index)):
        train_trigger1_dist_index[i] = np.pad(train_trigger1_dist_index[i], pad_width=(0,FLAGS.max_trigger_length - len(train_trigger1_dist_index[i])), mode='constant', constant_values=(0, FLAGS.max_trigger_length))

    for i in range(len(train_sent2_word_index)):
        train_sent2_word_index[i] = np.pad(train_sent2_word_index[i], pad_width=(0,FLAGS.max_sentence_length - len(train_sent2_word_index[i])), mode='constant', constant_values=(0, FLAGS.max_sentence_length))

    for i in range(len(train_sent2_dist_index)):
        train_sent2_dist_index[i] = np.pad(train_sent2_dist_index[i], pad_width=(0,FLAGS.max_sentence_length - len(train_sent2_dist_index[i])), mode='constant', constant_values=(0, FLAGS.max_sentence_length))

    for i in range(len(train_trigger2_word_index)):
        train_trigger2_word_index[i] = np.pad(train_trigger2_word_index[i], pad_width=(0,FLAGS.max_trigger_length - len(train_trigger2_word_index[i])), mode='constant', constant_values=(0, FLAGS.max_trigger_length))

    for i in range(len(train_trigger2_word_index)):
        train_trigger2_dist_index[i] = np.pad(train_trigger2_dist_index[i], pad_width=(0,FLAGS.max_trigger_length - len(train_trigger2_dist_index[i])), mode='constant', constant_values=(0, FLAGS.max_trigger_length))



    for i in range(len(test_sent1_word_index)):
        test_sent1_word_index[i] = np.pad(test_sent1_word_index[i], pad_width=(0,FLAGS.max_sentence_length - len(test_sent1_word_index[i])), mode='constant', constant_values=(0, FLAGS.max_sentence_length))

    for i in range(len(test_sent1_dist_index)):
        test_sent1_dist_index[i] = np.pad(test_sent1_dist_index[i], pad_width=(0,FLAGS.max_sentence_length - len(test_sent1_dist_index[i])), mode='constant', constant_values=(0, FLAGS.max_sentence_length))

    for i in range(len(test_trigger1_word_index)):
        test_trigger1_word_index[i] = np.pad(test_trigger1_word_index[i], pad_width=(0,FLAGS.max_trigger_length - len(test_trigger1_word_index[i])), mode='constant', constant_values=(0, FLAGS.max_trigger_length))

    for i in range(len(test_trigger1_word_index)):
        test_trigger1_dist_index[i] = np.pad(test_trigger1_dist_index[i], pad_width=(0,FLAGS.max_trigger_length - len(test_trigger1_dist_index[i])), mode='constant', constant_values=(0, FLAGS.max_trigger_length))

    for i in range(len(test_sent2_word_index)):
        test_sent2_word_index[i] = np.pad(test_sent2_word_index[i], pad_width=(0,FLAGS.max_sentence_length - len(test_sent2_word_index[i])), mode='constant', constant_values=(0, FLAGS.max_sentence_length))

    for i in range(len(test_sent2_dist_index)):
        test_sent2_dist_index[i] = np.pad(test_sent2_dist_index[i], pad_width=(0,FLAGS.max_sentence_length - len(test_sent2_dist_index[i])), mode='constant', constant_values=(0, FLAGS.max_sentence_length))

    for i in range(len(test_trigger2_word_index)):
        test_trigger2_word_index[i] = np.pad(test_trigger2_word_index[i], pad_width=(0,FLAGS.max_trigger_length - len(test_trigger2_word_index[i])), mode='constant', constant_values=(0, FLAGS.max_trigger_length))

    for i in range(len(test_trigger2_word_index)):
        test_trigger2_dist_index[i] = np.pad(test_trigger2_dist_index[i], pad_width=(0,FLAGS.max_trigger_length - len(test_trigger2_dist_index[i])), mode='constant', constant_values=(0, FLAGS.max_trigger_length))

##-----------------------------------------------------------------------------------
# TRAINING DATA 

    vocab_count = 1000
    X_train = []
    Y_train = []
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # sequence_length, trigger_length, num_classes, vocab_size, word_embedding_size ,dist_embedding_size, hidden_size, attention_size, decay_rate
            
            ##--------- CREATE MODEL----------
            
            model = Attention(sequence_length= FLAGS.max_sentence_length,
                        trigger_length = FLAGS.max_trigger_length,
                        num_classes = FLAGS.max_sentence_length,
                        vocab_size = vocab_count,
                        word_embedding_size = 100,
                        dist_embedding_size = 14,
                        hidden_size = FLAGS.hidden_size,
                        attention_size = 128,
                        decay_rate = 0.9
                    )
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.loss, global_step=global_step)

            batches = make_batches(train_sent1_word_index, train_sent1_dist_index, train_trigger1_word_index, train_trigger1_dist_index, train_sent2_word_index, train_sent2_dist_index, train_trigger2_word_index, train_trigger2_dist_index, label, time_diff)

            ##------------ TRAIN BATCHES --------------

            for batch in batches:
                feed_dict = {
                    model.input1_text1: batch[0],
                    model.input1_text2: batch[1],
                    model.trigger1_text1: batch[2],
                    model.trigger1_text2: batch[3],
                    model.input2_text1: batch[4],
                    model.input2_text2: batch[5],
                    model.trigger2_text1: batch[6],
                    model.trigger2_text2: batch[7],
                    model.labels = batch[8],
                    model.V_d = batch[9]
                    ## TODO :   set model.V_d and model.V_w
                }

                _, step, loss = sess.run([train_op, global_step, model.loss], feed_dict)
                
                # if step % FLAGS.evaluate_every == 0:
                #     print("\nEvaluation:")
                

            

if __name__ == "__main__":
  file_path = "dump_file"
  train(file_path)