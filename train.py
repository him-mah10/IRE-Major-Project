import tensorflow as tf
import numpy as np
import os
import datetime
import re
import time
import _pickle as cPickle
from model import Attention
from sklearn.metrics import f1_score, precision_score, recall_score

tf.flags.DEFINE_integer("max_sentence_length", 38, "Max sentence length in train/test data")
tf.flags.DEFINE_integer("max_trigger_length", 7, "Max trigger length in train/test data")
tf.flags.DEFINE_integer("hidden_size", 128, "Size of LSTM hidden layer")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_float("learning_rate", 0.01, "Which learning rate to start with.")
tf.flags.DEFINE_integer("display_every", 1000, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

FLAGS = tf.flags.FLAGS
checkpoint_dir = ""
checkpoint_prefix = ""

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
    test_label = []
    trigger_common = []
    time_diff = []
    print("sents " + str(len(sents)))
    for item in sents:
        # print(len(item))
        sent1_word_index.append(item[0])
        sent1_dist_index.append([val+38 for val in item[1]])
        trigger1_word_index.append(item[2])
        trigger1_dist_index.append([val+38 for val in item[3]])
        sent2_word_index.append(item[4])
        sent2_dist_index.append([val+38 for val in item[5]])
        trigger2_word_index.append(item[6])
        trigger2_dist_index.append([val+38 for val in item[7]])
        label.append([item[8], 1-item[8]])
        test_label.append(item[8])
        trigger_common.append([float(item[9])])
        time_diff.append([float(item[10])])

    return sent1_word_index, sent1_dist_index, trigger1_word_index, trigger1_dist_index, sent2_word_index, sent2_dist_index, trigger2_word_index, trigger2_dist_index, label, trigger_common, time_diff, test_label


def make_batches(train_sent1_word_index, train_sent1_dist_index, train_trigger1_word_index, train_trigger1_dist_index, train_sent2_word_index, train_sent2_dist_index, train_trigger2_word_index, train_trigger2_dist_index, label, trigger_common,time_diff, test_label):
    batches = []
    n = len(label)
    print(n)
    for i in range(0, n, FLAGS.batch_size):
        batch = []
        batch.append(train_sent1_word_index[i:min(i+FLAGS.batch_size, n)])
        batch.append(train_sent1_dist_index[i:min(i+FLAGS.batch_size, n)])
        batch.append(train_trigger1_word_index[i:min(i+FLAGS.batch_size, n)])
        batch.append(train_trigger1_dist_index[i:min(i+FLAGS.batch_size, n)])
        batch.append(train_sent2_word_index[i:min(i+FLAGS.batch_size, n)])
        batch.append(train_sent2_dist_index[i:min(i+FLAGS.batch_size, n)])
        batch.append(train_trigger2_word_index[i:min(i+FLAGS.batch_size, n)])
        batch.append(train_trigger2_dist_index[i:min(i+FLAGS.batch_size, n)])
        batch.append(label[i:min(i+FLAGS.batch_size, n)])
        batch.append(test_label[i:min(i+FLAGS.batch_size, n)])
        batch.append(trigger_common[i:min(i+FLAGS.batch_size, n)])
        batch.append(time_diff[i:min(i+FLAGS.batch_size, n)])

        batches.append(batch)

    return batches

def train():
    with tf.device('/gpu:0'):
            
        global checkpoint_dir
        train_sent1_word_index, train_sent1_dist_index, train_trigger1_word_index, train_trigger1_dist_index, train_sent2_word_index, train_sent2_dist_index, train_trigger2_word_index, train_trigger2_dist_index, train_label, train_trigger_common, train_time_diff, train_test_label = get_data(train_sents)
        # test_sent1_word_index, test_sent1_dist_index, test_trigger1_word_index, test_trigger1_dist_index, test_sent2_word_index, test_sent2_dist_index, test_trigger2_word_index, test_trigger2_dist_index, test_label, test_trigger_common, test_time_diff = get_data(test_sents)

        vocab_count = embedding_matrix.shape[0]
        print(vocab_count)

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

        print("doneee")
    ##-----------------------------------------------------------------------------------
    # TRAINING DATA
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto()
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # sequence_length, trigger_length, num_classes, vocab_size, word_embedding_size ,dist_embedding_size, hidden_size, attention_size, coref_size, decay_rate

                ##--------- CREATE MODEL----------

                model = Attention(sequence_length= FLAGS.max_sentence_length,
                            trigger_length = FLAGS.max_trigger_length,
                            num_classes = 2,
                            vocab_size = vocab_count,
                            word_embedding_size = 100,
                            dist_embedding_size = 14,
                            hidden_size = FLAGS.hidden_size,
                            attention_size = 128,
                            co_ref_size = 128,
                        )
                global_step = tf.Variable(0, name="global_step", trainable=False)
                train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(model.loss, global_step=global_step)

                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
                print("Writing to {}\n".format(out_dir))

                loss_summary = tf.summary.scalar("loss", model.loss)
                acc_summary = tf.summary.scalar("accuracy", model.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables())
                print("-----------------------------------------------------------------------------------------------")
                print(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(model.W_em1.assign(embedding_matrix))
                #sess.run(model.embedding_init, feed_dict = {model.embedding_placeholder: embedding_matrix})
                batches = make_batches(train_sent1_word_index, train_sent1_dist_index, train_trigger1_word_index, train_trigger1_dist_index, train_sent2_word_index, train_sent2_dist_index, train_trigger2_word_index, train_trigger2_dist_index, train_label,train_trigger_common,train_time_diff, train_test_label)
                print(len(batches))
                ##------------ TRAIN BATCHES --------------
                for i in range(0,1):
                    print("Epoch number: " + str(i))
                    for batch in batches:
                        # print(len(batches))
                        #print(batch[9])
                        feed_dict = {
                            model.input1_text1: batch[0],
                            model.input1_text2: batch[1],
                            model.trigger1_text1: batch[2],
                            model.trigger1_text2: batch[3],
                            model.input2_text1: batch[4],
                            model.input2_text2: batch[5],
                            model.trigger2_text1: batch[6],
                            model.trigger2_text2: batch[7],
                            model.labels: batch[8],
                            model.V_w: batch[10],
                            model.V_d: batch[11],
                            model.bsz_size: len(batch[0])
                        }
                        _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op,model.loss, model.accuracy], feed_dict)
                        #print(W_em1[0])
                        train_summary_writer.add_summary(summaries, step)
                        if step % FLAGS.display_every == 0:
                            time_str = datetime.datetime.now().isoformat()
                            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                        print(step)
                        if step % 100 == 0:
                            path = saver.save(sess, checkpoint_prefix, global_step=step)
                            print("Saved model checkpoint to {}\n".format(path))
                        #break
                    # if step % FLAGS.evaluate_every == 0:
                    #     print("\nEvaluation:")
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    print("Saved model checkpoint to {}\n".format(path))

def eval():
    with tf.device('/gpu:0'):

        # train_sent1_word_index, train_sent1_dist_index, train_trigger1_word_index, train_trigger1_dist_index, train_sent2_word_index, train_sent2_dist_index, train_trigger2_word_index, train_trigger2_dist_index, train_label, train_trigger_common, train_time_diff = get_data(train_sents)
        test_sent1_word_index, test_sent1_dist_index, test_trigger1_word_index, test_trigger1_dist_index, test_sent2_word_index, test_sent2_dist_index, test_trigger2_word_index, test_trigger2_dist_index, test_label, test_trigger_common, test_time_diff, test_label2 = get_data(test_sents)
        vocab_count = embedding_matrix.shape[0]
        print(vocab_count)

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

        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        # print(checkpoint_dir)
        # print(checkpoint_file)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto()
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
                input1_text1 = graph.get_operation_by_name("input1_text1").outputs[0]
                print(input1_text1)
                input1_text2 = graph.get_operation_by_name("input1_text2").outputs[0]
                trigger1_text1 = graph.get_operation_by_name("trigger1_text1").outputs[0]
                trigger1_text2 = graph.get_operation_by_name("trigger1_text2").outputs[0]
                input2_text1 = graph.get_operation_by_name("input2_text1").outputs[0]
                input2_text2 = graph.get_operation_by_name("input2_text2").outputs[0]
                trigger2_text1 = graph.get_operation_by_name("trigger2_text1").outputs[0]
                trigger2_text2 = graph.get_operation_by_name("trigger2_text2").outputs[0]
                V_w = graph.get_operation_by_name("trigger_common_words").outputs[0]
                V_d = graph.get_operation_by_name("events_date_diff").outputs[0]
                bsz_size = graph.get_operation_by_name("bsz_size").outputs[0]
                predictions = graph.get_operation_by_name("predictions").outputs[0]
                #embedding_init = graph.get_operation_by_name("init").outputs[0]
                #embedding_placeholder = graph.get_operation_by_name("placeholder").outputs[0]
                #sess.run(embedding_init, feed_dict = {embedding_placeholder: embedding_matrix})
                batches = make_batches(test_sent1_word_index, test_sent1_dist_index, test_trigger1_word_index, test_trigger1_dist_index, test_sent2_word_index, test_sent2_dist_index, test_trigger2_word_index, test_trigger2_dist_index, test_label,test_trigger_common,test_time_diff,test_label2)
                all_predictions = []
                for batch in batches:
                    print("btchno")
                    feed_dict = {
                        input1_text1: batch[0],
                        input1_text2: batch[1],
                        trigger1_text1: batch[2],
                        trigger1_text2: batch[3],
                        input2_text1: batch[4],
                        input2_text2: batch[5],
                        trigger2_text1: batch[6],
                        trigger2_text2: batch[7],
                        V_w: batch[10],
                        V_d: batch[11],
                        bsz_size: len(batch[0]),
#                        embedding_placeholder: embedding_matrix,
                    }
                    batch_predictions = sess.run(predictions, feed_dict)
                    #print(len(batch_predictions))
                    print(batch_predictions)
                    all_predictions = np.concatenate([all_predictions, batch_predictions])
                # final_label = []
                # for val in test_label:
                #     final_label.append(val[0])
                # print(final_label)
                correct_predictions = 0
                for i in range(len(all_predictions)):
                    if (all_predictions[i] == test_label2[i]):
                        correct_predictions += 1
                #print(correct_predictions)
                cnt = 0
                for i in range(len(all_predictions)):
                    if(all_predictions[i] == 0):
                        pass
                    else:
                        cnt +=1
                #print(cnt)
#                print(set(test_label2)-set(correct_predictions))

                # correct_predictions = float(sum(all_predictions == test_label2))
                print("\nTotal number of test examples: {}".format(len(test_label2)))
                print("Accuracy: {:g}\n".format(correct_predictions / float(len(test_label2))))
                f1_score1 = f1_score(test_label2, all_predictions)
                prec_score = precision_score(test_label2, all_predictions)
                rec_score = recall_score(test_label2, all_predictions)
                print("F1 score : ", f1_score1, prec_score, rec_score)            

file_path = "dump_file"
pic_in = open(file_path, "rb")
train_sents = cPickle.load(pic_in)
test_sents = cPickle.load(pic_in)
embedding_matrix = cPickle.load(pic_in)
train()
#out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs"))
#checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
#checkpoint_prefix = os.path.join(checkpoint_dir, "model")
#if not os.path.exists(checkpoint_dir):
#	os.makedirs(checkpoint_dir)

eval()
