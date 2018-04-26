#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import cPickle as pickle
import random
import numpy as np
import tensorflow as tf
import time
import sys
from keras.preprocessing import sequence
import codecs
import os
import logging
from data_lstm import DataModel

#hypyer-parameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 50)")
tf.flags.DEFINE_integer("rnn_dim", 200, "Number of rnn_dim (default: 100)")
tf.flags.DEFINE_float("learning_rate", 0.001, "Leanring rate for optimization (default: 0.001)")
tf.flags.DEFINE_float("uniform_min", -0.25, "uniform distribute min value")
tf.flags.DEFINE_float("uniform_max", 0.25, "uniform distribute max value")
tf.flags.DEFINE_integer("sequence_length", 30, "Max sequence length of input (default: 25)")
#training
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 512)")
tf.flags.DEFINE_integer("num_epochs", 4, "Number of training epochs (default: 20)")


# tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 100)")
# tf.flags.DEFINE_integer("valid_every", 100, "evaluate model after this many steps (default: 100)")
#
# #path
# tf.flags.DEFINE_string("train_path", "./data/sogou_train_segment.txt", "Training data path a folder")
# tf.flags.DEFINE_string("valid_path", "./data/sogou_valid_segment.txt", "Valid data path a folder")
# tf.flags.DEFINE_string("test_path", "./data/sogou_test_segment.txt", "Test data path a folder")
# tf.flags.DEFINE_string("word_to_index_path", "wv/5000w_word_to_index_50.pkl", "word to index path")
# tf.flags.DEFINE_string("index_to_vector_path", "wv/5000w_index_to_vector_50.pkl", "index to vector path")

tf.flags.DEFINE_integer("checkpoint_every", 10, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("valid_every", 2, "evaluate model after this many steps (default: 100)")
#path
tf.flags.DEFINE_string("train_path", "./data/sogou_train_segment_tmp.txt", "Training data path a folder")
tf.flags.DEFINE_string("valid_path", "./data/sogou_valid_segment_tmp.txt", "Valid data path a folder")
tf.flags.DEFINE_string("test_path", "./data/sogou_test_segment_tmp.txt", "Test data path a folder")
tf.flags.DEFINE_string("word_to_index_path", "wv/word_to_index_50_0929.pkl", "word to index path")
tf.flags.DEFINE_string("index_to_vector_path", "wv/index_2_vec_50_0929.pkl", "index to vector path")


#config
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
#embedding
tf.flags.DEFINE_boolean('if_emb_train', False, 'If word embedding trainable')
tf.flags.DEFINE_boolean('if_emb', True, 'If use pre-trained word vectors')
#gpu
tf.flags.DEFINE_string("gpu_id", '0', 'visible gpu for training')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id


path_suf = "_e0g1"

if not os.path.exists('./Model/model' + path_suf):
    os.makedirs('./Model/model'+ path_suf)
model_path = './Model/model' + path_suf + '/' + 'dual_encoder_model'

with open('./Model/model' + path_suf + '/' + 'parameters', 'w') as f:
    for attr, value in sorted(FLAGS.__flags.items()):
        print "{}={}".format(attr.upper(), value)
        f.write("{}={}".format(attr.upper(), value) + '\n')
log_path = './Model/model' + path_suf + '/' + 'log'
summary_path = './Model/model' + path_suf + '/' + 'summary_log'
print 'model_path: %s, log_path: %s' % (model_path, log_path)
logging.basicConfig(filename=log_path,
                    level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%y-%m-%d %h:%m:%s")

def load_data(path, word_to_index):
  data = []
  sep = " |<->| "
  with codecs.open(path, "r", "utf-8") as inf:
    for line in inf:
      sentences = map(lambda x: x.split(), line.strip().split(sep))
      if len(sentences) < 3:
        continue
      token_context = map(lambda x: word_to_index[x] if x in word_to_index else 1, sentences[0])
      padded_token_context = sequence.pad_sequences([token_context], maxlen=FLAGS.sequence_length, padding='post', truncating='post', value=0)[0]

      token_uttrance = map(lambda x: word_to_index[x] if x in word_to_index else 1, sentences[1])
      padded_token_uttrance = sequence.pad_sequences([token_uttrance], maxlen=FLAGS.sequence_length, padding='post', truncating='post', value=0)[0]
      lengths = [len(token_context), len(token_uttrance)]
      label = int(sentences[2][0])
      data.append([padded_token_context, padded_token_uttrance, lengths, [label]])
  return data


def main(_):
  print 'loading word_to_index ...'
  word_to_index = pickle.load(open(FLAGS.word_to_index_path))
  # word_to_index = {u"你好":0, u"晚安":1}
  vocab_size = len(word_to_index)
  print 'size of the vocab is', vocab_size
  with tf.Graph().as_default():
    # data_model = DataModel(vocab_size, FLAGS.embedding_dim, FLAGS.if_emb_train, FLAGS.rnn_dim, FLAGS.sequence_length, FLAGS.learning_rate)
    data_model = DataModel(vocab_size, FLAGS.embedding_dim, FLAGS.if_emb_train, FLAGS.rnn_dim, FLAGS.sequence_length, FLAGS.learning_rate)
    print data_model.input_context.name
    print data_model.input_utterance.name
    print data_model.target.name
    print data_model.context_len.name
    print data_model.utterance_len.name
    print data_model.probs.name

    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=20)
    print 'initializing global and local variables ...'
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    with open('./Model/model' + path_suf + '/' + 'parameters', 'aw') as otf:
      total_parameters = 0
      for variable in data_model.trainable_variables:
        otf.write(str(variable) + '\n')
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
          variable_parametes *= dim.value
        total_parameters += variable_parametes
      print 'number of parameters in the network', total_parameters


    print 'loading embedding ...'
    if FLAGS.if_emb:
      print 'loading index_to_vector'
      index_to_vector = pickle.load(open(FLAGS.index_to_vector_path))
      # index_to_vector = {"0": np.random.normal(0,1,[FLAGS.embedding_dim]),
      #                    "1": np.random.normal(0, 1, [FLAGS.embedding_dim]),
      #                    }
      sess.run(data_model.embedding_init,feed_dict={ data_model.embedding_placeholder: index_to_vector.values() })

    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(summary_path, sess.graph)
    step_count = 0
    for epoch in range(FLAGS.num_epochs):
      print "**************************************** start epoch %d ****************************************" % epoch
      logging.info("**************************************** start epoch %d ****************************************" % epoch)
      avg_cost = 0.0
      avg_auc = 0
      avg_acc = 0
      start = time.time()
      train_data = load_data(FLAGS.train_path, word_to_index)
      valid_data = load_data(FLAGS.valid_path, word_to_index)
      steps_train = len(train_data) // FLAGS.batch_size
      random.shuffle(train_data)
      for step in range(steps_train):
        batch_data = train_data[FLAGS.batch_size * step: FLAGS.batch_size * (step + 1)]
        batch_context = [x[0] for x in batch_data]

        batch_utterance = [x[1] for x in batch_data]
        batch_context_length = [x[2][0] for x in batch_data]
        batch_utterance_length = [x[2][1] for x in batch_data]
        batch_target = [x[3] for x in batch_data]
        feed_dict = {
          data_model.input_context: batch_context,
          data_model.input_utterance: batch_utterance,
          data_model.target: batch_target,
          data_model.context_len: batch_context_length,
          data_model.utterance_len: batch_utterance_length
        }
        _, c, auc, acc = sess.run([data_model.optimizer, data_model.losses, data_model.update_auc, data_model.accuracy],feed_dict=feed_dict)
        print "epoch: %d,  step: %d loss: %f, auc: %f, acc: %f" % (epoch, step + 1, c, auc, acc)

        avg_cost += c
        avg_auc += auc
        avg_acc += acc
        if (step + 1) % FLAGS.valid_every == 0:
          step_count += FLAGS.valid_every * 100
          summary = sess.run(merged_summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary, step_count)
          summary_writer.flush()
          print "step: %d, average loss: %f, auc: %f, acc: %f,  time: %f" % (step+1, avg_cost / FLAGS.valid_every, avg_auc / FLAGS.valid_every, avg_acc / FLAGS.valid_every, time.time() - start)
          logging.info("step: %d, average loss: %f, auc: %f, acc: %f, time: %f" % (step+1, avg_cost / FLAGS.valid_every, avg_auc / FLAGS.valid_every, avg_acc / FLAGS.valid_every, time.time() - start))
          avg_cost = 0.0
          avg_auc = 0.0
          avg_acc = 0.0
          start = time.time()
        if (step + 1) % FLAGS.checkpoint_every == 0:
          print "evaluating loss on valid data"
          random.shuffle(valid_data)
          steps_valid = len(valid_data) // FLAGS.batch_size
          avg_cost_valid = 0
          avg_auc_valid = 0
          avg_acc_valid = 0
          for s in range(steps_valid):
            batch_data_valid = valid_data[FLAGS.batch_size * s: FLAGS.batch_size * (s+1)]
            batch_context_valid = [x[0] for x in batch_data_valid]
            batch_context_valid_len = [x[2][0] for x in batch_data_valid]
            batch_utterance_valid = [x[1] for x in batch_data_valid]
            batch_utterance_valid_len = [x[2][1] for x in batch_data_valid]
            batch_target_valid = [x[3] for x in batch_data_valid]
            c, auc, acc = sess.run([data_model.losses,data_model.update_auc, data_model.accuracy], feed_dict={
              data_model.input_context: batch_context_valid,
              data_model.input_utterance: batch_utterance_valid,
              data_model.target: batch_target_valid,
              data_model.context_len: batch_context_valid_len,
              data_model.utterance_len: batch_utterance_valid_len,
            })
            avg_auc_valid += auc
            avg_cost_valid += c
            avg_acc_valid += acc
          avg_cost_valid /= steps_valid
          avg_auc_valid /= steps_valid
          avg_acc_valid /= steps_valid
          print "*** on valid set loss: %f, auc: %f, acc: %f ***" % (avg_cost_valid, avg_auc_valid, avg_acc_valid)
          logging.info("*** on valid set loss: %f, auc: %f, acc: %f ***" % (avg_cost_valid, avg_auc_valid, avg_acc_valid))
          save_path = saver.save(sess, model_path, global_step=step)
          print "Model saved in file: %s" % save_path
    summary_writer.close()
    print "evaluating loss on test data"
    test_data = load_data(FLAGS.test_path, word_to_index)
    steps_test = len(test_data) // FLAGS.batch_size
    avg_cost_test = 0
    avg_auc_test = 0
    avg_acc_test = 0
    # 重新初始化本地变量
    # sess.run(tf.local_variables_initializer())
    for step in range(steps_test):
      batch_data_test = test_data[FLAGS.batch_size * step: FLAGS.batch_size * (step + 1)]
      batch_context_test = [x[0] for x in batch_data_test]
      batch_utterance_test = [x[1] for x in batch_data_test]
      batch_target_test = [x[3] for x in batch_data_test]
      batch_context_test_len = [x[2][0] for x in batch_data_test]
      batch_utterance_test_len = [x[2][1] for x in batch_data_test]
      c, auc, acc= sess.run([data_model.losses, data_model.update_auc, data_model.accuracy], feed_dict={
        data_model.input_context: batch_context_test,
        data_model.input_utterance: batch_utterance_test,
        data_model.target: batch_target_test,
        data_model.context_len: batch_context_test_len,
        data_model.utterance_len: batch_utterance_test_len,
      })
      avg_cost_test += c
      avg_auc_test += auc
      avg_acc_test += acc
    avg_cost_test /= steps_test
    avg_auc_test /= steps_test
    avg_acc_test /= steps_test
    print "*** on test set loss: %f, auc: %f, acc: %f***" % (avg_cost_test, avg_auc_test, avg_acc_test)
    logging.info("*** on test set loss: %f, auc: %f, acc: %f ***" % (avg_cost_test, avg_cost_test, avg_acc_test))

if __name__ == "__main__":
  tf.app.run()