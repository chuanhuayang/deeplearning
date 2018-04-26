#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from keras.preprocessing import sequence
import codecs
import segment
import tensorflow as tf
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

#training
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 512)")
tf.flags.DEFINE_integer("sequence_length", 30, "Batch Size (default: 512)")

tf.flags.DEFINE_string("word_to_index_path", "wv/5000w_word_to_index_50.pkl", "word to index path")
tf.flags.DEFINE_string("index_to_vector_path", "wv/5000w_index_to_vector_50.pkl", "index to vector path")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

checkpoint_file_rank = tf.train.latest_checkpoint("./Model/model_e1g2/")
word_to_index_rank = pickle.load(open(FLAGS.word_to_index_path))
source_data_path = "data_m1g2/micloud_chatbot_sogou_random1000_20180226_evaluated.txt"
SEP = " |<->| "
batch_size = 512
threshold = 0.1

def transform_input_data(source_data, word_to_index, is_training_data=False):
  vector_data = []
  for item in source_data:
    token_context = map(lambda x: word_to_index[x] if x in word_to_index else 1, item[0])
    padded_token_context = sequence.pad_sequences([token_context], maxlen=FLAGS.sequence_length, padding='post', truncating='post', value=0)[0]

    token_uttrance = map(lambda x: word_to_index[x] if x in word_to_index else 1, item[1])
    padded_token_uttrance = sequence.pad_sequences([token_uttrance], maxlen=FLAGS.sequence_length, padding='post', truncating='post', value=0)[0]
    lengths = [len(token_context), len(token_uttrance)]
    if is_training_data:
      label = int(item[2])
    else:
      label = item[2]
    vector_data.append([padded_token_context, padded_token_uttrance, lengths, [label]])
  return vector_data

def eval_data_generate(path, batch_size):
  with codecs.open(path, "r", "utf-8") as inf:
    tmp_data = []
    for line in inf:
      line = line.split(SEP)
      if len(line) < 3:
        print line
        continue
      tmp_data.append([line[0], line[1], line[2]]) #2 is source or label
      if len(tmp_data) == batch_size:
        batch_data = tmp_data
        tmp_data = []
        yield  batch_data
    yield tmp_data


def segment_data(data):
  copy_data = []
  for item in data:
    copy_data.append([segment.segment(item[0]), segment.segment(item[1]), item[2]])
  return copy_data


def post_process_data(data, y_scores, threshold, remain_path, delete_path):
  try:
    remain_writer = codecs.open(remain_path, "a+", "utf-8")
    delete_writer = codecs.open(delete_path, "a+", "utf-8")
    for y_score, item in zip(y_scores, data):
      line = SEP.join(["%.8f" % y_score, item[0], item[1]]) + "\n"
      if y_score < threshold:
        delete_writer.write(line)
      else:
        remain_writer.write(line)
  except Exception as e:
    print e.message

def save_data(data, path):
  with codecs.open(path, "w", "utf-8") as writer:
    for item in data:
      item = [str(x) for x in item]
      line = SEP.join(item) + "\n"
      writer.write(line)

def predict():
  graph = tf.Graph()
  with graph.as_default():
    sess = tf.Session()
    saver_rank = tf.train.import_meta_graph("{}.meta".format(checkpoint_file_rank))
    saver_rank.restore(sess, checkpoint_file_rank)

    input_context = graph.get_operation_by_name('input_context').outputs[0]
    input_utterance = graph.get_operation_by_name('input_utterance').outputs[0]
    input_target = graph.get_operation_by_name('target').outputs[0]
    input_context_len = graph.get_operation_by_name('context_len').outputs[0]
    input_utterance_len = graph.get_operation_by_name('utterance_len').outputs[0]
    predict_probs = graph.get_operation_by_name('probs').outputs[0]

    iterator = eval_data_generate(source_data_path, batch_size)
    step = 0
    sort_data = []
    result_dict = {"tp":0.0, "fn":0.0, "fp":0.0, "tn":0.0 }
    for data in iterator:
      batch_xs = segment_data(data)
      batch_data = transform_input_data(batch_xs, word_to_index_rank, is_training_data=True)
      batch_context = [x[0] for x in batch_data]

      batch_utterance = [x[1] for x in batch_data]
      batch_context_length = [x[2][0] for x in batch_data]
      batch_utterance_length = [x[2][1] for x in batch_data]
      batch_target = [x[3] for x in batch_data]
      prob = sess.run(predict_probs,
                                    feed_dict={
                                      input_context: batch_context,
                                      input_utterance: batch_utterance,
                                      # input_target: batch_target,
                                      input_context_len: batch_context_length,
                                      input_utterance_len: batch_utterance_length
                                    })
      print "step %d, process %d sample" % (step + 1, (step + 1) * 512)
      prob = prob.flatten()
      assert len(prob) == len(data)
      sort_data.extend([[y, item[0], item[1], item[2].strip()] for y, item in zip(prob, data)])
      step += 1
      batch_target = map(lambda s: 1 if s == 1 else 0, np.array(batch_target).flatten())
      prediction = map(lambda s: 1 if s > 0.2 else 0,  prob)
      for y, y_ in zip(batch_target, prediction):
        if y == 1 and y_ == 1:
          result_dict["tp"] += 1
        elif y == 1 and y_ == 0:
          result_dict['fn'] += 1
        elif y == 0 and y_ == 1:
          result_dict['fp'] += 1
        else:
          result_dict['tn'] += 1

    print "acc: %f" % ((result_dict['tp'] + result_dict['tn']) / (result_dict['tp'] + result_dict['tn'] + result_dict['fp'] + result_dict['fn']))
    sort_data = sorted(sort_data, reverse=True)
    save_data(sort_data, "data_m1g2/micloud_chatbot_sogou_random1000_20180226_evaluated_sorted.txt")

def process(input_path, output_path):
  graph = tf.Graph()
  with graph.as_default():
    sess = tf.Session()
    saver_rank = tf.train.import_meta_graph("{}.meta".format(checkpoint_file_rank))
    saver_rank.restore(sess, checkpoint_file_rank)

    input_context = graph.get_operation_by_name('input_context').outputs[0]
    input_utterance = graph.get_operation_by_name('input_utterance').outputs[0]
    input_context_len = graph.get_operation_by_name('context_len').outputs[0]
    input_utterance_len = graph.get_operation_by_name('utterance_len').outputs[0]
    predict_probs = graph.get_operation_by_name('probs').outputs[0]

    iterator = eval_data_generate(input_path, batch_size)
    step = 0
    sort_data = []
    for data in iterator:
      batch_xs = segment_data(data)
      batch_data = transform_input_data(batch_xs, word_to_index_rank, is_training_data=False)
      batch_context = [x[0] for x in batch_data]

      batch_utterance = [x[1] for x in batch_data]
      batch_context_length = [x[2][0] for x in batch_data]
      batch_utterance_length = [x[2][1] for x in batch_data]
      prob = sess.run(predict_probs,
                      feed_dict={
                        input_context: batch_context,
                        input_utterance: batch_utterance,
                        # input_target: batch_target,
                        input_context_len: batch_context_length,
                        input_utterance_len: batch_utterance_length
                      })
      print "step %d, process %d sample" % (step + 1, (step + 1) * 512)
      prob = prob.flatten()
      assert len(prob) == len(data)
      sort_data.extend([[y, item[0], item[1], item[2].strip()] for y, item in zip(prob, data)])
      step += 1
    sort_data = sorted(sort_data, reverse=True)
    save_data(sort_data, output_path)

if __name__ =="__main__":
  input_path = "/home/mi/data/zhidao_corpus/micloud_chatbot_zhidao_convs_1101_1220_remained.txt"
  output_path = "/home/mi/data/zhidao_corpus/micloud_chatbot_zhidao_convs_1101_1220_remained_sorted.txt"
  process(input_path, output_path)
  # predict()