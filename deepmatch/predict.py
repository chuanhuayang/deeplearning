#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from keras.preprocessing import sequence
import tensorflow as tf
import pickle
import numpy as np
import segment
from sklearn import metrics
import sys
def segment_data(data):
  copy_data = []
  for item in data:
    x = [segment.segment_line(item[0]), segment.segment_line(item[1])]
    x.extend(item[2:])
    copy_data.append(x)
  return np.array(copy_data)

class Predicter(object):
  def __init__(self, batch_size, embedding_size, sequence_length,  model_path, word_to_index_path):
    self.batch_size = batch_size
    self.embedding_size = embedding_size
    self.sequence_length = sequence_length
    # self.checkpoint_file_rank = tf.train.latest_checkpoint(model_path)
    self.checkpoint_file_rank = "Model/model_e0g1/dual_encoder_model_14_28_29-199"
    self.word_to_index = pickle.load(open(word_to_index_path))
    self.SEP = "\t"


  def transform_input_data(self, source_data, is_training_data=False):
    vector_data = []
    for item in source_data:
      token_context = map(lambda x: self.word_to_index[x] if x in self.word_to_index else 1, item[0])
      padded_token_context = sequence.pad_sequences([token_context], maxlen=self.sequence_length, padding='post', truncating='post', value=0)[0]

      token_uttrance = map(lambda x: self.word_to_index[x] if x in self.word_to_index else 1, item[1])
      padded_token_uttrance = sequence.pad_sequences([token_uttrance], maxlen=self.sequence_length, padding='post', truncating='post', value=0)[0]
      lengths = [len(token_context), len(token_uttrance)]
      if is_training_data:
        label = int(item[2])
        vector_data.append([padded_token_context, padded_token_uttrance, lengths, [label]])
      else:
        vector_data.append([padded_token_context, padded_token_uttrance, lengths])
    return vector_data

  def data_loader(self, data_path):
    with open(data_path) as inf:
      tmp_data = []
      for line in inf:
        arr = line.strip().split(self.SEP)[1:]
        tmp_data.append(arr) #post, resp, label
        if len(tmp_data) == self.batch_size:
          batch_data = tmp_data
          tmp_data = []
          yield  batch_data
      yield tmp_data

  def save_data(self, data, path):
    with open(path, "w") as writer:
      count = 1
      for item in data:
        tokens = [str(x) for x in item]
        line = str(count) + self.SEP + self.SEP.join(tokens) + "\n"
        writer.write(line)
        count += 1
  def predict(self, input_path, output_path, threshold, is_training_data):
    graph = tf.Graph()
    with graph.as_default():
      sess = tf.Session()
      saver_rank = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file_rank))
      saver_rank.restore(sess, self.checkpoint_file_rank)

      input_context = graph.get_operation_by_name('input_context').outputs[0]
      input_utterance = graph.get_operation_by_name('input_utterance').outputs[0]
      input_context_len = graph.get_operation_by_name('context_len').outputs[0]
      input_utterance_len = graph.get_operation_by_name('utterance_len').outputs[0]
      predict_probs = graph.get_operation_by_name('probs').outputs[0]
      is_training = graph.get_operation_by_name("is_training").outputs[0]

      iterator = self.data_loader(input_path)
      text_data = []
      labels = []
      probs = []
      for data in iterator:
        batch_xs = segment_data(data)
        batch_data = self.transform_input_data(batch_xs, is_training_data)
        batch_context = [x[0] for x in batch_data]

        batch_utterance = [x[1] for x in batch_data]
        batch_context_length = [x[2][0] for x in batch_data]
        batch_utterance_length = [x[2][1] for x in batch_data]

        prob = sess.run([predict_probs],
                                      feed_dict={
                                        input_context: batch_context,
                                        input_utterance: batch_utterance,
                                        input_context_len: batch_context_length,
                                        input_utterance_len: batch_utterance_length,
                                        is_training: False
                                      })
        if is_training_data:
          batch_target = [x[3] for x in batch_data]
          batch_target = np.array(batch_target).reshape(-1)
          labels.extend(batch_target)
        prob = np.array(prob).reshape(-1)
        probs.extend(prob)
        predication = [1 if s >= threshold else 0 for s in probs]
        text_data.extend(data)
      result = [[y] for x, y in zip(text_data, predication)]
      if is_training_data and len(labels) == len(probs):
        auc = metrics.roc_auc_score(labels, probs)

        accuracy = metrics.accuracy_score(labels, predication)
        precision = metrics.precision_score(labels, predication)
        recall = metrics.recall_score(labels, predication)
        f1 = metrics.f1_score(labels, predication)
        print "threshold :%f, precision : %f, recall: %f, auc: %f, accuracy: %f, f1: %f" % (threshold, precision, recall, auc, accuracy, f1)
        result = [[x[0], x[1], label, y] for x, label, y in zip(text_data, labels, predication)]
      self.save_data(result, output_path)

def process(input_path, output_path, threshold=0.1, is_training_data=False):
  predictor = Predicter(batch_size=256,embedding_size=50, sequence_length=100, model_path="./Model/model_e0g1/", word_to_index_path= "wv/5000w_word_to_index_50.pkl")
  predictor.predict(input_path, output_path, threshold, is_training_data)


if __name__ =="__main__":
  # for t in np.arange(0.01, 0.5, 0.01):
  #     process("data/test.txt", "data/test_labeled.txt", threshold=t, is_training_data=True)
  process("data/atec_nlp_sim_valid.csv", "data/atec_nlp_sim_valid_result.csv", threshold=0.09, is_training_data=True)
  # process(sys.argv[1], sys.argv[2], threshold=0.09, is_training_data=False)
