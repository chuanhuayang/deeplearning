import tensorflow as tf
import numpy as np
import time
x = tf.placeholder(tf.int32, [None, 1000])
embedding = tf.get_variable('embedding', shape=[100, 256])
x_embedding = tf.nn.embedding_lookup(embedding,x)
source_sentence_length = tf.placeholder(tf.int32, [None])
encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=256)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, x_embedding, sequence_length=source_sentence_length, dtype=tf.float32)
with tf.Session() as sess:
  tf.global_variables_initializer().run()
  X_batch = np.random.randint(0, 100, size=[512, 1000])
  time0 = time.time()
  for i in range(100):
    encoder_outputs.eval(feed_dict={x: X_batch, source_sentence_length: [10]*512})
    time1 = time.time()
    print('sequence_length_10, time: %.9f' % (time1-time0))
    time2 = time.time()
  for i in range(100):
      encoder_outputs.eval(feed_dict={x: X_batch, source_sentence_length: [1000]*512})
      time3 = time.time()
      print('sequence_length_1000, time: %.9f' % (time3-time2))