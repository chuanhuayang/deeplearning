# -*- coding: utf-8 -*-
import tensorflow as tf

class DeepMatch(object):
    def __init__(self, vocab_size, embedding_dim, if_emb_train, rnn_dim, num_layers, dropout_rate, sequence_len, learning_rate, is_training=True):



        self.x1 = tf.placeholder(tf.int32, [None, sequence_len], name="x1")
        self.x2 = tf.placeholder(tf.int32, [None, sequence_len], name="x2")

        self.label = tf.placeholder(tf.int8, [None, 1], name="label")
        self.x1_length = tf.placeholder(tf.int8, [None], name="x1_length")
        self.x2_length = tf.placeholder(tf.int8, [None], name="x2_length")

        with tf.name_scope("embedding_layer"):
            w_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], minval=-0.25, maxval=0.25),trainable=if_emb_train, name="weight_emb_x1")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
            self.embedding_init = w_embedding.assign(self.embedding_placeholder)
        x1_embedded = tf.nn.embedding_lookup(w_embedding, self.x1, name="embed_x1")
        x2_embedded = tf.nn.embedding_lookup(w_embedding, self.x2, name="embed_x2")

        # lstm cell
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(rnn_dim)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(rnn_dim)

        # dropout
        if is_training:
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))
        # multi-layer
        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

        # get the length of each sample
        self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)

        # forward and backward
        self.outputs, _, _ = tf.nn.static_bidirectional_rnn(
            lstm_cell_fw,
            lstm_cell_bw,
            self.inputs_emb,
            dtype=tf.float32,
            sequence_length=self.length
        )

        # softmax
        self.outputs = tf.reshape(tf.concat(self.outputs, 1), [-1, self.hidden_dim * 2])
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
        # Build the RNN
        with tf.variable_scope("rnn", reuse=True):
            cell = tf.nn.rnn_cell.LSTMCell(
                rnn_dim,
                forget_bias=2.0,
                use_peepholes=True,
                state_is_tuple=True)
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell,
                tf.concat([x1_embedded, utterance_embedded], 0),
                sequence_length=tf.concat([self.context_len, self.utterance_len], 0),
                dtype=tf.float32)
            encoding_context, encoding_utterance = tf.split(rnn_states.h, 2, 0)

        with tf.variable_scope("prediction", reuse=True):
            matrix_m = tf.get_variable("M", shape=[rnn_dim, rnn_dim],initializer=tf.truncated_normal_initializer())
            # "Predict" a  response: c * matrix_m
            generated_response = tf.matmul(encoding_context, matrix_m)
            generated_response = tf.expand_dims(generated_response, 2)
            encoding_utterance = tf.expand_dims(encoding_utterance, 2)

        # Dot product between generated response and actual response, (c * matrix_m) * r
        logits = tf.squeeze(tf.matmul(generated_response, encoding_utterance, True), [2])
        # Apply sigmoid to convert logits to probabilities
        self.probs = tf.sigmoid(logits, name="probs")
        # Calculate the binary cross-entropy loss
        self.losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(self.target), logits=logits, name="losses"))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.losses)
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.trainable_variables = tf.trainable_variables()
        self.auc, self.update_auc = tf.metrics.auc(self.target, self.probs, name="auc")

        correct_prediction = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.target, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("losses", self.losses)
        tf.summary.scalar("auc", self.update_auc)
        tf.summary.scalar("acc", self.accuracy)



# class DataModelDouble(object):
#     def __init__(self, vocab_size, embedding_dim, if_emb_train, rnn_dim, sequence_len, learning_rate):
#         self.input_context = tf.placeholder(tf.int32, [None, sequence_len], name="input_context")
#         self.input_utterance = tf.placeholder(tf.int32, [None, sequence_len], name="input_utterance")
#         self.target = tf.placeholder(tf.int8, [None, 1], name="target")
#         self.context_len = tf.placeholder(tf.int8, [None], name="context_len")
#         self.utterance_len = tf.placeholder(tf.int8, [None], name="utterance_len")
#
#         with tf.name_scope("embedding_layer"):
#             w_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], minval=-0.25, maxval=0.25),
#                                       trainable=if_emb_train, name="weight_emb_post")
#             self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
#             self.embedding_init = w_embedding.assign(self.embedding_placeholder)
#         context_embedded = tf.nn.embedding_lookup(w_embedding, self.input_context, name="embed_context")
#         utterance_embedded = tf.nn.embedding_lookup(w_embedding, self.input_utterance, name="embed_utterance")
#
#         # Build the RNN
#         with tf.variable_scope("rnn") as vs:
#             # We use an LSTM Cell
#             cell_context = tf.nn.rnn_cell.LSTMCell(
#                 rnn_dim,
#                 forget_bias=2.0,
#                 use_peepholes=True,
#                 state_is_tuple=True)
#             cell_utterance = tf.nn.rnn_cell.LSTMCell(
#                 rnn_dim,
#                 forget_bias=2.0,
#                 use_peepholes=True,
#                 state_is_tuple=True)
#             # Run the context
#             rnn_outputs_context, rnn_states_context = tf.nn.dynamic_rnn(
#                 cell_context,
#                 context_embedded,
#                 sequence_length=self.context_len,
#                 dtype=tf.float32)
#             # Run the utterance
#             rnn_outputs_utterance, rnn_stats_utterance = tf.nn.dynamic_rnn(
#                 cell_utterance,
#                 utterance_embedded,
#                 sequence_length=self.utterance_len,
#                 dtype=tf.float32)
#
#         with tf.variable_scope("prediction") as vs:
#             matrix_m = tf.get_variable("M", shape=[rnn_dim, rnn_dim], initializer=tf.truncated_normal_initializer())
#             # "Predict" a  response: c * matrix_m
#             generated_response = tf.matmul(rnn_states_context, matrix_m)
#             generated_response = tf.expand_dims(generated_response, 2)
#             encoding_utterance = tf.expand_dims(rnn_stats_utterance, 2)
#
#         # Dot product between generated response and actual response, (c * matrix_m) * r
#         logits = tf.squeeze(tf.matmul(generated_response, encoding_utterance, True), [2])
#         # Apply sigmoid to convert logits to probabilities
#         self.probs = tf.sigmoid(logits, name="probs")
#         # Calculate the binary cross-entropy loss
#         self.losses = tf.reduce_mean(
#             tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(self.target), logits=logits, name="losses"))
#         self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.losses)
#         self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#         self.trainable_variables = tf.trainable_variables()
#         self.auc, self.update_auc = tf.metrics.auc(self.target, self.probs, name="auc")
#         self.acc, self.update_acc = tf.metrics.accuracy(self.target, self.probs, name="acc")
#         tf.summary.scalar("losses", self.losses)
#         tf.summary.scalar("auc", self.auc)
#         tf.summary.scalar("update_auc", self.update_auc)
#         tf.summary.scalar("update_acc", self.update_acc)