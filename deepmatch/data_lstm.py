# -*- coding: utf-8 -*-
import tensorflow as tf

class DataModel(object):
    def __init__(self, vocab_size, embedding_dim, if_emb_train, rnn_dim, sequence_len, learning_rate, delta):
        self.input_context = tf.placeholder(tf.int32, [None, sequence_len], name="input_context")
        self.input_utterance = tf.placeholder(tf.int32, [None, sequence_len], name="input_utterance")
        self.target = tf.placeholder(tf.int8, [None, 1], name="target")
        self.context_len = tf.placeholder(tf.int8, [None], name="context_len")
        self.utterance_len = tf.placeholder(tf.int8, [None], name="utterance_len")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        with tf.name_scope("embedding_layer"):
            w_embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], minval=-0.25, maxval=0.25),trainable=if_emb_train, name="weight_emb_post")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
            self.embedding_init = w_embedding.assign(self.embedding_placeholder)
        context_embedded = tf.nn.embedding_lookup(w_embedding, self.input_context, name="embed_context")
        utterance_embedded = tf.nn.embedding_lookup(w_embedding, self.input_utterance, name="embed_utterance")
        # Build the RNN
        with tf.variable_scope("rnn") as vs:
            # We use an LSTM Cell

            cell = tf.contrib.rnn.LSTMCell(
                rnn_dim,
                forget_bias=2.0,
                use_peepholes=True,
                state_is_tuple=True)
            if self.is_training is True:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.75)
            # Run the utterance and context through the RNN
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell,
                tf.concat([context_embedded, utterance_embedded], 0),
                sequence_length=tf.concat([self.context_len, self.utterance_len], 0),
                dtype=tf.float32)
            encoding_context, encoding_utterance = tf.split(rnn_states.h, 2, 0)

        # a implement of neural tensor network
        with tf.variable_scope("ntn"):
            k = 3
            d=100
            W = tf.Variable(tf.truncated_normal([d, d, k]))
            V = tf.Variable(tf.zeros([k, 2 * d]))
            b = tf.Variable(tf.zeros([k, 1]))
            U = tf.Variable(tf.ones([1, k]))

            e1v = tf.transpose(encoding_context)
            e2v = tf.transpose(encoding_utterance)
            num_rel_r = tf.expand_dims(tf.shape(e1v)[1], 0)
            pre_activation= list()

            print("e1v: " + str(e1v.get_shape()))
            print("W[:,:,slice]: " + str(W[:, :, 0].get_shape()))
            print("e2v: " + str(e2v.get_shape()))

            # print("Starting preactivation funcs")
            for slice in range(k):
                pre_activation.append(tf.reduce_sum(e1v * tf.matmul(W[:, :, slice], e2v), 0))

            pre_activation= tf.stack(pre_activation)

            temp2_pos = tf.matmul(V, tf.concat([e1v, e2v], 0))

            # print("   temp2_pos: "+str(temp2_pos.get_shape()))
            pre_activation_ = pre_activation + temp2_pos + b

            # print("Starting activation funcs")
            activation = tf.tanh(pre_activation_)

            score_pos = tf.reshape(tf.matmul(U, activation), num_rel_r)
            logits = tf.expand_dims(score_pos, 1)
            print "score:" + str(logits.get_shape())


        # with tf.variable_scope("prediction"):
        #     matrix_m = tf.get_variable("M", shape=[rnn_dim, rnn_dim],initializer=tf.truncated_normal_initializer())
        #     # "Predict" a  response: c * matrix_m
        #     generated_response = tf.matmul(encoding_context, matrix_m)
        #     generated_response = tf.expand_dims(generated_response, 2)
        #     encoding_utterance = tf.expand_dims(encoding_utterance, 2)
        #     # Dot product between generated response and actual response, (c * matrix_m) * r
        #     logits = tf.squeeze(tf.matmul(generated_response, encoding_utterance, True), [2])

        print "logits:" + str(logits.get_shape())
        # Apply sigmoid to convert logits to probabilities
        self.probs = tf.sigmoid(logits, name="probs")
        # Calculate the binary cross-entropy loss

        tv = tf.trainable_variables()  # 得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
        regularization_cost = delta * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])  # 0.001是lambda超参数
        self.losses = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(self.target), logits=logits)) + regularization_cost
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.losses)
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.trainable_variables = tf.trainable_variables()
        tf.summary.scalar("loss", self.losses)
