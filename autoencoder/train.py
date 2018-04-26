#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
import sklearn.preprocessing as prep
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from autoencoder import AdditiveGaussianNoiseAutoencoder
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


def stand_scalar(X_train):
  preprocessor = prep.StandardScaler().fit(X_train)
  return preprocessor



def get_random_block_from_data(data, batch_size):
  start_index = np.random.randint(0, len(data)-batch_size)
  return data[start_index: (start_index + batch_size)]



def show_one_image(data, width=None, height=None):
  if width is not None and height is not None:
    data = data.reshape(width,height)
  plt.imshow(data, cmap='binary')  # 黑白显示
  plt.show()

def save_one_image(data, width=None, height=None):
  if width is not None and height is not None:
    data = data.reshape(width,height)
  plt.imshow(data, cmap='binary')  # 黑白显示

def main():
  mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
  test_x = mnist.test.images[:10]
  test_y = mnist.test.labels[:10]
  preprocessor = stand_scalar(mnist.train.images)
  # X_train, X_test = preprocessor.transform(mnist.train.images), preprocessor.transform(mnist.test.images)
  X_train, X_test = mnist.train.images, mnist.test.images

  n_samples = int(mnist.train.num_examples)
  train_epochs = 60
  batch_size = 128
  display_step = 1
  restruct_step = 1

  autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=32, transfer_function=tf.nn.relu,active_function=tf.nn.sigmoid,
                                                 optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.01), scale=0.01)
  test(test_x, test_y, 0)
  for epoch in range(train_epochs):
    avg_cost = 0
    total_batch = int(n_samples/batch_size)
    for i in range(total_batch):
      batch_xs = get_random_block_from_data(X_train, batch_size)
      cost = autoencoder.partial_fit(batch_xs)
      avg_cost += cost / n_samples
    # reconstruct input image
    if epoch % restruct_step == 0:
      reconstruction = autoencoder.reconstruct(test_x)
      # reconstruction = preprocessor.inverse_transform(reconstruction)
      test(reconstruction, test_y, epoch + 1)
    if epoch % display_step == 0:
      print "Epoch: %04d, cost=%.9f" % (epoch+1, avg_cost)





def test(data, label, epoch = 0):
  image_name = "mnist_train/image_%d_%d_%d.bmp"
  iid = 0
  for x, label in zip(data, label):
    label = np.argmax(label)
    x *= 255
    x = np.array(x, dtype="uint8")
    x = x.reshape(28,28)
    im = Image.fromarray(x.reshape(28,28))
    path = image_name % (iid, label, epoch)
    im.save(path)
    iid += 1
if __name__ == "__main__":
  main()

