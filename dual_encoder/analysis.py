#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import codecs
import numpy as np
pos_data = []
neg_data = []
sep = " |<->| "
count = 0
with codecs.open("data_m1g2/micloud_chatbot_sogou_random1000_20180226_evaluated_sorted.txt", "r", "utf-8") as inf:
  otf = codecs.open("data_m1g2/micloud_chatbot_sogou_random1000_20180226_evaluated_sorted_0.8.txt", "w", 'utf-8')
  count = 0
  for line in inf:
    score, post, resp, label = line.strip().split(sep)
    score = float(score)
    if label == "1":
      pos_data.append([score, label])
    else:
      neg_data.append([score, label])
    if 0.4 < score < 0.6:
      count +=1
    if label !="1" and score > 0.8:
      otf.write(line)
    if label == "1" and score < 0.1:
      print count, ":", line,
      count += 1
  otf.close()

print count
print "positive samples：" , len(pos_data)
print "negative samples：" , len(neg_data)
pos_data = np.array(pos_data, dtype=np.float32)
neg_data = np.array(neg_data, dtype=np.float32)
plt.xlim(0,1)
# plt.ylim(0, 70)
plt.title("micloud_chatbot_sogou_random1000_20180226_evaluated_sorted")
plt.hist(pos_data[:,0], 10, density=True, facecolor='g', alpha=0.75)
plt.hist(neg_data[:,0], 10, density=True, facecolor='r', alpha=0.75)



# x = np.random.normal(0,1,100000)
# plt.hist(x,100, density=True, facecolor='g', alpha=0.75)
# plt.xlim(0,1)
# plt.xlabel("relevance score")
# plt.ylabel("sample frequence")


plt.show()