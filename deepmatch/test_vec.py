#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import codecs
from collections import defaultdict

import numpy as np

import segment

data = []
length_sen = []
rs_count = {"0": 0, "1": 0}
sen_count = defaultdict(lambda :0)
with codecs.open("data/atec_nlp_sim_train.txt", "r", "utf-8-sig") as inf:
  for line in inf:
    s1, s2, score = line.strip().split("\t")
    length_sen.append(len(s1))
    length_sen.append(len(s2))
    rs_count[score] += 1
    sen_count[s1] += 1
    sen_count[s2] += 1
    data.append([s1, s2, score])
  print "sample number is : %d, positive number is: %d, negative number is: %d" % (np.sum(rs_count.values()), rs_count['1'], rs_count['0'])
  print "max_length: %f, min_length: %f, avg_length: %f" % (np.max(length_sen), np.min(length_sen), np.average(length_sen))

  # 可以构造一部分负样本出来，正样本或许也可以弄一点出来，不过先不管了
  # count = 0
  # for key in sen_count:
  #   if sen_count[key] > 1:
  #     print count, key, sen_count[key]
  #     count += 1
segment_data = []
segment_length = []
long_count = 0
for x in data:
  s1, s2 = segment.segment(x[0]), segment.segment(x[1])
  segment_data.append([s1, s2, x[2]])
  segment_length.append(len(s1))
  segment_length.append(len(s2))
  if len(s1) > 50 or len(s2) > 50:
    long_count += 1
    print long_count, "\t".join(x)

print "max_segment_length: %f, min_length: %f, avg_length: %f" % (np.max(segment_length), np.min(segment_length), np.average(segment_length))

with codecs.open("data/atec_nlp_sim_segment.txt", "w", "utf-8") as otf:
  for x in segment_data:
    otf.write(" ".join(x[0]) + "\t" + " ".join(x[1]) + "\t" + x[2] + "\n")




