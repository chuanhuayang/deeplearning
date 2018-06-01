#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import jieba

def segment(sen):
  if not isinstance(sen, unicode):
    sen = sen.decode('utf-8')
  rs = jieba.cut(sen)
  words = []
  for w in rs:
    if len(w.strip()) > 0:
      words.append(w)
  return words

def segment_line(sen):
  words = segment(sen)
  return " ".join(words)