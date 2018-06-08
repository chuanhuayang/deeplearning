#!/usr/bin/env python
# encoding: utf-8

import re
import os
import jieba

re_eng = re.compile(r'\w[\w\s]+\w')
re_white = re.compile(r'\s+')


class Tokenizer(object):
    miuiad_dict_path = os.path.abspath(os.path.join(os.path.realpath(__file__),
                                                    '../dicts/miui_ad_100w_custome.dict'))
    misearch_dict_path = os.path.abspath(os.path.join(os.path.realpath(__file__),
                                                    '../dicts/miglobal_search.dict'))

    def __init__(self):
        self.dt_misearch = None
        self.dt_miuiad = None
        self.dt_posseg = None

    def choose_dt(self, dicttype):
        if dicttype == 'miuiad':
            if self.dt_miuiad is None:
                self.dt_miuiad = jieba.Tokenizer()
                self.dt_miuiad.load_userdict(self.miuiad_dict_path)
            dt = self.dt_miuiad
        elif dicttype == 'misearch':
            if self.dt_misearch is None:
                self.dt_misearch = jieba.Tokenizer()
                self.dt_misearch.load_userdict(self.misearch_dict_path)
            dt = self.dt_misearch
        elif dicttype == 'posseg':
            if self.dt_posseg is None:
                import jieba.posseg as pseg
                self.dt_posseg = pseg
            dt = self.dt_posseg
        else:
            dt = jieba.dt
        return dt

    def cut(self, s, dicttype='misearch'):
        dt = self.choose_dt(dicttype)
        return list(dt.cut(s))


tokenizer = Tokenizer()


def segment(s, dicttype='misearch'):
    """
    分词
    :param s: unicode or str 
    :param dicttype: miuiad or misearch
    :return: list of unicode
    """
    words = tokenizer.cut(s, dicttype)
    result = filter(lambda w: w!=' ', words)
    return result


def format_messy_line(line):
    TMP_ENG_DELIMITER = '|<->|'
    engs = re_eng.findall(line)
    for eng in engs:
        tmp_eng = TMP_ENG_DELIMITER.join(eng.split())
        line = line.replace(eng, tmp_eng)
    line = ''.join(line.split())
    line = line.replace(TMP_ENG_DELIMITER, ' ')
    return line


def segment_line(line, dicttype='misearch', fmt_messy=False):
    """
    会先压缩字符串的空白符，然后分词
    你 好, hello world. -> 你好 , hello world .
    :param line: unicode string 
    :param dicttype: miuiad or misearch
    :return: unicode string
    """
    if not isinstance(line, unicode):
        line = line.decode('utf-8')
    if fmt_messy:
        line = format_messy_line(line)
    line = line.strip()
    words = segment(line, dicttype)
    line = ' '.join(words)
    line = re_white.sub(' ', line)
    return line


# 处理整个文件
# 文件格式为: post |<->| resp
# 分隔符可自定义
def deal_file(infile, outfile, dicttype, fmt_messy, sep=' |<->| '):
    with open(infile) as inf:
        with open(outfile, 'w') as outf:
            count = 0
            for line in inf:
                count += 1
                if count % 10000 == 0:
                    print 'Has deal %s lines' % count
                parts = line.split(sep)
                parts = map(lambda x: segment_line(x, dicttype, fmt_messy), parts)
                line = sep.join(parts)
                line = line.encode('utf8')
                outf.write(line + '\n')
    print 'Total deal %s lines' % count


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-i', '--infile', help='source file')
    parser.add_argument('-o', '--outfile', help='segmented file')
    parser.add_argument('-t', '--dicttype', help='dict type, miuiad or misearch')
    parser.add_argument('-m', '--fmtMessy', action='store_true', default=False,
                        help='format messy line before segment, default: False,'
                             'like "你 好世界， hello world!" -> "你好世界，hello world!"')
    args = parser.parse_args()
    deal_file(args.infile, args.outfile, args.dicttype, args.fmtMessy)
