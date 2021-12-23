import re
from test_text import *
import string

def divide_sentences(text):
    sentence_list = re.split('([.?])',text)
    sentence_list.append("")
    sentence_list = ["".join(i) for i in zip(sentence_list[0::2],sentence_list[1::2])]
    del sentence_list[-1]
    return len(sentence_list),sentence_list

def divide_text(text,intervals,write_file=False):
    _,sentence_list=divide_sentences(text)
    txt=''
    for index,sentence in enumerate(sentence_list):
        if index in intervals:
            txt=txt+'\n\n'
        txt=txt+sentence
    if write_file:
        txtfile=open('divide_result.txt','w')
        txtfile.write(txt)
        txtfile.close()
        return None
    return txt,txt.split('\n\n')

divide_text(text_04,[9,18],1)

