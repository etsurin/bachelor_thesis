import re
from test_text import *
import string
import spacy

def divide_sentences(text):
    tool = spacy.load('en_core_web_sm')
    processed_text = tool(text)
    sentence_list=[]
    for sentence in processed_text.sents:
        sentence_list.append(str(sentence))
    return sentence_list

def divide_text(text,intervals,write_file=False):
    sentence_list=divide_sentences(text)
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

