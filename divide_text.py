import re
from test_text import *
import string
import spacy
from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def divide_sentences(text):
    tool = spacy.load('en_core_web_sm')
    processed_text = tool(text)
    sentence_list=[]
    for sentence in processed_text.sents:
        sentence_list.append(str(sentence))
    return sentence_list

def list2txt(sentence_list,start=None,end=None):
    '''
    sentences[start:end]
    '''
    if start==None:
        start=0
    if end==None:
        end=len(sentence_list)

    txt=''
    if isinstance(start,list):
        start = sorted(start)
        for index in start:
            txt=txt+sentence_list[index]
    else:
        if start>=end:
            return ''
        for i in range(start,end):
            txt=txt+sentence_list[i]
    return txt

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

def get_divide_index(text,metric=None):
    result = list()
    if metric=='length':
        max_length = 1024
        sentence_list = divide_sentences(text)
        token_buffer = list()
        for index,sentence in enumerate(sentence_list):
            token_buffer = token_buffer + tokenizer.encode(sentence)
            if len(token_buffer)> max_length:
                result.append(index)
                token_buffer =tokenizer.encode(sentence)
    return result

# divide_text(text_04,[9,18])

