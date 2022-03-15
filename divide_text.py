import re
from text import *
from tqdm import tqdm
import string
import spacy
from transformers import BartTokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
tool = spacy.load('en_core_web_sm')

def divide_sentences(text):
    processed_text = tool(text)
    sentence_list=[]
    for sentence in processed_text.sents:
        sentence_list.append(str(sentence))
    return sentence_list


def divide_text(text,intervals,write_file=False):
    if isinstance(text,str):
        sentence_list=divide_sentences(text)
    else:
        sentence_list=text
    txt=''
    for index,sentence in enumerate(sentence_list):
        if index in intervals:
            txt=txt+'[DIVIDE HERE]'
        txt=txt+sentence
    if write_file:
        txtfile=open('divide_result.txt','w')
        txtfile.write(txt)
        txtfile.close()
        return None
    return txt,txt.split('[DIVIDE HERE]')

def get_snippets(text,metric=None,tokenized=False):
    result = list()
    if metric=='length':
        max_length = 1024
        sentence_list = divide_sentences(text)
        token_buffer = list()
        txt=''
        for sentence in sentence_list:
            token_buffer = token_buffer
            tmp = tokenizer.encode(sentence)
            if len(token_buffer+tmp)> max_length:
                if tokenized:
                    result.append(token_buffer)
                else:
                    result.append(txt)
                token_buffer = tmp
                txt=sentence
            else:
                token_buffer = token_buffer + tmp
                txt=txt+sentence
        if tokenized:
            result.append(token_buffer)
        else:
            result.append(txt)
    return result

def main():
    data=processjson('traincut_duplication')


if __name__=='__main__':
    main()
