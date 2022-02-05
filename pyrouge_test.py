import nltk
from generator import *
text_num = 19

def get_summaries():
    result_file = open('generated_predictions.txt','r',encoding='utf-8').readlines()
    preds = result_file
    for index in range(text_num):
        reference = get_demo(index)['abstract']
        file=open('./ref/summ.'+str(index)+'.txt','w')
        file.write("\n".join(nltk.sent_tokenize(reference)))
        file=open('./summ/summ.'+str(index)+'.txt','w')
        file.write("\n".join(nltk.sent_tokenize(preds[index])))

get_summaries()