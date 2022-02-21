from test_text import *
from transformers import BartTokenizer
from divide_text import *
from tqdm import tqdm
from match import *
device = 'cuda'
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

data = processjson('my_test')
txt=''
for index, item in tqdm(enumerate(data)):
    len_text = len(tokenizer.encode(item['description']))
    len_summ = len(tokenizer.encode(item['abstract']))
    len_text_s = len(divide_sentences(item['description']))
    len_summ_s = len(tokenizer.encode(item['abstract']))
    txt+=str(len_text)+','+str(len_summ)+','+str(len_text_s)+','+str(len_summ_s)+'\n'

file = open('statistics.txt','r',encoding='utf-8')
file.write(txt)
file.close()