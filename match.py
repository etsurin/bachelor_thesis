from divide_text import *
from test_text import *
from tester import rouge

# def list2dict(sentence_list):
#     sentence_dict=dict()
#     for index, sentence in enumerate(sentence_list):
#         sentence_dict[str(index)]=sentence
#     return sentence_dict

# def dict2txt(dict):
#     sortedlist = sorted(dict.items(), key=lambda item:int(item[0]))
#     txt = ''
#     for snippet in sortedlist:
#         txt=txt+snippet[1]+' '
#     return txt

def match(text,target):
    target_list = divide_sentences(target)
    sublist=list()
    tmp_sublist=list()
    type='rouge2'
    while(1):
        tmp_sublist=sublist
        for index,item in enumerate(target_list):
            if index in sublist:
                pass
            tmp_txt = list2txt(target_list,tmp_sublist)
            txt=list2txt(target_list,sublist)+item
            if rouge(text,txt,[type])[type]>rouge(text,tmp_txt,[type])[type]:
                # target_list.remove(item)
                copysublist=sublist
                copysublist.append(index)
                tmp_sublist=copysublist
        if tmp_sublist==sublist:
            break
        else:
            sublist=tmp_sublist
    return list2txt(target_list,sublist)

def generate_new_training_sample(text, summary):
    '''
    '''
    divide_list = get_divide_index(text,'length')
    _,text_list=divide_text(text, divide_list)
    result = list()
    for snippet in text_list:
        tmp=dict()
        tmp['description']=snippet
        tmp['abstract']=list2txt(match(snippet,summary))
        if len(tmp['abstract']):
            result.append(tmp)
        else:
            pass
    return result

data = processjson('my_test')
testitem = data[331]

divide_list = get_divide_index(testitem['description'],'length')
_,text_list=divide_text(testitem['description'], divide_list)
summ=testitem['abstract']+'My favorite singer is Nakamori Akina, who was born in July 13th, 1965. And she led the fashion trend in the 1980s in Japan. '
for text in text_list:
    print(rouge(text,testitem['abstract']),rouge(text,summ))


    
# print(generate_new_training_sample(testitem['description'],testitem['abstract']))

        
        

