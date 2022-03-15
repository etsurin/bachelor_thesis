from match import *
from tqdm import tqdm

file = open('matchsta.txt','w',encoding='utf-8')
def main():
    newdata=list()
    data = processjson('my_test')
    txt=''
    for index,item in tqdm(enumerate(data)):
        if index>500:
            break
        text=item['description']
        summ=item['abstract']
        txt+=str(index)+' '+str((len(text),len(summ)))
        newsamples = generate_new_training_sample(text,summ)
        for sample in newsamples:
            newdata.append(sample)
            newtext = sample['description']
            newsumm = sample['abstract']
            txt+=' '+str((len(newtext),len(newsumm)))
        txt+='\n'
    file.write(txt)
    file.close()


if __name__=='__main__':
    main()