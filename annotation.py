from divide_text import *
from txtprocessor import plot_loss


def get_info_list(datasetname):
    data=processjson(datasetname)
    output=list()
    for sample in data:
        tmp=list()
        passage = sample['description']
        if 'ami' in datasetname:
            sentence_list=passage.split('\n')
        else:
            sentence_list=divide_sentences(passage)
        for index,sentence in enumerate(sentence_list):
            if index<2:
                continue
            tmp.append(len(tokenizer.encode(sentence)))
        output.append(tmp)
    return output

def get_loss_list(infolist, datasetname):
    filename='losses_%s.pkl' %datasetname
    losslist = pickle.load(open(filename,'rb'))
    output=list()
    p = 0
    for infoitem in infolist:
        length=len(infoitem)
        tmp =losslist[p:p+length]
        output.append(tmp)
        p=p+length
    return output


def autodivide(lossinfo,sentenceinfo):

    divide_list=list()
    index_skjt=0 #sakujitsu
    index_otti=0 #ototoi
    tight_min=400
    loose_min=750
    max_length=1150
    topnum=int(0.05*len(lossinfo))
    maxvalue=sorted(lossinfo,reverse=True)[:topnum]
    # print(maxvalue)
    index = 1
    while(index<len(sentenceinfo)):
        now_length = sum(sentenceinfo[index_skjt:index])
        if now_length <tight_min:
            if lossinfo[index] in maxvalue and len(divide_list) :
                if sum(sentenceinfo[index_otti:index])<=max_length:
                    if lossinfo[index]>=lossinfo[divide_list[-1]]:
                        divide_list[-1]=index
                        # print('case1_1: divide id: %d loss:%f' %(index,lossinfo[index]))
                        index_skjt=index
                        index+=1
                    else:
                        index+=1
                    continue
                else:
                    tmp_index=index
                    tmp_min=sum(sentenceinfo[index_otti:index])*0.4
                    tmp_max=sum(sentenceinfo[index_otti:index])*0.6
                    while(sum(sentenceinfo[index_otti:tmp_index])>tmp_max):
                        tmp_index-=1
                    max_loss=0
                    while(1):
                        if lossinfo[tmp_index]>max_loss:
                            divideindex=tmp_index
                            max_loss=lossinfo[tmp_index]
                        tmp_length=sum(sentenceinfo[index_otti:tmp_index])
                        tmp_index-=1
                        if tmp_length<tmp_min or tmp_index==0:
                            break
                    divide_list[-1]=divideindex
                    divide_list.append(index)
                    # print('case1_2: divide id: %d %d' %(divideindex,index))
                    index_otti=divideindex
                    index_skjt=index                  
                    index+=1
                    continue
            else:
                index+=1
                continue
        
        if now_length >=tight_min and now_length< loose_min:
            if lossinfo[index] in maxvalue:
                divide_list.append(index)
                # print('case2: divide id: %d loss:%f' %(index,lossinfo[index]))
                index_skjt=index
                index_otti=index_skjt
                index+=1
            else:
                index+=1
            continue
        if now_length>=loose_min:
            max_loss=0
            tmp_length=now_length
            tmp_index=index
            while(1):
                if tmp_index>=len(sentenceinfo):
                    break
                if tmp_length>max_length:
                    break                
                if lossinfo[tmp_index]>=max_loss:
                    # print('maxloss=%f, tmp_index=%d, lossinfo[tmp_index]=%f' %(max_loss, tmp_index, lossinfo[tmp_index]))
                    divideindex=tmp_index
                    max_loss=lossinfo[tmp_index]
                tmp_length=tmp_length+sentenceinfo[tmp_index]
                tmp_index+=1

            divide_list.append(divideindex)
            # print('case3: divide id: %d loss:%f' %(divideindex,lossinfo[divideindex]))
            index=divideindex
            index_skjt=divideindex
            index_otti=index_skjt

    divide_list=[index+2 for index in divide_list]
    return divide_list

def main():
    datasetname='ami_train'
    allinfolist=get_info_list(datasetname)
    alllosslist=get_loss_list(allinfolist,datasetname)
    rawdata=processjson(datasetname)
    print(len(rawdata)==len(allinfolist))
    newset=list()
    infolist=list()
    for index in range(len(allinfolist)):
        divide_list = autodivide(alllosslist[index],allinfolist[index])
        _,textlist=divide_text(rawdata[index]['description'],divide_list)
        infolist.append(len(textlist))
        for text in textlist:
            tmp=dict()
            tmp['description']=text
            tmp['abstract']=rawdata[index]['abstract']
            newset.append(tmp)
    print('****finish processing****total %d samples***' %len(newset))
    jsonfile = open('./textfile/%s_anno.json'%datasetname,'w' )
    for sample in newset:
        json.dump(sample,jsonfile)
    pickle.dump(infolist, open('%s_anno.pkl' %datasetname, 'wb'))


def demo():
    datasetname='ami_test'
    allinfolist=get_info_list(datasetname)
    alllosslist=get_loss_list(allinfolist,datasetname)
    print('***finish preprocessing info and loss***')
    index=0
    divide_list= autodivide(alllosslist[index],allinfolist[index])
    plot_loss(alllosslist[index],divide_list)   

if __name__=='__main__':
    demo()