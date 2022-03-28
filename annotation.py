from divide_text import *
from txtprocessor import plot_loss


def get_info_list(datasetname):
    data=processjson(datasetname)
    output=list()
    for sample in tqdm(data):
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
    safecount=0
    while(index<len(sentenceinfo)):
        # print(index)
        safecount+=1
        if safecount>len(sentenceinfo)+5:
            break
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
                index_otti=index_skjt
                index_skjt=index
                index+=1
            else:
                index+=1
            continue
        if now_length>=loose_min and now_length<=max_length:
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
            index=divideindex+1
            index_otti=index_skjt
            index_skjt=divideindex
            
        if now_length>=max_length:
            divide_list.append(index)
            index_otti=index_skjt
            index_skjt=index
            index=index+1
    
    if len(divide_list) and len(sentenceinfo[divide_list[-1]:]) < tight_min:
        del divide_list[-1]
    divide_list=[index+2 for index in divide_list]
    return divide_list

def main():
    datasetname='ami_test'
    # allinfolist=pickle.load(open('tmp_%s.pkl' %datasetname, 'rb'))
    allinfolist=get_info_list(datasetname)
    alllosslist=get_loss_list(allinfolist,datasetname)
    rawdata=processjson(datasetname)
    print(len(rawdata)==len(allinfolist))
    newset=list()
    infolist=list()
    for index in tqdm(range(len(allinfolist))):
        # if index<1737:
        #     continue
        # print(alllosslist[index])
        # print(allinfolist[index])
        # print(len(allinfolist[index])==len(allinfolist[index]))
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

def get_divide_indexes(datasetname):
    allinfolist=get_info_list(datasetname)
    alllosslist=get_loss_list(allinfolist,datasetname)
    rawdata=processjson(datasetname)
    print(len(rawdata)==len(allinfolist))
    output = list()
    for index in range(len(allinfolist)):
        divide_list = autodivide(alllosslist[index],allinfolist[index])
        divide_list.append(len(allinfolist[index]))
        # print(divide_list)
        output.append(divide_list)
    return output    

def debug():
    loss = [2.3515799045562744, 3.587576150894165, 4.129663467407227, 2.5744407176971436, 3.3050320148468018, 3.2095212936401367, 2.472079277038574, 3.206564426422119, 2.4536759853363037, 2.2818377017974854, 1.181156873703003, 1.885697841644287, 2.8144125938415527, 3.4177727699279785, 1.445846438407898, 2.608379602432251, 3.353236198425293, 3.313612461090088, 2.6420867443084717, 2.906754732131958, 2.293593645095825, 2.8276402950286865, 3.8969314098358154, 2.2918701171875, 2.5220327377319336, 2.960987091064453, 3.567401885986328, 3.0763049125671387, 2.8942711353302, 2.3835301399230957, 2.5542564392089844, 2.433392286300659, 2.131594657897949, 2.6842262744903564, 3.355440616607666, 3.1972484588623047, 3.2258341312408447, 2.681248188018799, 2.973982572555542, 2.9319686889648438, 2.137864351272583, 2.3867626190185547, 2.456348419189453, 2.133725643157959, 3.009502410888672, 1.7674505710601807, 2.922492504119873, 2.1369593143463135, 3.311776638031006, 3.484457015991211, 2.25905179977417, 2.803236484527588, 3.022736072540283, 2.787600517272949, 2.18241024017334, 2.5295701026916504, 2.23738169670105, 2.780573606491089, 3.227720022201538, 3.196718454360962, 2.909918785095215, 2.839235782623291, 2.5181620121002197, 1.6088496446609497, 2.0502631664276123, 2.1745829582214355, 2.4497411251068115, 1.2779217958450317, 2.3921327590942383, 3.3534083366394043, 2.9713499546051025, 3.895900249481201, 2.4943315982818604, 3.0470149517059326, 2.790579319000244, 2.554238796234131, 2.305668354034424, 2.7463905811309814, 3.211763381958008, 3.1337223052978516, 2.807469129562378, 3.3346269130706787, 4.153022289276123, 0.8033920526504517, 2.1065633296966553, 3.8692786693573, 3.394951105117798, 2.3663341999053955, 2.4416866302490234, 3.295121431350708, 3.2816708087921143, 2.6126041412353516, 2.8883628845214844, 2.959900379180908, 2.7069804668426514, 3.1603963375091553, 2.4479806423187256, 4.29784631729126, 2.5423333644866943, 3.009129762649536, 2.5051329135894775, 2.3627219200134277, 2.1038734912872314]
    s = [37, 39, 40, 24, 575, 68, 60, 84, 34, 60, 41, 75, 64, 49, 94, 187, 45, 34, 79, 78, 39, 87, 46, 116, 94, 50, 23, 53, 49, 33, 141, 74, 25, 33, 53, 49, 671, 48, 517, 384, 75, 45, 37, 54, 35, 33, 36, 55, 35, 58, 33, 159, 45, 53, 75, 23, 24, 33, 54, 36, 384, 52, 77, 47, 53, 99, 55, 41, 37, 74, 32, 45, 116, 45, 53, 48, 80, 33, 54, 36, 463, 128, 41, 45, 28, 91, 18, 57, 23, 57, 23, 108, 41, 19, 43, 83, 46, 15, 21, 34, 21, 34, 22]
    print(autodivide(loss,s))

def demo():
    datasetname='ami_test'
    allinfolist=get_info_list(datasetname)
    alllosslist=get_loss_list(allinfolist,datasetname)
    print('***finish preprocessing info and loss***')
    index=9
    divide_list= autodivide(alllosslist[index],allinfolist[index])
    plot_loss(alllosslist[index],divide_list)   

if __name__=='__main__':
    main()