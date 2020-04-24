####################
# 给定 trigger 预测 这个trigger的属性 subject time value ratio..
##  原始句子 company time trigger1 m m m, org time trigger2 m m m
## 想分别预测 trigger1 的相关 attribute:company time m m m  ,
# trigger2和相关的属性 都作为unlabel
## 输入  company time triggerRoot   m     m     m,     org   time    trigger   m       m       m
## 输出  subject time  trigger   value  value value  unlabel unlabel unlabel unlabel unlabel unlabel
# 原始数据1条 ：2018年实现总资产81万元，总负债45万元
# 第一个样本要预测的关系[trigger:总资产 argument1:81万元 关系:value ; arg2:time 关系:time]
# TIME , 实 现 triggerRoot mEntity , trigger mEntity 。
# 第二个样本要预测的关系 [trigger:总负债 argument 45万元 关系value;arg2:time 关系:time]
# TIME , 实 现 trigger mEntity , triggerRoot mEntity 。

import json
import re
import sys
import os
import copy
import numpy as np

lenll=[]



p_num=re.compile('\d+')


event_mention_typ={'m','TIME','trigger','WithIn','company','ORG'}
## trigger time 没有other

def remove_num(w):# ratio3->ratio
    return p_num.sub('',w)
def replace_num(line):
    line=p_num.sub('m',line)
    return line

def number_the_triggers(wordll):
    cnt=0
    ret=[]
    for idx,w in enumerate(wordll):
        if w=='trigger':
            w=w+str(cnt)
            cnt+=1
            ret.append(w)
        ###
        else:ret.append(w)
    return ret




def run(reader,writer):
    global vocaby,vocabx,lenll
    for line in reader.readlines():
        #### 每个句子
        line = line.strip()
        lab, d = line.split('\t')
        d = json.loads(d)
        #print('')
        ### get rawsent
        wordll = [w['typ'] if w['typ'] in event_mention_typ else w['val'] for w in d['words']]
        wordll=['mEntity' if w=='m' else w for w in wordll]
        ### trigger ->trigger1 trigger2
        #wordll = number_the_triggers(wordll)
        whether_event_mention = [True if w['typ'] in event_mention_typ else False for w in d['words']]
        ### event
        for event in d['events']:
            ## 每个事件  一个ROOT  多个attribute
            this_event_y = ['unlabel'] * len(wordll)
            if 'trigger' not in event:continue
            rootPos = event['trigger']
            this_event_y[rootPos] = 'trigger'

            for rela, attPos in event.items():
                if rela=='trigger':continue

                this_event_y[attPos] = rela
            ###
            #print('')
            ### combine x y -> [{x:x,y:y},{},,,]
            xydictll = [{'w': w, 'y': y, 'eventflag': flag} for w, y, flag in
                        zip(wordll, this_event_y, whether_event_mention)]
            #### 一个句子多个TRIGGER 时候 把其中ROOT 的trigger ->trigger_root
            xydictll[rootPos]['w']='triggerRoot'
            #print('')
            #############
            # word -> char
            charll = []
            for dic in xydictll:
                if dic['eventflag'] == False:  # 只把不是EVENT MENTION的单词拆成字
                    for char in dic['w']:
                        charll.append({'char': char, 'y': 'unlabel'})
                else:
                    charll.append({'char': dic['w'], 'y': dic['y']})
            #### ###
             # p(attribute|trigger-root)
            ##### ratio4->ratio
            charll1=copy.deepcopy(charll)
            for wdic in charll:
                wdic['y']=remove_num(wdic['y'])

            writer.write(json.dumps(charll, ensure_ascii=False) + '\n')
            ######字典
            for char in charll:
                vocabx.add(char['char'])
                vocaby.add(char['y'])
            ###### length
            lenll.append(len(charll))







f='./trigger.train'
writer=open('train.json','w')

vocabx,vocaby=set(),set()


reader=open(f)
run(reader,writer)

#########
# test
f = './trigger.dev'
writer = open('test.json', 'w')


reader = open(f)
run(reader, writer)


######## xvocab
writerv=open('vocabx.txt','w')
writerv.write('PAD\n')
writerv.write('EOS\n')
writerv.write('UNK\n')
for w in vocabx:
    writerv.write(w+'\n')

writerv=open('vocaby.txt','w')
writerv.write('PAD\n')
writerv.write('EOS\n')
writerv.write('UNK\n')
for w in vocaby:
    writerv.write(w+'\n')





print (np.histogram(lenll))

# (array([29374,  8635,  2215,   887,   584,   140,    65,   140,    20,10]),
#  array([  6. ,  36.7,  67.4,  98.1, 128.8, 159.5, 190.2, 220.9, 251.6, 282.3, 313. ]))



