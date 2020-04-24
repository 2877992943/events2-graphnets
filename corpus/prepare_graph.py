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


### node edge
# node fea edge fea
# global


import json
import re
import sys
import os
import copy
import numpy as np
import networkx as nx
import pandas as pdd

trainfile='train.json'
testfile='realtestset.json'


from problem_util_yr.loadDict.read_json_tool import read_json
from test_networkx import make_x_y

p_num=re.compile('\d+')


event_mention_typ={'m','TIME','trigger','WithIn','company','ORG'}
## trigger time 没有other




def get_str2id(fpath):
    reader=open(fpath)
    str2id={}
    for line in reader.readlines():
        line=line.strip()
        if line and line not in str2id:
            str2id[line]=len(str2id)
    return str2id



tokenx=get_str2id('vocabx.txt')
tokeny=get_str2id('vocaby.txt')
testset=set()
trainset=set()


gene=read_json(testfile)
cnt=0
tmp=[]
for ll in gene:
    xll=[w['char'] for w in ll]
    yll=[w['y'] for w in ll]
    if len(xll)==0:continue
    gx,gy,_,_=make_x_y(xll,yll,tokenx,tokeny)
    #print ('')
    cnt+=1
    if cnt%100==0:print (cnt)
    ###
    tmp.append({'x':gx,'y':gy})
    if len(tmp)>=100:
        pdd.to_pickle(tmp,'./pkls/%d.pkl'%cnt)
        tmp=[]


