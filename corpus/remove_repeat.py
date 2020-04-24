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


#### 去掉测试集合中重复的

import json
import re
import sys
import os
import copy
import numpy as np

lenll=[]

from problem_util_yr.loadDict.read_json_tool import read_json

p_num=re.compile('\d+')


event_mention_typ={'m','TIME','trigger','WithIn','company','ORG'}
## trigger time 没有other



testset=set()
trainset=set()


f='test.json'
gene=read_json(f)
for ll in gene:
    line=''.join([w['char'] for w in ll])
    testset.add(line)


f='train.json'
gene=read_json(f)
for ll in gene:
    line=''.join([w['char'] for w in ll])
    trainset.add(line)


realtestset=testset-trainset
print (len(realtestset))
writer=open('realtestset.json','w')

##############
#
f='test.json'
gene=read_json(f)
for ll in gene:
    line=''.join([w['char'] for w in ll])
    if line not in realtestset:continue
    ####
    writer.write(json.dumps(ll,ensure_ascii=False)+'\n')