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
from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

event_mention_typ={'m','TIME','trigger','WithIn','company','ORG','mEntity','triggerRoot'}

import sonnet as snt
import tensorflow as tf



##字典
token=dict(zip(list('这是一个测试'),list(range(len('这是一个测试')))))
token['triggerRoot']=100
yll=['u','u','u','u','u','u']
tokeny={'u':3}


ll='这是一个测试'


def make_x_y(ll,yll,token,tokeny):#xll yll  token=x_str2id  tokeny=y_str2id
    g = nx.Graph()
    gy=nx.Graph()
    n=len(ll)
    #加节点 特征
    for ni,char in enumerate(ll):
        g.add_node(ni,features=[token[char]],char=char)
        y=yll[ni]
        if y=='unlabel':
            y='PAD'
        gy.add_node(ni,features=[tokeny[y]],char=y)

    #加边
    for ii in range(n)[:-1]:
        g.add_edge(ii,ii+1,features=[0])
        g.add_edge(ii+1, ii,features=[0])
        #
        gy.add_edge(ii,ii+1,features=[0])
        gy.add_edge(ii+1, ii, features=[0])

    # global
    g.graph['features']=0
    gy.graph['features']=0

    ### 增加 4 个triggerRoot node
    g.add_node(n,features=[token['triggerRoot']],char='triggerRoot')
    g.add_node(n+1,features=[token['triggerRoot']],char='triggerRoot')
    g.add_node(n + 2, features=[token['triggerRoot']], char='triggerRoot')
    g.add_node(n + 3, features=[token['triggerRoot']], char='triggerRoot')
    #### ygraph 增加4个 pad
    gy.add_node(n,features=[0],char='PAD')
    gy.add_node(n+1,features=[0],char='PAD')
    gy.add_node(n + 2, features=[0], char='PAD')
    gy.add_node(n + 3, features=[0], char='PAD')

    for nod in g.nodes(data=True):
        nid,feadic=nod
        if nid in [n,n+1,n+2,n+3]:continue # 是triggerRoot节点
        ###
        if feadic['char'] not in event_mention_typ:continue # 额外的triggerRoot 只和event mention链接
        g.add_edge(nid,n,features=[0])
        g.add_edge(n, nid, features=[0])
        #
        g.add_edge(n+1, nid, features=[0])
        g.add_edge(nid,n+1, features=[0])
        g.add_edge(n + 2, nid, features=[0])
        g.add_edge(nid, n + 2, features=[0])
        g.add_edge(n + 3, nid, features=[0])
        g.add_edge(nid, n + 3, features=[0])
        #print ('')





    gtx = utils_np.networkxs_to_graphs_tuple([g])
    gty=utils_np.networkxs_to_graphs_tuple([gy])
    #print ('')

    return g,gy,gtx,gty

def gt_array2tensor(gtx):
    gtx = gtx.replace(nodes=tf.constant(gtx.nodes),
                      edges=tf.constant(gtx.nodes),
                      globals=tf.constant(gtx.nodes),
                      n_edge=tf.constant(gtx.nodes),
                      n_node=tf.constant(gtx.nodes),
                      receivers=tf.constant(gtx.nodes),
                      senders=tf.constant(gtx.nodes))
    return gtx

#####
if __name__=='__main__':
    g,gy,gtx,gty=make_x_y(ll,yll,token,tokeny)
    print ('')
    gtx=gtx.replace(nodes=tf.constant(gtx.nodes),
                    edges=tf.constant(gtx.nodes))
    # gtx = gtx.replace(edges=tf.constant(gtx.nodes))
    # gtx = gtx.replace(globals=tf.constant(gtx.nodes))
    # gtx = gtx.replace(n_edge=tf.constant(gtx.nodes))
    # gtx = gtx.replace(n_node=tf.constant(gtx.nodes))
    # gtx = gtx.replace(receivers=tf.constant(gtx.nodes))
    # gtx = gtx.replace(senders=tf.constant(gtx.nodes))
    print ('')


