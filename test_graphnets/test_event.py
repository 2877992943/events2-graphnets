from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time


from graph_nets import utils_np
from graph_nets import utils_tf
#from graph_nets.demos_tf2 import models
import models_v04

import numpy as np

import sonnet as snt
import tensorflow as tf
import pandas as pdd
import os
import random

SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)

def create_graph_dicts_tf(num_examples, num_elements_min_max):
  """Generate graphs for training.

  Args:
    num_examples: total number of graphs to generate
    num_elements_min_max: a 2-tuple with the minimum and maximum number of
      values allowable in a graph. The number of values for a graph is
      uniformly sampled withing this range. The upper bound is exclusive, and
      should be at least 2 more than the lower bound.

  Returns:
    inputs: contains the generated random numbers as node values.
    sort_indices: contains the sorting indices as nodes. Concretely
      inputs.nodes[sort_indices.nodes] will be a sorted array.
    ranks: the rank of each value in inputs normalized to the range [0, 1].
  """
  num_elements = tf.random.uniform(
      [num_examples],
      minval=num_elements_min_max[0],
      maxval=num_elements_min_max[1],
      dtype=tf.int32)
  inputs_graphs = []
  sort_indices_graphs = []
  ranks_graphs = []
  for i in range(num_examples):
    values = tf.random.uniform(shape=[num_elements[i]])
    sort_indices = tf.cast(
        tf.argsort(values, axis=-1), tf.float32)
    ranks = tf.cast(
        tf.argsort(sort_indices, axis=-1), tf.float32) / (
            tf.cast(num_elements[i], tf.float32) - 1.0)
    inputs_graphs.append({"nodes": values[:, None]})
    sort_indices_graphs.append({"nodes": sort_indices[:, None]})
    ranks_graphs.append({"nodes": ranks[:, None]})
  return inputs_graphs, sort_indices_graphs, ranks_graphs


def create_linked_list_target(batch_size, input_graphs):
  """Creates linked list targets.

  Returns a graph with the same number of nodes as `input_graph`. Each node
  contains a 2d vector with targets for a 1-class classification where only one
  node is `True`, the smallest value in the array. The vector contains two
  values: [prob_true, prob_false].
  It also contains edges connecting all nodes. These are again 2d vectors with
  softmax targets [prob_true, prob_false]. An edge is True
  if n+1 is the element immediately after n in the sorted list.

  Args:
    batch_size: batch size for the `input_graphs`.
    input_graphs: a `graphs.GraphsTuple` which contains a batch of inputs.

  Returns:
    A `graphs.GraphsTuple` with the targets, which encode the linked list.
  """
  target_graphs = []
  for i in range(batch_size):
    input_graph = utils_tf.get_graph(input_graphs, i)
    num_elements = tf.shape(input_graph.nodes)[0]
    si = tf.cast(tf.squeeze(input_graph.nodes), tf.int32)
    nodes = tf.reshape(tf.one_hot(si[:1], num_elements), (-1, 1))
    x = tf.stack((si[:-1], si[1:]))[None]
    y = tf.stack(
        (input_graph.senders, input_graph.receivers), axis=1)[:, :, None]
    edges = tf.reshape(
        tf.cast(
            tf.reduce_any(tf.reduce_all(tf.equal(x, y), axis=1), axis=1),
            tf.float32), (-1, 1))
    target_graphs.append(input_graph._replace(nodes=nodes, edges=edges))
  return utils_tf.concat(target_graphs, axis=0)


def compute_accuracy(target, output):
  """Calculate model accuracy.

  Returns the number of correctly predicted links and the number
  of completely solved list sorts (100% correct predictions).

  Args:
    target: A `graphs.GraphsTuple` that contains the target graph.
    output: A `graphs.GraphsTuple` that contains the output graph.

  Returns:
    correct: A `float` fraction of correctly labeled nodes/edges.
    solved: A `float` fraction of graphs that are completely correctly labeled.
  """
  tdds = utils_np.graphs_tuple_to_data_dicts(target)
  odds = utils_np.graphs_tuple_to_data_dicts(output)
  cs = []
  ss = []
  for td, od in zip(tdds, odds):
    num_elements = td["nodes"].shape[0]
    xn = np.argmax(td["nodes"], axis=-1)
    yn = np.argmax(od["nodes"], axis=-1)

    xe = np.reshape(
        np.argmax(
            np.reshape(td["edges"], (num_elements, num_elements, 2)), axis=-1),
        (-1,))
    ye = np.reshape(
        np.argmax(
            np.reshape(od["edges"], (num_elements, num_elements, 2)), axis=-1),
        (-1,))
    c = np.concatenate((xn == yn, xe == ye), axis=0)
    s = np.all(c)
    cs.append(c)
    ss.append(s)
  correct = np.mean(np.concatenate(cs, axis=0))
  solved = np.mean(np.stack(ss))
  return correct, solved


def create_data(batch_size, num_elements_min_max):
  """Returns graphs containing the inputs and targets for classification.

  Refer to create_data_dicts_tf and create_linked_list_target for more details.

  Args:
    batch_size: batch size for the `input_graphs`.
    num_elements_min_max: a 2-`tuple` of `int`s which define the [lower, upper)
      range of the number of elements per list.

  Returns:
    inputs: a `graphs.GraphsTuple` which contains the input list as a graph.
    targets: a `graphs.GraphsTuple` which contains the target as a graph.
    sort_indices: a `graphs.GraphsTuple` which contains the sort indices of
      the list elements a graph.
    ranks: a `graphs.GraphsTuple` which contains the ranks of the list
      elements as a graph.
  """
  inputs, sort_indices, ranks = create_graph_dicts_tf(
      batch_size, num_elements_min_max)
  inputs = utils_tf.data_dicts_to_graphs_tuple(inputs)
  sort_indices = utils_tf.data_dicts_to_graphs_tuple(sort_indices)
  ranks = utils_tf.data_dicts_to_graphs_tuple(ranks)

  inputs = utils_tf.fully_connect_graph_dynamic(inputs)
  sort_indices = utils_tf.fully_connect_graph_dynamic(sort_indices)
  ranks = utils_tf.fully_connect_graph_dynamic(ranks)

  targets = create_linked_list_target(batch_size, sort_indices)
  nodes = tf.concat((targets.nodes, 1.0 - targets.nodes), axis=1)
  edges = tf.concat((targets.edges, 1.0 - targets.edges), axis=1)
  targets = targets._replace(nodes=nodes, edges=edges)

  return inputs, targets, sort_indices, ranks # input node[7,1] edge  target edge[49,2] node[7,2]


def get_graphtuple(gll): # gll =graphlist
    return utils_np.networkxs_to_graphs_tuple(gll)

def gt_array2tensor(gtx):
    gtx = gtx.replace(nodes=tf.constant(gtx.nodes),
                      edges=tf.constant(gtx.edges),
                      globals=tf.constant(gtx.globals),
                      n_edge=tf.constant(gtx.n_edge),
                      n_node=tf.constant(gtx.n_node),
                      receivers=tf.constant(gtx.receivers),
                      senders=tf.constant(gtx.senders))
    return gtx

fpk='../corpus/pkls'
def create_data_v04():
    for _ in range(epoch):
        for fn in os.listdir(fpk):
            fn=os.path.join(fpk,fn)
            ll=pdd.read_pickle(fn)
            random.shuffle(ll)
            bsz=10
            for batchi in list(range(len(ll)))[::bsz]:
                batch=ll[batchi:batchi+bsz]
                # list [{'x':g,'y':g},,,]
                gxll=[p['x'] for p in batch]
                gyll=[p['y'] for p in batch]
                gtx,gty=get_graphtuple(gxll),get_graphtuple(gyll)  # graphtuple
                gtx=gt_array2tensor(gtx)
                gty=gt_array2tensor(gty)
                yield gtx,gty


def create_loss_v04(target,outputs):
    eps=0.0000001
    node_y=target.nodes #[n,1]
    node_y=node_y[:,0]
    m1=node_y > 0
    nonzero=tf.dtypes.cast(m1, tf.float32)#[n,1]
    nonzero_num=tf.reduce_sum(nonzero)#非零个数
    nodey_onehot=tf.one_hot(node_y,depth=ynode_vocab) #[n,vocabsz]
    lossll=[]
    for pred in outputs:
        node_pred=pred.nodes #[n,vocabsz]
        node_pred=tf.nn.log_softmax(node_pred,axis=1) #-> probability -> log(prob)
        node_pred=node_pred*nodey_onehot
        node_pred=tf.reduce_sum(node_pred,axis=-1) #->[n,] get maxProbability
        loss=-tf.reduce_sum(node_pred*nonzero)/(eps+nonzero_num)
        lossll.append(loss)
        #tf.compat.v1.losses.softmax_cross_entropy(nodey_onehot, node_pred)
    return lossll
    #print ('')
    # losss = [
    #     tf.compat.v1.losses.softmax_cross_entropy(target.nodes, output.nodes)# +
    #     #tf.compat.v1.losses.softmax_cross_entropy(target.edges, output.edges)
    #     for output in outputs
    # ]
    #return tf.stack(losss) # target [0,0,1] output=logits not probability

def create_loss_v04_1(target,outputs):
    eps=0.0000001
    node_y=target.nodes #[n,1]
    node_y=node_y[:,0]
    #m1=node_y > 0
    #nonzero=tf.dtypes.cast(m1, tf.float32)#[n,1]
    #nonzero_num=tf.reduce_sum(nonzero)#非零个数
    nodey_onehot=tf.one_hot(node_y,depth=ynode_vocab) #[n,vocabsz]
    lossll=[]
    for pred in outputs:
        node_pred=pred.nodes #[n,vocabsz]
        node_pred=tf.nn.log_softmax(node_pred,axis=1) #-> probability -> log(prob)
        node_pred=node_pred*nodey_onehot
        node_pred=tf.reduce_sum(node_pred,axis=-1) #->[n,] get maxProbability
        #loss=-tf.reduce_sum(node_pred*nonzero)/(eps+nonzero_num)
        loss = -tf.reduce_mean(node_pred) # sum
        lossll.append(loss)
        #tf.compat.v1.losses.softmax_cross_entropy(nodey_onehot, node_pred)
    return lossll



def create_acc(target,outputs):
    ### 不算PAD ID=0的值
    pred=outputs[-1]
    node_y = target.nodes[:,0]
    pred=tf.argmax(pred.nodes,axis=-1) #[n,]
    cnt_y,correct_cnt,cnt_p=0,0,0#非零的 数量
    ###

    ###
    for ni,y in enumerate(node_y):
        ##
        predi=pred[ni]
        #
        if predi==y and predi.numpy()!=0:correct_cnt+=1 # pred=y并且不等于 0 PAD
        #
        if y.numpy()!=0:cnt_y+=1# y有几个非零
        if predi.numpy()!=0:cnt_p+=1 #pred 有几个非零
        # PAD 不计算在准确里
    return correct_cnt/(0.00001+cnt_p),correct_cnt/(0.00001+cnt_y)



def create_loss(target, outputs):
  """Returns graphs containing the inputs and targets for classification.

  Refer to create_data_dicts_tf and create_linked_list_target for more details.

  Args:
    target: a `graphs.GraphsTuple` which contains the target as a graph.
    outputs: a `list` of `graphs.GraphsTuple`s which contains the model
      outputs for each processing step as graphs.

  Returns:
    A `list` of ops which are the loss for each processing step.
  """
  # if not isinstance(outputs, collections.Sequence):
  #   outputs = [outputs]
  losss = [
      tf.compat.v1.losses.softmax_cross_entropy(target.nodes, output.nodes) +
      tf.compat.v1.losses.softmax_cross_entropy(target.edges, output.edges)
      for output in outputs
  ]
  return tf.stack(losss)



def update_step(inputs_tr, targets_tr):
  with tf.GradientTape() as tape:
    outputs_tr = model(inputs_tr, num_processing_steps_tr)
    # Loss.
    loss_tr = create_loss_v04_1(targets_tr, outputs_tr)
    loss_tr = tf.math.reduce_sum(loss_tr) / num_processing_steps_tr

  gradients = tape.gradient(loss_tr, model.trainable_variables)
  optimizer.apply(gradients, model.trainable_variables)
  return outputs_tr, loss_tr

# def plot_linked_list(ax, graph, sort_indices):
#   """Plot a networkx graph containing weights for the linked list probability."""
#   nd = len(graph.nodes())
#   probs = np.zeros((nd, nd))
#   for edge in graph.edges(data=True):
#     probs[edge[0], edge[1]] = edge[2]["features"][0]
#   ax.matshow(probs[sort_indices][:, sort_indices], cmap="viridis")
#   ax.grid(False)


epoch=30
gene=create_data_v04() # gtx gty
num_processing_steps_tr =  10

# Optimizer.
learning_rate = 1e-3
optimizer = snt.optimizers.Adam(learning_rate)
ynode_vocab=16
model = models_v04.EncodeProcessDecode(#edge_output_size=2,
                                       node_output_size=ynode_vocab)
#############
# test debug
for gtx,gty in gene:
    break
outputs_tr = model(gtx, num_processing_steps_tr)
# Loss.
loss_tr = create_loss_v04_1(gty, outputs_tr)
# acc
acc,recall=create_acc(gty,outputs_tr)
print ('')

#################
# train
step=0
for gtx,gty in gene:
    out,loss=update_step(gtx,gty)
    step+=1
    if step%10==0:

        acc,recall=create_acc(gty,out)
        print(' step %d loss %f acc %f recall %f'%(step, loss,acc,recall))















