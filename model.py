import os
import sys
import argparse
import subprocess
import pdb
import time
import random
import _pickle as cPickle
import glob
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import math
import networkx as nx
import numpy as np
from utils import cal_knn
from itertools import count
from heapq import heappush, heappop
#from hntm import HierarchicalNeuralTopicModel
from tree import get_descendant_idxs

from evaluate import compute_topic_specialization
from configure import get_parser
from data_structure import Instance, get_batches
# from tree import get_tree_idxs

MAX_CHILD_NODES=20
def get_tree_idxs(tree):
    tree_idxs = {}
    tree_idxs[0] = [i for i in range(1, tree//MAX_CHILD_NODES +1)]
    for parent_idx in tree_idxs[0]:
        tree_idxs[parent_idx] = [parent_idx*MAX_CHILD_NODES+i for i in range(1, tree % MAX_CHILD_NODES +1)]
    return tree_idxs


def update_checkpoint(config, checkpoint, global_step):
    checkpoint.append(config.path_model + '-%i' % global_step)
    if len(checkpoint) > config.max_to_keep:
        path_model = checkpoint.pop(0) + '.*'
        for p in glob.glob(path_model):
            os.remove(p)
    cPickle.dump(checkpoint, open(config.path_checkpoint, 'wb'))
    


def print_topic_sample(sess, model, topic_prob_topic=None, recur_prob_topic=None, topic_freq_tokens=None, parent_idx=0, depth=0):
    if depth == 0: # print root
        assert len(topic_prob_topic) == len(recur_prob_topic) == len(topic_freq_tokens)
        freq_tokens = topic_freq_tokens[parent_idx]
        recur_topic = recur_prob_topic[parent_idx]
        prob_topic = topic_prob_topic[parent_idx]
        print(parent_idx, 'R: %.3f' % recur_topic, 'P: %.3f' % prob_topic, ' '.join(freq_tokens))
    
    child_idxs = model.tree_idxs[parent_idx]
    depth += 1
    
    for child_idx in child_idxs:
        freq_tokens = topic_freq_tokens[child_idx]
        recur_topic = recur_prob_topic[child_idx]
        prob_topic = topic_prob_topic[child_idx]
        print('  '*depth, child_idx, 'R: %.3f' % recur_topic, 'P: %.3f' % prob_topic, ' '.join(freq_tokens))
        
        if child_idx in model.tree_idxs: 
            print_topic_sample(sess, model, topic_prob_topic=topic_prob_topic, recur_prob_topic=recur_prob_topic, topic_freq_tokens=topic_freq_tokens, parent_idx=child_idx, depth=depth)
    print()    

from collections import defaultdict

import copy
import numpy as np
import tensorflow as tf

from nn import rnn, tsbp, sbp

from tree import get_topic_idxs, get_child_to_parent_idxs, get_depth, get_ancestor_idxs, get_descendant_idxs

class DoublyRNNCell:
    def __init__(self, dim_hidden, dim_latent_path, keep_prob, 
                 output_layer=None, vis=False):
        self.vis=vis
        self.keep_prob = keep_prob
        self.dim_hidden = dim_hidden
        self.dim_latent_path = dim_latent_path

        self.ancestral_layer=tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='ancestral')
        self.fraternal_layer=tf.layers.Dense(units=dim_hidden, activation=tf.nn.tanh, name='fraternal')
        self.hidden_layer = tf.layers.Dense(units=dim_hidden, name='hidden')
        
        self.coord_layer=tf.layers.Dense(units=dim_hidden, name='coord')

        
        self.output_layer=output_layer
        
    def __call__(self, state_ancestral, state_fraternal, topic_idx, topic_coords,reuse=True):
        
        with tf.variable_scope('input', reuse=reuse):
            state_ancestral = self.ancestral_layer(state_ancestral)
            state_fraternal = self.fraternal_layer(state_fraternal)

        
        with tf.variable_scope('output', reuse=reuse):
            coord = None
            state_hidden = self.hidden_layer(state_ancestral + state_fraternal)
            if self.output_layer is not None: 
#                 output = self.output_layer(state_hidden)+self.coord_layer(tf.nn.tanh(topic_coords[topic_idx]))
                output = self.output_layer(state_hidden)
#                 output = self.output_layer(tf.concat([state_hidden, topic_coords[topic_idx]], axis=-1))
            else:
                output = state_hidden
                
#             coord = self.coord_layer_bn(self.coord_layer(self.coord_layer_h1(output)))
        return output, state_hidden, coord
    
    def get_initial_state(self, name):
        initial_state = tf.get_variable(name, [1, self.dim_hidden], dtype=tf.float32)
        return initial_state
    
    def get_zero_state(self, name):
        zero_state = tf.zeros([1, self.dim_hidden], dtype=tf.float32, name=name)
        return zero_state    


def doubly_rnn(dim_hidden, dim_latent_path, keep_prob, tree_idxs, topic_coords, initial_state_parent=None, initial_state_sibling=None, output_layer=None, name='', vis=False):
    outputs, states_parent = {}, {}
    coords = {}
    with tf.variable_scope(name, reuse=False):
        doubly_rnn_cell = DoublyRNNCell(dim_hidden, dim_latent_path, keep_prob, output_layer,vis)

        if initial_state_parent is None: 
            initial_state_parent = doubly_rnn_cell.get_initial_state('init_state_parent')
        if initial_state_sibling is None: 
            initial_state_sibling = doubly_rnn_cell.get_zero_state('init_state_sibling')
        output, state_sibling, coord = doubly_rnn_cell(initial_state_parent, initial_state_sibling, 0, topic_coords, reuse=False)
        outputs[0], states_parent[0], coords[0] = output, state_sibling, coord

        for parent_idx, child_idxs in tree_idxs.items():
            state_parent = states_parent[parent_idx]
            state_sibling = initial_state_sibling
            for child_idx in child_idxs:
                output, state_sibling, coord = doubly_rnn_cell(state_parent, state_sibling, child_idx, topic_coords)
                outputs[child_idx], states_parent[child_idx], coords[child_idx] = output, state_sibling, coord

    return outputs, states_parent, coords

def tf_log(x):
    return tf.log(x+1e-10)
    # return tf.log(tf.clip_by_value(x, 1e-5, x))

def sample_latents(means, logvars):
    # reparameterize
    noises = tf.random.normal(tf.shape(means))#*0.01
    latents = means + tf.exp(logvars)**0.5 * noises
    return latents

def compute_kl_loss(means, logvars, means_prior=None, logvars_prior=None):
    if means_prior is None and logvars_prior is None:
        kl_losses = tf.reduce_sum(-0.5 * (logvars - tf.square(means) - tf.exp(logvars) + 1.0), 1) # sum over latent dimentsion    
    elif means_prior is not None:
        kl_losses = tf.reduce_sum(-0.5 * (logvars - tf.square(means-means_prior) - tf.exp(logvars) + 1.0), 1) # sum over latent dimentsion    
    return kl_losses

def compute_kl_losses(means, logvars, means_prior=None, logvars_prior=None):
    if means_prior is None and logvars_prior is None:
        kl_losses = tf.reduce_sum(-0.5 * (logvars - tf.square(means) - tf.exp(logvars) + 1.0), -1) # sum over latent dimentsion    
    elif means_prior is not None and logvars_prior is None:
        kl_losses = tf.reduce_sum(-0.5 * (logvars - tf.square(means-means_prior) - tf.exp(logvars) + 1.0), -1) # sum over latent dimentsion 
    else:
        #kl_losses= 0.5 * tf.reduce_sum(tf.exp(logvars-logvars_prior) + tf.square(means_prior - means) / tf.clip_by_value(tf.exp(logvars_prior), 1e-5, tf.exp(logvars_prior)) - 1 + (logvars_prior - logvars), -1) # sum over latent dimentsion   
        kl_losses= 0.5 * tf.reduce_sum(tf.exp(logvars-logvars_prior) + tf.square(means_prior - means) / (tf.exp(logvars_prior)+1e-10) - 1 + (logvars_prior - logvars), -1) # sum over latent dimentsion    
    return kl_losses

def softmax_with_temperature(logits, axis=None, name=None, temperature=1.):
    if axis is None: axis = -1
    #torch.exp(distances - torch.logsumexp(distances, dim=-1, keepdim=True))
    return tf.exp(logits / temperature) / tf.reduce_sum(tf.exp(logits / temperature), axis=axis)#tf.exp( (logits / temperature) - tf.math.reduce_logsumexp((logits / temperature), axis=axis))#tf.exp(logits / temperature) / tf.reduce_sum(tf.exp(logits / temperature), axis=axis)

def get_level_nodes(tree_depth, max_lvl):
    level_nodes = defaultdict(list)
    for level in range(max_lvl):
        for key, value in tree_depth.items():
            if value == level+1:
                level_nodes[level].append(key)
    level_nodes = dict(level_nodes)
    return level_nodes
def get_leaf_parents(tree_idxs, level_nodes, n_depth, topic_idxs, child_to_parent_idxs):
#     leaf_parents = {x:[0] for x in level_nodes[n_depth-1]}

#     for node in level_nodes[1]:
#         for leaf, parent in leaf_parents.items():
#           if leaf in tree_idxs[node]:
#             parent.append(node)
    print('child_to_parent_idxs get_leaf_parents', child_to_parent_idxs)
    print('tree_idxs get_leaf_parents', tree_idxs)
    leaf_parents = {x:[] for x in topic_idxs if x not in tree_idxs.keys()}
    for leaf, parents in leaf_parents.items():
        x = leaf
        while(x!=0):
            x = child_to_parent_idxs[x]
            parents.append(x)
        parents.sort()

    return  leaf_parents


def compute_pi_leaf2root(topic_coords, d, leaf_parents, node_level, tree_idxs, n_depth, batch_size, dist_type='gauss'):
    N = batch_size
    leaf_prob = {}
    gamma_topic = {}
    
#     if dist_type=='gauss':
#         distance_d_root = 1e-20+tf.exp(tf.reduce_sum(-0.5*tf.pow(d - topic_coords[0], 2),-1))
#     if dist_type=='inv':
#         distance_d_root = 1./(1.+tf.reduce_sum(tf.pow(d - topic_coords[0], 2),-1))
    
#     distance_d_root = 1e-20+tf.exp(tf.reduce_sum(-0.5*tf.pow(d - topic_coords[0], 2),-1))
    gamma_topic[0] =  tf.ones([N,1])#tf.expand_dims(distance_d_root/ distance_d_root, axis=-1)#tf.ones([N,1])
    
    sum_distances_path = {}
    
    
    d = tf.expand_dims(d, 1)
    distance_c_d = {}
    for parent, childs in tree_idxs.items():
        sum_childs = {}
        sum_childs[parent] = 0
        for child in childs:
            level = node_level.get(child)
            
            # coord = tf.concat([topic_coords.get(child), tf.expand_dims((level) * topic_coords_z[:,0], 1)], axis=-1) #1x3
            coord = topic_coords.get(child)
            #d = tf.exp(tf.reduce_sum(-0.5*tf.pow(d - coord, 2),-1)) 
            
            #distance_temp = 1./(1.+tf.reduce_sum(tf.pow(d - coord, 2),-1)) #1./(1.+tf.reduce_sum(tf.pow(d - coord, 2),-1))#tf.exp(tf.reduce_sum(-0.5*tf.pow(d - coord, 2),-1))
            if dist_type=='gauss':
                distance_temp = tf.exp(tf.reduce_sum(-0.5*tf.pow(d - coord, 2),-1))
            if dist_type=='inv':
                distance_temp = 1./(1.+tf.reduce_sum(tf.pow(d - coord, 2),-1))
            distance_c_d[child] = distance_temp
            sum_childs[parent]+=distance_temp

        for child in childs:
            gamma_topic[child] = distance_c_d[child]/ (sum_childs[parent]+1e-20)
#             if node_level.get(child)==n_depth-1: #leaf nodes
            if child not in tree_idxs.keys():
                list_idxs = []
                parent_nodes = leaf_parents[child]
                list_idxs.extend(parent_nodes)
                list_idxs.extend([child])
                leaf_prob[child] = tf.expand_dims(tf.reduce_prod(tf.concat([gamma_topic[node] for node in list_idxs], axis=1),axis=1), axis=-1)

    return leaf_prob, gamma_topic



def level_dist5(topic_coords, d, level_nodes, dist_type='gauss'):
    eta_topic = {}
    
    sum_eta = 0
    d_root = tf.sqrt(tf.reduce_sum((d-topic_coords[0])**2, axis=-1))#tf.norm(d,axis=-1)
    
    if dist_type=='gauss':
        eta_topic[0] = tf.exp(-0.5*tf.pow(d_root, 2))
    if dist_type=='inv':
        eta_topic[0] = 1./(1.+tf.pow(0.5*d_root, 2))
    
    sum_eta+=eta_topic[0]
    for level, nodes in level_nodes.items():
        if level!=0:
            distance_root_node = {}
            for node in nodes:
                distance_root_node[node] = tf.sqrt(tf.reduce_sum((d-topic_coords[node])**2, axis=-1))#distance_1(topic_coords, 0, node)#
            min_level_dist = tf.reduce_min(tf.convert_to_tensor(list(distance_root_node.values())), axis=0)

    #         eta_topic[level] =  tf.exp(-0.5*tf.pow(d_root - min_level_dist, 2))#1./(1.+tf.pow(d_root - tf.reduce_mean(sum_distance_level[level]), 2))#tf.exp(-0.5*tf.pow(d_root - tf.reduce_mean(sum_distance_level[level]), 2))
            if dist_type=='gauss':
                eta_topic[level] = tf.exp(-0.5*tf.pow(min_level_dist, 2)) #tf.exp(-0.5*tf.pow(d_root-min_level_dist, 2))#
            if dist_type=='inv':
                eta_topic[level] = 1./(1.+tf.pow(0.5*(min_level_dist), 2))#1./(1.+tf.pow(0.5*(d_root-min_level_dist), 2))#
            sum_eta+=eta_topic[level]

    sm_eta = []
    for level, nodes in level_nodes.items():
        sm_eta.append(eta_topic[level]/(sum_eta+1e-20))
    # prob_depth = tf.concat(sm_eta, 1)
    prob_depth = tf.stack(sm_eta, axis=-1)
    return prob_depth

import itertools


def distance_1(C_xy, idx1, idx2):
    return tf.sqrt(tf.reduce_sum((C_xy[idx1] - C_xy[idx2])*(C_xy[idx1] - C_xy[idx2]))+1e-20)

def _dijkstra(G, source, get_weight, pred=None, paths=None, target=None):
    G_succ = G.adj

    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {source: 0}
    c = count()
    fringe = []  # use heapq with (distance,label) tuples
    push(fringe, (0, next(c), source))
    while fringe:
        (d, _, v) = pop(fringe)
        
        if v in dist:
            continue  # already searched this node.
        dist[v] = d
        if v == target:
            break

        for u, e in G_succ[v].items():

            cost = get_weight(v, u, e)
            if cost is None:
                continue
            vu_dist = dist[v] + get_weight(v, u, e)

            if u in dist:
                if vu_dist < dist[u]:
                    raise ValueError('Contradictory paths found:',
                                     'negative weights?')
            elif u not in seen or vu_dist < seen[u]:
                seen[u] = vu_dist
                push(fringe, (vu_dist, next(c), u))
                if paths is not None:
                    paths[u] = paths[v] + [u]
                if pred is not None:
                    pred[u] = [v]
            elif vu_dist == seen[u]:
                if pred is not None:
                    pred[u].append(v)

    if paths is not None:
        return (dist, paths)
    if pred is not None:
        return (pred, dist)
    return dist
def single_source_dijkstra_path_length(G, source, weight_matrix=None, weight='weight'):
    get_weight = lambda u, v, data: data.get(weight, weight_matrix[min(u,v), max(u,v)])

    return _dijkstra(G, source, get_weight)
def all_pairs_dijkstra_path_length(G, weight_matrix=None,weight='weight'):
 
    length = single_source_dijkstra_path_length

    return {n: length(G, n, weight_matrix=weight_matrix, weight=weight) for n in G}
def distance_matrix(topic_coords, topics_spec, level_nodes, leaf_parents, topics):
    graph_pair_dist = {}
    leaf_prev_parent = {e:max(i) for e,i in leaf_parents.items()}
    
#     for (i,j) in itertools.combinations(topics, 2):
#         graph_pair_dist[(i,j)] = -1

    for leaf, parents in leaf_parents.items():
        prev_a = max(parents)
        d_ancestors = topics_spec[0,prev_a]#topic_coords_weight[(0, prev_a)]#1#distance_1(topic_coords, 0, prev_a)
        d_leaf_prev_a = topics_spec[prev_a,leaf]#topic_coords_weight[(leaf, prev_a)]#distance_1(topic_coords, leaf, prev_a)   

        graph_pair_dist[(0, leaf)] = d_ancestors + d_leaf_prev_a
        graph_pair_dist[(prev_a, leaf)] = d_leaf_prev_a
#         graph_pair_dist[(parents[0], parents[1])] = d_ancestors
        graph_pair_dist[(0, prev_a)] = d_ancestors
    

    for level, nodes in level_nodes.items():

            for combination in itertools.combinations(nodes, 2):
                v1 = min(combination)
                v2 = max(combination)
                if level==1:
                    graph_pair_dist[(v1, v2)] = graph_pair_dist[(0,v1)] + graph_pair_dist[(0, v2)]
                if level==2:
                    if leaf_prev_parent[v1] == leaf_prev_parent[v2]: #same parent
                        graph_pair_dist[(v1, v2)] = graph_pair_dist[(leaf_prev_parent[v1],v1)] + graph_pair_dist[(leaf_prev_parent[v2], v2)] #distance_1(topic_coords, leaf_prev_parent[v1], v1) + distance_1(topic_coords, leaf_prev_parent[v2], v2)
                    else: #different parent
                        graph_pair_dist[(v1, v2)] = graph_pair_dist[(0,v1)] + graph_pair_dist[(0, v2)]

    for i, j in itertools.product(level_nodes[1], level_nodes[2]):
        if (i,j) not in graph_pair_dist:
            graph_pair_dist[(i,j)] = graph_pair_dist[(0,i)] + graph_pair_dist[(0,j)]
                               
    return graph_pair_dist

def distance_matrix2(G, topics_weight, topic_idxs):
    graph_pairs = {topic:{} for topic in topic_idxs}
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), 1):
            if len(lst[i:i + n])==n:
                yield lst[i:i + n]

    for node in topic_idxs:
        shortest_path_source=nx.shortest_path(G,source=node)
        for k, nodes in shortest_path_source.items():
            d = 0
            for pair in chunks(nodes,2):
                d=d+topics_weight[min(pair), max(pair)]
            graph_pairs[node][k]=d 
    return graph_pairs


def compute_topic_coord_reg_kk3(G, topic_coords, topic_coords_weight, level_nodes, tree_idxs, child_parents, topic_idxs, topic_bow, tree_topic_embeddings):

    topics_vec = tf.nn.l2_normalize(topic_bow, 1)


    topics_weight = {}
    for parent,childs in tree_idxs.items():
        
        parent_norm = tf.nn.l2_normalize(tree_topic_embeddings[parent]) #tf.nn.l2_normalize(topic_bow[topic_idxs.index(parent)]) #
        for j in childs:
            child_norm = tf.nn.l2_normalize(tree_topic_embeddings[j]) #tf.nn.l2_normalize(topic_bow[topic_idxs.index(j)]) #
            
            weight = 1 - tf.reduce_sum(tf.multiply(parent_norm, child_norm))##tf.sqrt(tf.reduce_sum((tree_topic_embeddings[parent] - tree_topic_embeddings[j])**2)+1e-20)###

            topics_weight[parent,j] = 1.+weight#tf.clip_by_value(1.+weight, clip_value_min=0.5, clip_value_max=3.)#tf.clip_by_value(weight, clip_value_min=0.5, 

    graph_pair_dist = distance_matrix2(G, topics_weight, topic_idxs)

    v = {j:i for i,j in enumerate(topic_idxs)}
    m1 = []
    for i in topic_idxs:
        stack_col = []
        for j in topic_idxs:
            temp = graph_pair_dist[i][j]
            stack_col.append(temp)
        m1.append(tf.stack(stack_col))

    m = tf.stack(m1)

    
    eyeD1 = tf.eye(len(topic_coords.keys()))
    minuseyeD1 = 1-eyeD1
    A = tf.reshape(tf.convert_to_tensor(list(topic_coords.values())), [len(topic_idxs), 2])
    
    
    r = tf.reduce_sum(A*A, 1)

    # turn r into column vector
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    D = tf.sqrt(tf.sqrt(D**2+1e-20))#tf.sqrt(D+eyeD1)#tf.sqrt(tf.sqrt(D**2))
    
    cost_matrix = (1./2.)*(D/(m+eyeD1+1e-20)-1)**2
    cost = tf.matrix_band_part(cost_matrix, 0, -1)
    cost = tf.reduce_sum(cost*minuseyeD1)

    return cost            
import copy    
class HierarchicalNeuralTopicModel():
    def __init__(self, config):
        self.config = config
        self.vis = config.vis

        self.t_variables = {}
        self.tree_idxs = config.tree_idxs
        print(self.tree_idxs)
        self.topic_idxs = get_topic_idxs(self.tree_idxs)
        print(self.topic_idxs)
        self.child_to_parent_idxs = get_child_to_parent_idxs(self.tree_idxs)
        print('self.child_to_parent_idxs')
        print(self.child_to_parent_idxs)
        self.tree_depth = get_depth(self.tree_idxs)
        self.n_depth = max(self.tree_depth.values())
        
        if self.vis:
            self.level_nodes = get_level_nodes(self.tree_depth, self.config.tree_max_depth)
            print('self.tree_depth: ',self.tree_depth)
            print('self.n_depth: ', self.n_depth)
            print('self.level_nodes: ', self.level_nodes)
            self.leaf_parents = get_leaf_parents(self.tree_idxs, self.level_nodes, self.n_depth, self.topic_idxs, self.child_to_parent_idxs)
            
            self.node_level = {}
            for level, nodes in self.level_nodes.items():
                for node in nodes:
                    self.node_level[node] = level
                    
            # networkx
            self.G = nx.Graph()
            self.G.add_nodes_from(self.topic_idxs)


            pairs = [[(parent,j) for j in childs] for parent,childs in self.tree_idxs.items()]
            flat_pairs = [item for sublist in pairs for item in sublist]
            self.G.add_edges_from(flat_pairs)
            
        self.build()
        
    def build(self):
        

        def get_prob_topic(tree_prob_leaf, prob_depth):
            tree_prob_topic = defaultdict(float)
            leaf_ancestor_idxs = {leaf_idx: get_ancestor_idxs(leaf_idx, self.child_to_parent_idxs) for leaf_idx in tree_prob_leaf}
            print('leaf_ancestor_idxs: ', leaf_ancestor_idxs)
            for leaf_idx, ancestor_idxs in leaf_ancestor_idxs.items():
                prob_leaf = tree_prob_leaf[leaf_idx]
                for i, ancestor_idx in enumerate(ancestor_idxs):
                    if(ancestor_idx==leaf_idx):
                        sum_prob_depth = tf.reduce_sum(prob_depth[:, 0:i], -1)
                        tree_prob_topic[ancestor_idx] = prob_leaf * tf.expand_dims(1-sum_prob_depth, -1)
                    else:    
                        prob_ancestor = prob_leaf * tf.expand_dims(prob_depth[:, i], -1)
                        tree_prob_topic[ancestor_idx] += prob_ancestor
            prob_topic = tf.concat([tree_prob_topic[topic_idx] for topic_idx in self.topic_idxs], -1)
            return prob_topic     
        
        def get_tree_topic_bow(tree_topic_embeddings):
            tree_topic_bow = {}
            for topic_idx, depth in self.tree_depth.items():
                topic_embedding = tree_topic_embeddings[topic_idx]
                temperature = tf.constant(self.config.depth_temperature ** (1./depth), dtype=tf.float32)
                logits = self.beta_bn(tf.matmul(topic_embedding, self.bow_embeddings, transpose_b=True))#tf.matmul(topic_embedding, self.bow_embeddings, transpose_b=True)
                tree_topic_bow[topic_idx] = softmax_with_temperature(logits, axis=-1, temperature=temperature)
                
            return tree_topic_bow
        
        def get_topic_loss_reg(tree_topic_embeddings):
            def get_tree_mask_reg(all_child_idxs):        
                tree_mask_reg = np.zeros([len(all_child_idxs), len(all_child_idxs)], dtype=np.float32)
                for parent_idx, child_idxs in self.tree_idxs.items():
                    neighbor_idxs = child_idxs
                    for neighbor_idx1 in neighbor_idxs:
                        for neighbor_idx2 in neighbor_idxs:
                            neighbor_index1 = all_child_idxs.index(neighbor_idx1)
                            neighbor_index2 = all_child_idxs.index(neighbor_idx2)
                            if neighbor_index1==neighbor_index2:
                              tree_mask_reg[neighbor_index1, neighbor_index2] = 0.
                            else:
                              tree_mask_reg[neighbor_index1, neighbor_index2] = tree_mask_reg[neighbor_index2, neighbor_index1] = 1.
                return tree_mask_reg
            
            all_child_idxs = list(self.child_to_parent_idxs.keys())
            
            self.diff_topic_embeddings = tf.concat([tree_topic_embeddings[child_idx] - tree_topic_embeddings[self.child_to_parent_idxs[child_idx]] for child_idx in all_child_idxs], axis=0)
            diff_topic_embeddings_norm = self.diff_topic_embeddings / (tf.norm(self.diff_topic_embeddings+1e-20, axis=1, keepdims=True))
            # self.topic_dots = tf.clip_by_value(tf.matmul(diff_topic_embeddings_norm, tf.transpose(diff_topic_embeddings_norm)), -1., 1.)        
            self.topic_dots = tf.matmul(diff_topic_embeddings_norm, tf.transpose(diff_topic_embeddings_norm))
            
            self.tree_mask_reg = get_tree_mask_reg(all_child_idxs)
            
            #self.topic_losses_reg = tf.square(self.topic_dots - tf.eye(len(all_child_idxs))) * self.tree_mask_reg
            self.topic_losses_reg = tf.square(self.topic_dots - tf.ones((len(all_child_idxs), len(all_child_idxs)))) * self.tree_mask_reg
            self.topic_loss_reg = tf.reduce_sum(self.topic_losses_reg) #/ (2*tf.reduce_sum(self.tree_mask_reg))
            return self.topic_loss_reg
           
        # -------------- Build Model --------------
        tf.reset_default_graph()
        
        tf.set_random_seed(self.config.seed)
        
        self.t_variables['bow'] = tf.placeholder(tf.float32, [None, self.config.dim_bow])
        self.t_variables['keep_prob'] = tf.placeholder(tf.float32)
        self.t_variables['epoch'] = tf.placeholder(tf.float32)

        # encode bow
        with tf.variable_scope('topic/enc', reuse=False):
            # self.topic_coords_z = tf.get_variable('topic_z', [1, self.n_depth], dtype=tf.float32)
#             self.topic_coords_z = tf.get_variable('topic_z', [1, 1], dtype=tf.float32)
            # self.topic_coords_z = tf.Variable([1.], trainable=False)
#             hidden_bow_ = tf.layers.Dense(units=self.config.dim_hidden_bow, activation=tf.nn.tanh, name='hidden_bow')(self.t_variables['bow'])
            hidden_bow_1 = tf.layers.Dense(units=self.config.dim_hidden_bow, activation=tf.nn.tanh, name='hidden_bow_1_docx')(self.t_variables['bow'])
            hidden_bow_2 = tf.layers.Dense(units=self.config.dim_hidden_bow, activation=tf.nn.tanh, name='hidden_bow_2_docx')(hidden_bow_1)
        
            hidden_bow = tf.layers.Dropout(self.t_variables['keep_prob'])(hidden_bow_2)
            
            means_bow = tf.layers.Dense(units=self.config.dim_latent_bow, name='mean_bow_docx')(hidden_bow)
            means_bow = tf.layers.BatchNormalization(name='mean_bow_bn_docx')(means_bow)
            self.logvars_bow = tf.layers.Dense(units=self.config.dim_latent_bow, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), bias_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), name='logvar_bow_docx')(hidden_bow)
            self.logvars_bow = tf.layers.BatchNormalization(name='logvar_bow_bn_docx')(self.logvars_bow)
            self.latents_bow = tf.layers.BatchNormalization(name='latent_bow_bn_docx')(sample_latents(means_bow, self.logvars_bow)) # sample latent vectors
#             self.latents_bow = sample_latents(means_bow, self.logvars_bow)
#             self.latents_bow = (sample_latents(means_bow, self.logvars_bow))
#             self.doc_x = tf.layers.Dense(units=2, name='doc_x')(tf.layers.Dense(units=32, activation=tf.nn.softplus, name='doc_x')(self.latents_bow))
#             self.latents_bow = sample_latents(means_bow, self.logvars_bow)
            #prob_layer = lambda h: tf.nn.sigmoid(tf.matmul(self.latents_bow, h, transpose_b=True))
            self.topic_bn = tf.layers.BatchNormalization(name='topic_bn')
            self.beta_bn = tf.layers.BatchNormalization(name='beta_bn')
            emb_layer = lambda h: tf.layers.Dense(units=self.config.dim_emb, name='output')(tf.nn.tanh(h))#self.config.dim_emb
            
            
            
            
            if self.vis:
#                 self.topic_coords_list = tf.get_variable('topic_coords', [len(self.topic_idxs), 1, 2], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # topic coordinates
#                 self.topic_coords = { i : self.topic_coords_list[self.topic_idxs.index(i)] for i in self.topic_idxs}
                
#                 self.topic_coords = { i : tf.layers.BatchNormalization()(tf.get_variable('topic_coords'+str(i), [1, 2], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())) for i in self.topic_idxs}#tf.random_normal_initializer(mean=0.0, stddev=0.01)
                
                
                self.topic_coords = { i : self.topic_bn(tf.get_variable('topic_coords'+str(i), [1, 2], dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))) for i in self.topic_idxs}#tf.random_normal_initializer(mean=0.0, stddev=0.01)

                self.topic_coords_weights = {}

                self.tree_topic_embeddings, tree_states_sticks_topic, _  = doubly_rnn(self.config.dim_emb, self.config.dim_latent_path, self.t_variables['keep_prob'], self.tree_idxs,
                                                                                                    topic_coords=None, output_layer=emb_layer, name='sticks_topic', vis=self.vis)

                self.doc_coord = self.latents_bow
                self.prob_depth = level_dist5(self.topic_coords, self.doc_coord, self.level_nodes, dist_type=self.config.dist_type)#bp(sticks_depth, self.n_depth)
                
                
                self.tree_prob_leaf, self.gamma_topic = compute_pi_leaf2root(self.topic_coords, self.doc_coord, self.leaf_parents, self.node_level, self.tree_idxs, self.n_depth, self.config.batch_size, dist_type=self.config.dist_type)

            else:
                tree_sticks_topic, tree_states_sticks_topic, self.topic_coords = doubly_rnn(self.config.dim_latent_bow, self.tree_idxs, output_layer=prob_layer, name='sticks_topic', vis=self.vis)
            
                sticks_depth, hidden_state_depth = rnn(self.config.dim_latent_bow, self.n_depth, output_layer=prob_layer, name='prob_depth')
                self.tree_prob_leaf = tsbp(tree_sticks_topic, self.tree_idxs) #dict of N_leaf has size of N * 1
                self.prob_depth = sbp(sticks_depth, self.n_depth)

            self.prob_topic = get_prob_topic(self.tree_prob_leaf, self.prob_depth)# n_batch x n_topic

        # decode bow
        with tf.variable_scope('shared', reuse=False):
            self.bow_embeddings = tf.get_variable('emb', [self.config.dim_bow, self.config.dim_emb], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # embeddings of vocab
#             self.bow_embeddings = tf.get_variable('emb', [self.config.dim_bow, 256], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) # embeddings of vocab

        with tf.variable_scope('topic/dec', reuse=False):
            # emb_layer = lambda h: tf.layers.Dense(units=self.config.dim_emb, name='output')(tf.nn.tanh(h))
            # self.tree_topic_embeddings, tree_states_topic_embeddings, _ = doubly_rnn(self.config.dim_emb, self.config.dim_latent_path, self.tree_idxs, output_layer=emb_layer, name='emb_topic', vis=False)
            # self.tree_topic_embeddings = self.topic_coords
            self.tree_topic_bow = get_tree_topic_bow(self.tree_topic_embeddings) # bow vectors for each topic
            self.topic_bow = tf.concat([self.tree_topic_bow[topic_idx] for topic_idx in self.topic_idxs], 0) # KxV

            self.logits_bow = tf_log(tf.matmul(self.prob_topic, self.topic_bow)) # predicted bow distribution N_Batch x  V
        
        # define losses
        self.global_step = tf.Variable(0, name='global_step',trainable=False)

        self.coord_reg3 = compute_topic_coord_reg_kk3(self.G, self.topic_coords, self.topic_coords_weights, self.level_nodes, self.tree_idxs, self.leaf_parents, self.topic_idxs,
                                                     self.topic_bow, self.tree_topic_embeddings)
        
        
        self.topic_losses_recon = -tf.reduce_sum(tf.multiply(self.t_variables['bow'], self.logits_bow), 1)
        self.topic_loss_recon = tf.reduce_mean(self.topic_losses_recon) # negative log likelihood of each words
        
        self.topic_losses_kl = compute_kl_losses(means_bow, self.logvars_bow, means_prior=0., logvars_prior=tf.math.log(1.)) # KL divergence b/w latent dist & gaussian std
        self.topic_loss_kl = tf.reduce_mean(self.topic_losses_kl, 0) #mean of kl_losses over batches        
        
        self.topic_embeddings = tf.concat([self.tree_topic_embeddings[topic_idx] for topic_idx in self.topic_idxs], 0) # temporary
        self.topic_loss_reg = get_topic_loss_reg(self.tree_topic_embeddings)
        
        
        #reg3_ = self.t_variables['epoch']*10
        
        print('self.config.reg3: ', self.config.reg3)
        #reg_lvl0 = tf.reduce_sum(self.prob_depth[:,0])
        self.loss = self.config.batch_size * (self.topic_loss_recon + self.topic_loss_kl) - self.config.reg * self.topic_loss_reg  + self.config.reg3 * (self.coord_reg3)# + self.config.reg4 
        # define optimizer
        if self.config.opt == 'Adam':
            optimizer = tf.train.AdamOptimizer(self.config.lr)
        elif self.config.opt == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(self.config.lr)

        self.grad_vars = optimizer.compute_gradients(self.loss)
#         print(self.grad_vars)
#         self.clipped_grad_vars = [(tf.clip_by_value(grad, -self.config.grad_clip, self.config.grad_clip), var) for grad, var in self.grad_vars]
        
#         self.opt = optimizer.apply_gradients(self.clipped_grad_vars, global_step=self.global_step)
        self.opt = optimizer.apply_gradients(self.grad_vars, global_step=self.global_step)

        # monitor
        self.n_bow = tf.reduce_sum(self.t_variables['bow'], 1)
        self.topic_ppls = tf.divide(self.topic_losses_recon + self.topic_losses_kl, tf.maximum(1e-5, self.n_bow))
    
        # growth criteria
        self.n_topics = tf.multiply(tf.expand_dims(self.n_bow, -1), self.prob_topic)
        
        self.arcs_bow = tf.acos(tf.matmul(tf.linalg.l2_normalize(self.bow_embeddings, axis=-1), tf.linalg.l2_normalize(self.topic_embeddings, axis=-1), transpose_b=True)) # n_vocab x n_topic
        self.rads_bow = tf.multiply(tf.matmul(self.t_variables['bow'], self.arcs_bow), self.prob_topic) # n_batch x n_topic
    
    def get_feed_dict(self, batch, epoch, mode='train'):
        
        bow = np.array([instance.bow for instance in batch]).astype(np.float32)
        current_batch_size=bow.shape[0]
        keep_prob = self.config.keep_prob if mode == 'train' else 1.0
        feed_dict = {self.t_variables['epoch']: epoch,
                    self.t_variables['bow']: bow, 
                    self.t_variables['keep_prob']: keep_prob
        }
        return  feed_dict, current_batch_size
    
     
    
    def update_tree(self, topic_prob_topic, recur_prob_topic, disable_add=False):
        assert len(self.topic_idxs) == len(recur_prob_topic) == len(topic_prob_topic)
        update_tree_flg = False

        def add_topic(topic_idx, tree_idxs, num_childs):
            if topic_idx in tree_idxs:
                if num_childs==1:
                    # child_idx = min([10*topic_idx+i for i in range(1, 10) if 10*topic_idx+i not in tree_idxs[topic_idx]])
                    child_idx = min([MAX_CHILD_NODES*topic_idx+i for i in range(1, MAX_CHILD_NODES) if MAX_CHILD_NODES*topic_idx+i not in tree_idxs[topic_idx]])
                    tree_idxs[topic_idx].append(child_idx)        
                else:
                    child_idx = MAX_CHILD_NODES*topic_idx+1
                    tree_idxs[topic_idx] = [MAX_CHILD_NODES*topic_idx+i+1 for i in range(num_childs)]
                    
            return tree_idxs, child_idx
        def remove_branch(idx, tree_idxs):
            if idx in tree_idxs:
                removed_childs = tree_idxs.pop(idx)
                for child in removed_childs:
                    remove_branch(child, tree_idxs)

        def remove_topic(parent_idx, child_idx, tree_idxs):
            if parent_idx in tree_idxs:
                tree_idxs[parent_idx].remove(child_idx)
                if child_idx in tree_idxs:
                    remove_branch(child_idx, tree_idxs)
            return tree_idxs
        
        if not disable_add:
            added_tree_idxs = copy.deepcopy(self.tree_idxs)
            for parent_idx, child_idxs in self.tree_idxs.items():
                parent_depth = self.tree_depth[parent_idx]
                prob_topic = topic_prob_topic[parent_idx]
                add_thresh = (1.0/(self.config.tree_expected_min_child*2**(parent_depth-1)))
                add_thresh = max(add_thresh, 0.01)
                if prob_topic > add_thresh:
                    update_tree_flg = True
                    added_tree_idxs, parent_idx = add_topic(parent_idx, added_tree_idxs, 1)

            for level, nodes in self.level_nodes.items():
                for node in nodes:
                    if (node not in self.tree_idxs.keys()):
                        prob_topic = topic_prob_topic[node]
                        add_thresh = (1.0/(self.config.tree_expected_min_child*2**(level)))
                        add_thresh = max(add_thresh, 0.01)
                        if (level+1<self.config.tree_max_depth):
                            if prob_topic > add_thresh:
                                update_tree_flg = True
                                added_tree_idxs[node] = []
                                added_tree_idxs, parent_idx = add_topic(node, added_tree_idxs, 2)
        else:
            added_tree_idxs = copy.deepcopy(self.tree_idxs)
        
        removed_tree_idxs = copy.deepcopy(added_tree_idxs)
        for parent_idx, child_idxs in self.tree_idxs.items():
            probs_child = np.array([recur_prob_topic[child_idx] for child_idx in child_idxs])
            for prob_child, child_idx in zip(probs_child, child_idxs):
                curr_node_level = self.node_level[child_idx]
                remove_thresh = (1.0/(self.config.tree_expected_max_child*2**(curr_node_level)))
                remove_thresh = max(remove_thresh, 0.01)
                if prob_child < remove_thresh:
                    update_tree_flg = True
                    removed_tree_idxs = remove_topic(parent_idx, child_idx, removed_tree_idxs)
        
        removed_tree_idxs_copy = copy.deepcopy(removed_tree_idxs)
        for parent_idx, child_idxs in removed_tree_idxs.items():
            if(child_idxs==[]): removed_tree_idxs_copy.pop(parent_idx)
        removed_tree_idxs = removed_tree_idxs_copy
        
        return removed_tree_idxs, update_tree_flg    



def compute_topic_specialization(sess, model, instances, verbose=False):
    topics_vec = sess.run(tf.nn.l2_normalize(model.topic_bow, 1))
    norm_bow = np.sum([instance.bow for instance in instances], 0)
    norm_vec = norm_bow / np.linalg.norm(norm_bow)

    topics_spec = 1 - topics_vec.dot(norm_vec)

    depth_topic_idxs = defaultdict(list)
    for topic_idx, depth in model.tree_depth.items():
        depth_topic_idxs[depth].append(topic_idx)

    depth_specs = {}
    if verbose: print('Topic Specialization:', end=' ')
    for depth, topic_idxs in depth_topic_idxs.items():
        topic_indices = np.array([model.topic_idxs.index(topic_idx) for topic_idx in topic_idxs])
        depth_spec = np.mean(topics_spec[topic_indices])
        depth_specs[depth] = depth_spec
        if verbose: print('depth %i = %.2f' % (depth, depth_spec), end=', ')
    print('')
    
    return depth_specs

def compute_hierarchical_affinity(sess, model, verbose=False):
    def get_cos_sim(parent_to_child_idxs):
        parent_child_bows = {parent_idx: np.concatenate([normed_tree_topic_bow[child_idx] for child_idx in child_idxs], 0) for parent_idx, child_idxs in parent_to_child_idxs.items()}
        cos_sim = np.mean([np.mean(normed_tree_topic_bow[parent_idx].dot(child_bows.T)) for parent_idx, child_bows in parent_child_bows.items()])
        return cos_sim    
    
    tree_topic_bow = {topic_idx: tf.nn.l2_normalize(topic_bow) for topic_idx, topic_bow in model.tree_topic_bow.items()}
    
    normed_tree_topic_bow = sess.run(tree_topic_bow)

    parent_to_non_child_idxs = {}
    for level,nodes in model.level_nodes.items():
        if level==0:
            continue
        for node1 in nodes:
            parent_to_non_child_idxs[node1] = []
            for node2 in nodes:
                if node1!=node2 and node2 in model.tree_idxs:
                    parent_to_non_child_idxs[node1].extend(model.tree_idxs[node2])
    parent_to_non_child_idxs = {k: v for k, v in parent_to_non_child_idxs.items() if v != []}  

    parent_to_child_idxs = copy.deepcopy(model.tree_idxs)
    del parent_to_child_idxs[0]

    child_cos_sim = get_cos_sim(parent_to_child_idxs)
    unchild_cos_sim = get_cos_sim(parent_to_non_child_idxs)

    
    if verbose: print('Hierarchical Affinity: child = %.2f, non-child = %.2f'%(child_cos_sim, unchild_cos_sim))
    return child_cos_sim, unchild_cos_sim
def _compute_hierarchical_affinity(sess, model, verbose=False):
    def get_cos_sim(parent_to_child_idxs):
        parent_child_bows = {parent_idx: np.concatenate([normed_tree_topic_bow[child_idx] for child_idx in child_idxs], 0) for parent_idx, child_idxs in parent_to_child_idxs.items()}
        cos_sim = np.mean([np.mean(normed_tree_topic_bow[parent_idx].dot(child_bows.T)) for parent_idx, child_bows in parent_child_bows.items()])
        return cos_sim    
    
    tree_topic_bow = {topic_idx: tf.nn.l2_normalize(topic_bow) for topic_idx, topic_bow in model.tree_topic_bow.items()}
    
    normed_tree_topic_bow = sess.run(tree_topic_bow)

    third_child_idxs = [child_idx for child_idx, depth in model.tree_depth.items() if depth==3]
    second_parent_to_child_idxs = {parent_idx:child_idxs for parent_idx, child_idxs in model.tree_idxs.items() if model.tree_depth[parent_idx] == 2}
    second_parent_to_unchild_idxs = {parent_idx: [child_idx for child_idx in third_child_idxs if child_idx not in child_idxs] for parent_idx, child_idxs in second_parent_to_child_idxs.items()}

    if sum(len(unchilds) for unchilds in second_parent_to_unchild_idxs.values()) > 0:
        child_cos_sim = get_cos_sim(second_parent_to_child_idxs)
        unchild_cos_sim = get_cos_sim(second_parent_to_unchild_idxs)
    else:
        child_cos_sim = get_cos_sim(second_parent_to_child_idxs)
        unchild_cos_sim = 0
    
    if verbose: print('Hierarchical Affinity: child = %.2f, non-child = %.2f'%(child_cos_sim, unchild_cos_sim))
    return child_cos_sim, unchild_cos_sim



import math
def validate(sess, batches, epoch, model, config):
    losses = []
    ppl_list = []
    prob_topic_list = []
    n_bow_list = []
    n_topics_list = []
    doc_list = []
    for batch in batches:
#         feed_dict, current_batch_size = model.get_feed_dict(batch, epoch, instances_dev.bow)
        feed_dict, current_batch_size = model.get_feed_dict(batch, epoch, mode='test')
        if current_batch_size!=model.config.batch_size:
            continue
        loss_batch, topic_loss_recon_batch, topic_loss_kl_batch, topic_loss_reg_batch, ppls_batch, prob_topic_batch, n_bow_batch, n_topics_batch, x \
            = sess.run([model.loss, model.topic_loss_recon, model.topic_loss_kl, model.topic_loss_reg, model.topic_ppls, model.prob_topic, model.n_bow, model.n_topics, model.latents_bow], feed_dict = feed_dict)
        losses += [[loss_batch, topic_loss_recon_batch, topic_loss_kl_batch, topic_loss_reg_batch]]
        ppl_list += list(ppls_batch)
        prob_topic_list.append(prob_topic_batch)
        n_bow_list.append(n_bow_batch)
        n_topics_list.append(n_topics_batch)
        
        doc_list.extend(x)
    print(losses)
    loss_mean, topic_loss_recon_mean, topic_loss_kl_mean, topic_loss_reg_mean = np.mean(losses, 0)
    ppl_mean = np.exp(np.mean(ppl_list))
    
    probs_topic = np.concatenate(prob_topic_list, 0)
    
    n_bow = np.concatenate(n_bow_list, 0)
    n_topics = np.concatenate(n_topics_list, 0)
    probs_topic_mean = np.sum(n_topics, 0) / np.sum(n_bow)
    
    
    # cal knn
    c = []
    labels = np.load('data/'+config.data_type+'/labels.npy', allow_pickle=True, encoding='bytes')
    label_names=list(set(labels))

    for idx, points in enumerate(doc_list):
        c.append(labels[idx])

    knn = cal_knn(doc_list,  c)
    print('KNN: ', knn)
    return loss_mean, topic_loss_recon_mean, topic_loss_kl_mean, topic_loss_reg_mean, ppl_mean, probs_topic_mean


