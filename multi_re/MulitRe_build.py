import math
from collections import defaultdict
import gc

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from sklearn.metrics import roc_auc_score
from torch import tensor
from Earl import Earl
from copy import deepcopy


from tqdm import tqdm
from time import time

from attnio_model_new import AttnIO


def _get_instance_path_loss(graph, path):
    epsilon = 1e-30
    # denominator = (graph.num_nodes()*epsilon+1)
    scores = []
    for i in range(len(path)):
        # node_time_scores = ((graph.ndata["a_"+str(i)])/denominator) + (epsilon/denominator)
        node_time_scores = graph.ndata["a_"+str(i)] + epsilon
        node_time = path[i]
        score = node_time_scores[node_time]
        scores.append(-torch.log(score))
    scores = torch.stack(scores)
    return scores.sum(-1)


class MultiReModel(nn.Module):
    def __init__(self, opt):
        super(MultiReModel, self).__init__()

        self.device = opt["device"]
        self.n_entity_seen = opt["n_entity_seen"]
        self.n_entity_unseen = opt["n_entity_unseen"]

        self.n_relation = opt["n_relation"]
        self.out_dim = opt["out_dim"]
        self.in_dim = opt["in_dim"]
        self.lr = opt["lr"]
        self.lr_reduction_factor = opt["lr_reduction_factor"]
        self.epochs = opt["epoch"]
        self.attn_heads = opt["attn_heads"]
        self.beam_size = opt["beam_size"]
        self.clip = opt["clip"]
        self.self_loop_id = opt["self_loop_id"]
        self.batch_size = opt['batch_size']

        self.model_directory = opt["model_directory"]
        self.model_name = opt["model_name"]

        self.n_hop=opt['n_hop']
        self.n_max = opt['n_max']

        self.entity_emb_seen = Embedding(self.n_entity_seen + 1, 768)
        self.entity_emb_seen.weight.data.copy_(opt["entity_embeddings_seen"])
        #opt["entity_embedding"]就是所有的embedding
        self.relation_emb = Embedding(self.n_relation + 1, 768)
        self.relation_emb.weight.data.copy_(opt["relation_embeddings"])

        self.attnIO = AttnIO(self.in_dim, self.out_dim, self.attn_heads, self.entity_emb_seen, self.relation_emb, self.self_loop_id, self.device)
        self.earl = Earl(self.in_dim,opt["relation_embeddings"],self.device)
        
        self.attnIO = self.attnIO.cuda()
        self.earl=self.earl.cuda()
        # self.optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, self.parameters()), self.lr)
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.metrics = defaultdict(float)
        self.counts = defaultdict(int)
        self.reset_metrics()
    
    def reset_metrics(self):
        for key in self.metrics:
            self.metrics[key] = 0.0
        for key in self.counts:
            self.counts[key] = 0
    
    def report(self):
        m = {}
        # Top-k recommendation Recall
        for x in sorted(self.metrics):
            if x.startswith("recall"):
                m[x] = self.metrics[x] / self.counts[x]

        for k, v in m.items():
            # clean up: rounds to sigfigs and converts tensors to floats
            m[k] = round_sigfigs(v, 4)
        return m

    def forward(self, dialogue_representation, seed_entities, subgraph,sample_mask,node2nodeId,flag,state,startEntity_in_seenKG):
        if  (state == 'train' and flag == 0) or (state == 'test' and flag == 0) or (state == 'valid' and not startEntity_in_seenKG):
            graph = deepcopy(subgraph)
            new_ndata = self.earl(graph = graph,encoder_output = dialogue_representation, seed_entities = seed_entities,sample_mask = sample_mask,node2nodeId = node2nodeId)
            subgraph = self.attnIO(graph = subgraph, seed_set = seed_entities, dialogue_context = dialogue_representation,unseen_embeddings = new_ndata,_new = 1)
        else:
            #print(subgraph.ndata)
            subgraph = self.attnIO(subgraph, seed_entities, dialogue_representation,_new = 0)
        return subgraph


    def process_batch(self, batch_data,train):     

        batch_loss = []

        for one_batch_data in batch_data:
            one_dialog_loss = []
            flag,state,sample_num = one_batch_data[7],one_batch_data[8],one_batch_data[9]

            for i in range(sample_num):

                dialogue_representation, seed_entities, subgraph, paths,sample_mask,node2nodeID,startEntity_in_seenKG  = one_batch_data[0][i],one_batch_data[1][i],one_batch_data[2][i],one_batch_data[3][i],one_batch_data[4][i],one_batch_data[5][i],one_batch_data[6][i]

                if len(seed_entities) == 0:
                    continue
                if ((state == 'train' and flag == 0) or (state == 'test' and flag == 0) or (state == 'valid' and not startEntity_in_seenKG)) and len(sample_mask) == 0:
                    continue
                else:
                    updated_subgraphs = []
                    dialogue_representation = dialogue_representation.to(self.device) # [1,768]
                    seed_entities = seed_entities.to(self.device)
                    subgraph = subgraph.to(self.device)
                    paths = paths.to(self.device)
             
                    updated_subgraph, expilcit_entity_rep = self(dialogue_representation, seed_entities, subgraph,sample_mask,node2nodeID,flag,state,startEntity_in_seenKG)
                    updated_subgraphs.append(updated_subgraph)

                    instance_loss = _get_instance_path_loss(updated_subgraph, paths)
                    one_dialog_loss.append(instance_loss)
            
            # 如果没有进行训练，则暂存一个0，后续删除
            if len(one_dialog_loss) > 0:
                one_dialog_loss = torch.stack(one_dialog_loss).sum(-1)/len(one_dialog_loss)
            else:
                one_dialog_loss = tensor(0.0).to(self.device)
            batch_loss.append(one_dialog_loss)
        
        # 找到不符合要求的数据的索引
        indices_to_remove = [i for i, loss in enumerate(batch_loss) if loss.item() == 0.0]

        # 如果全部数据都不符合要求，则保留一个值为 0 的张量
        if len(indices_to_remove) == len(batch_loss):
            indices_to_remove.pop()

        # 删除不符合要求的数据
        for index in sorted(indices_to_remove, reverse=True):
            batch_loss.pop(index)

        # 计算平均损失
        batch_loss = torch.stack(batch_loss).sum(-1)/len(batch_loss)

        if train:
            self.optimizer.zero_grad()
            if batch_loss.item() != 0.0:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                self.optimizer.step()

        return batch_loss.item()
    
    def train_model(self,train_dataloader,valid_dataloader):
        self.optimizer.zero_grad()
        ins = 0 
        for epoch in range(self.epochs):
            self.train()

            # 训练阶段
            train_loss = 0
            cnt = 0
            for batch in tqdm(train_dataloader):
                train_loss += self.process_batch(batch,True)
                cnt += 1
            train_loss /= cnt

            # # 验证阶段
            # dev_loss,count = 0,0

            # cnt = 0
            # for batch in tqdm(valid_dataloader):
            #     batch_loss = self.process_batch(batch,False)
            #     dev_loss += batch_loss
            #     cnt += 1

            # # 全部平均
            # dev_loss /= cnt
            
            #self.lr_scheduler.step(dev_loss)
            print("Epoch: %d, Train Loss: %f" %(epoch, train_loss))
            # print("Epoch: %d, Dev All Loss: %f" %(epoch, dev_loss))
            
            # Logging parameters
            p = list(self.named_parameters())
            # logger.scalar_summary("Train Loss", train_loss, ins+1)
            # logger.scalar_summary("Dev Loss", dev_loss, ins+1)
            ins+=1
            if (epoch+1)%2==0:
                torch.save(self.state_dict(), f'{self.model_directory}{self.model_name}_epoch-{epoch+1}')
            
            if epoch == 4 or epoch == 9 or epoch ==14:
                self.lr *= self.lr_reduction_factor

