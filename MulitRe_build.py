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
from Earl import Earl


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

        self.entity_emb_seen = Embedding(self.n_entity_seen, 768)
        self.entity_emb_seen.weight.data.copy_(opt["entity_embeddings_seen"])
        #opt["entity_embedding"]就是所有的embedding
        self.relation_emb = Embedding(self.n_relation, 768)
        self.relation_emb.weight.data.copy_(opt["relation_embeddings"])

        self.attnIO = AttnIO(self.in_dim, self.out_dim, self.attn_heads, self.entity_emb_seen, self.relation_emb, self.self_loop_id, self.device)
        self.earl = Earl()
        
        self.model=self.model.cuda()
        self.optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, self.parameters()), self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)

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

    def forward(self, dialogue_representation, seed_entities, subgraph,flag):
        subgraph = self.model(subgraph, seed_entities, dialogue_representation)
        return subgraph


    def process_batch(self, batch_data,train):     
        batch_loss = []
        flag,batch_data = batch_data['flag'],batch_data['datas']
        for data in batch_data:
            one_dialog_loss = 0
            for sample in data:
                dialogue_representation, seed_entities, subgraph, paths = sample[0],sample[1],sample[2],sample[3]
                sample_mask = sample[4] if len(sample) == 5 else None

                
                updated_subgraphs = []

                dialogue_representation = dialogue_representation.to(self.device) # [1,768]
                seed_entity = seed_entities.to(self.device)
                subgraph = subgraph.to(self.device)
                path = paths.to(self.device)
                updated_subgraph, expilcit_entity_rep = self(dialogue_representation, seed_entity, subgraph,flag)
                updated_subgraphs.append(updated_subgraph)
                instance_loss = _get_instance_path_loss(updated_subgraph, path)
                one_dialog_loss += instance_loss

            batch_loss.append(one_dialog_loss)
                
        batch_loss = torch.stack(batch_loss).sum(-1)/len(batch_data)
            
        
        if train:
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            self.optimizer.step()

        return batch_loss.item(), updated_subgraphs
    
    def train_model(self, MulitRe_dataset_train):
        self.optimizer.zero_grad()
        ins=0
        for epoch in range(self.epochs):
            self.train()
            train_loss = 0
            cnt = 0

            
            dialogID = 1
            # 训练阶段
            while dialogID - 1<= MulitRe_dataset_train.train_unseen[1]:
                batch_data = MulitRe_dataset_train.get_batch_data(dialogID)
                batch_loss = self.process_batch(batch_data,True)
                train_loss += batch_loss 
                cnt += 1
                dialogID += len(batch_data)
            
            train_loss = train_loss/cnt

            
            cnt = 0
            dev_loss = 0
            while MulitRe_dataset_train.valid_seen[0] <= dialogID - 1 <= MulitRe_dataset_train.valid_unseen[1]:
                batch_data = MulitRe_dataset_train.get_batch_data(dialogID)
                batch_loss = self.process_batch(batch_data,False)
                dev_loss += batch_loss 
                cnt += 1
                dialogID += len(batch_data)
        
            dev_loss = dev_loss/cnt
            
            self.lr_scheduler.step(dev_loss)
            print("Epoch: %d, Train Loss: %f" %(epoch, train_loss))
            print("Epoch: %d, Dev Loss: %f" %(epoch, dev_loss))
            
            # Logging parameters
            p = list(self.named_parameters())
            # logger.scalar_summary("Train Loss", train_loss, ins+1)
            # logger.scalar_summary("Dev Loss", dev_loss, ins+1)
            ins+=1
            if (epoch+1)%2==0:
                torch.save(self.state_dict(), self.model_directory+self.model_name+"_"+str(epoch+1))       

