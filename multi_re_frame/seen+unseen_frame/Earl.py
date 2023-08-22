import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import GRUCell
import numpy as np
from torch import tensor

import time


def is_row_empty(tensor, row_index):
    row = tensor[row_index]
    return torch.all(row == 0).item()

class Earl(Module):
    def __init__(self,
                 num_embed_units,
                 entity_embeddings,
                 relation_embeddings,
                 device
                 ):
        super(Earl,self).__init__()
        #self.sub_hidden = sub_hidden # seed_entity的embedding。维度为[len(seed_entity),num_embed_units]
        # self.triple_num = triple_num    # 采样得到的知识图谱元组数量。
        # self.encoder_batch_size = encoder_batch_size
        self.num_embed_units = num_embed_units 
        self.entity_embeddings = entity_embeddings
        self.relation_embeddings = relation_embeddings
        self.device = device

        self.path_cell = GRUCell(num_embed_units,num_embed_units)

        self.sub_linear = nn.Linear(num_embed_units,num_embed_units)   # Equation 3
        self.sub_tanh = nn.Tanh()

        self.obj_linear = nn.Linear(num_embed_units,num_embed_units)   # Equarion 4
        self.obj_tanh = nn.Tanh()

        print(type(self.entity_embeddings),type(self.relation_embeddings))


    def forward(self,dialogue_representation, extra_dialogue_representation,seed_entities, subgraph, sub_embedding,entity_state_1_jump,
                entity_state_2_jump,head_entities_1_jump,head_entities_2_jump,tail_entities_1_jump,tail_entities_2_jump,edge_relations_1_jump,
                edge_relations_2_jump,unseen_rel,node2nodeID,nodeID2node,last_graph , last_entity_rep):
        '''
        nodeID2node:新节点编号 --> 原节点编号
        node2nodeID:原节点编号 --> 新节点编号
        节点在新图中的编号与下标一致
        '''

        old_ndata = subgraph.ndata['nodeId']

        entity_embeddings = torch.zeros((len(old_ndata),self.num_embed_units), requires_grad = True).to(self.device) # 该子图中所有节点的emebdding


        seed = int(seed_entities[0].to('cpu'))

        seed_index = nodeID2node[seed]
        seed_state = 0 if len(sub_embedding) else 1

        old_ID2embedding = {}

        # 起始节点如果为seen，直接索引embeddings；为unseen，使用sample_mask
        if not seed_state:
            sub_hidden = sub_embedding[0].to(self.device)

            # Equation 3
            sub_embedding = self.sub_linear(sub_hidden) 
            sub_embedding = self.sub_tanh(sub_embedding)
            
        else:
            sub_embedding = self.entity_embeddings(tensor(seed_index).to(self.device)).to(self.device)

        sub_embedding = sub_embedding.unsqueeze(0)
        entity_embeddings[seed] = sub_embedding

        # Equation 5
        r0 = self.path_cell(dialogue_representation,sub_embedding)


        seen_1_tail,unseen_1_tail = entity_state_1_jump[0],entity_state_1_jump[1]
        seen_2_tail,unseen_2_tail,head_2_jump = entity_state_2_jump[0],entity_state_2_jump[1],entity_state_2_jump[2]

        seen_1_tail = list(set(seen_1_tail))
        seen_2_tail = list(set(seen_2_tail))

        unseen_1_rel,unseen_2_rel = unseen_rel[0],unseen_rel[1]

        # print(seen_1_tail,seen_2_tail)
        seen_1_jump_embedding = self.entity_embeddings(tensor(seen_1_tail).to(self.device)) if len(seen_1_tail) else []
        seen_2_jump_embedding = self.entity_embeddings(tensor(seen_2_tail).to(self.device)) if len(seen_2_tail) else []

        # 遍历第一跳所有seen尾节点
        for i in range(len(seen_1_jump_embedding)):
            old_id = seen_1_tail[i]
            new_id = node2nodeID[old_id]
            embedding = seen_1_jump_embedding[i]

            old_ID2embedding[old_id] = embedding
            entity_embeddings[new_id] = embedding.unsqueeze(0)

        # 遍历第二跳所有seen尾节点
        for i in range(len(seen_2_jump_embedding)):
            old_id = seen_2_tail[i]
            new_id = node2nodeID[old_id]
            embedding = seen_2_jump_embedding[i]

            entity_embeddings[new_id] = embedding.unsqueeze(0)


        # 遍历第一跳的unseen尾节点
        rel_embedding_1 = self.relation_embeddings(tensor(unseen_1_rel).to(self.device)) if len(unseen_1_rel) else []
        for i in range(len(unseen_1_tail)):
            old_id = unseen_1_tail[i]
            new_id = node2nodeID[old_id]

            # Equation 6
            rel_embedding = rel_embedding_1[i].unsqueeze(0)

            rj = self.path_cell(r0,rel_embedding)

            # Equation 4
            embedding = self.obj_linear(rj)
            embedding = self.obj_tanh(embedding)

            old_ID2embedding[old_id] = embedding

            entity_embeddings[new_id] = embedding.unsqueeze(0)


        
        # 遍历第二跳所有unseen尾节点
        unseen_entity_2_jump_embedding = {}
        rel_embedding_2 = self.relation_embeddings(tensor(unseen_2_rel).to(self.device)) if len(unseen_2_rel) else []
        for i in range(len(unseen_2_tail)):
            tail_id = tail_entities_2_jump[i]
            new_id = node2nodeID[old_id]
            head_id = head_2_jump[i]

            head_embedding = old_ID2embedding[head_id]
            rel_embedding = rel_embedding_2[i]
            tail_embedding = (head_embedding + rel_embedding).squeeze()
            if new_id not in unseen_entity_2_jump_embedding.keys():
                unseen_entity_2_jump_embedding[new_id] = [tail_embedding]
            else:
                unseen_entity_2_jump_embedding[new_id].append(tail_embedding)

                
        # 求avgpool
        for key,value in unseen_entity_2_jump_embedding.items():
            new_id = key
            tail_embeddings_all = value
            stacked_tensor = torch.stack(tail_embeddings_all)
            avgpool_embedding = torch.mean(stacked_tensor,dim = 0)
            entity_embeddings[new_id] = avgpool_embedding.unsqueeze(0)

       # print('earl:',end_time - start_time)

        # --------------------------------------------------#

        return entity_embeddings

