import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import GRUCell
import numpy as np
from torch import tensor


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


    def forward(self,graph,encoder_output,seed_entities,sample_mask,node2nodeId, nodeID2node, entity_state, state):
        '''
        nodeID2node:新节点编号 --> 原节点编号
        node2nodeID:原节点编号 --> 新节点编号
        节点在新图中的编号与下标一致
        '''
        id2embedding = {}

        heads,tails = graph.edges()[0].tolist(),graph.edges()[1].tolist()

        old_ndata = graph.ndata['nodeId']


        entity_embeddings = torch.zeros((len(old_ndata),self.num_embed_units), requires_grad = True).to(self.device) # 该子图中所有节点的emebdding

        # print(nodeID2node)
        # print(node2nodeId)
        # print(heads)
        # print(tails)
        # print(entity_state)

        seed = int(seed_entities[0].to('cpu'))

        seed_index = nodeID2node[seed]
        seed_state = 0 if len(sample_mask) else 1

        

        head_indices = [idx for idx,num in enumerate(heads) if num == seed]
        rels = [graph.edata['edge_type'][idx] for idx in head_indices]
        obj_id = [tails[idx] for idx in head_indices]

        # print('rels',rels)
        # print('obj_id',obj_id)

        tails_state = []
        for i in range(len(obj_id)):
            tails_state.append(entity_state[nodeID2node[obj_id[i]]])


        rel_embedding = [self.relation_embeddings[key] for key in rels]
        
        # 起始节点如果为seen，直接索引embeddings；为unseen，使用sample_mask
        if not seed_state:
            sub_hidden = sample_mask[0].to(self.device)

            # Equation 3
            sub_embedding = self.sub_linear(sub_hidden) 
            sub_embedding = self.sub_tanh(sub_embedding)
            
        else:
            sub_embedding = self.entity_embeddings(tensor(seed_index).to(self.device)).to(self.device)

        sub_embedding = sub_embedding.unsqueeze(0)
        entity_embeddings[seed] = sub_embedding

        # Equation 5
        r0 = self.path_cell(encoder_output,sub_embedding)

        #id2embedding[int(seed)] = sub_embedding.squeeze()

        for j in range(len(obj_id)):
            tail_index = obj_id[j]

            if tails_state[j] == 1:
                origin_index = nodeID2node[tail_index]
                entity_embeddings[tail_index] = self.entity_embeddings(tensor(origin_index).to(self.device))   
            else:
                # Equation 6
                rel_embedding[j] = rel_embedding[j].unsqueeze(0).to(self.device)
                rj = self.path_cell(r0,rel_embedding[j])

                # Equation 4
                obj_embedding = self.obj_linear(rj)
                obj_embedding = self.obj_tanh(obj_embedding)

                entity_embeddings[tail_index] = obj_embedding.squeeze()

                #id2embedding[int(obj_id[j])] = obj_embedding.squeeze()
        
        # print('id2embedding',id2embedding.keys())
        # print('old_ndata',old_ndata)

        # new_ndata = []
        # for id in old_ndata:
        #     reindex_id = node2nodeId[id]

        #     if reindex_id in id2embedding:
        #         new_ndata.append(id2embedding[reindex_id])
        #     else:
        #         new_ndata.append(torch.randn(768).to(self.device))

        # # for key,value in node2nodeId.items():
        # #     new_ndata.append(id2embedding[value])
        # new_ndata = torch.stack(new_ndata)

        #print(new_ndata.shape)

        #print(entity_embeddings)

        return entity_embeddings


