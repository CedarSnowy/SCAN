import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import GRUCell
# 定义双向 RNN 编码器模型
class BiRNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(BiRNNEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 双向 RNN
        self.encoder = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
    
    def forward(self, input_seq, seq_lengths):
        # 打包输入序列
        packed_seq = nn.utils.rnn.pack_padded_sequence(input_seq, seq_lengths, batch_first=True, enforce_sorted=False)
        
        # 前向传播
        encoder_outputs, encoder_state = self.encoder(packed_seq)
        
        # 解包输出序列
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)

        return encoder_outputs, encoder_state

class Earl(Module):
    def __init__(self,
                 num_embed_units,
                 relation_embeddings,
                 device
                 ):
        super(Earl,self).__init__()
        #self.sub_hidden = sub_hidden # seed_entity的embedding。维度为[len(seed_entity),num_embed_units]
        # self.triple_num = triple_num    # 采样得到的知识图谱元组数量。
        # self.encoder_batch_size = encoder_batch_size
        self.num_embed_units = num_embed_units 
        self.relation_embeddings = relation_embeddings
        self.device = device

        self.path_cell = GRUCell(num_embed_units,num_embed_units)

        self.sub_linear = nn.Linear(num_embed_units,num_embed_units)   # Equation 3
        self.sub_tanh = nn.Tanh()

        self.obj_linear = nn.Linear(num_embed_units,num_embed_units)   # Equarion 4
        self.obj_tanh = nn.Tanh()


    def forward(self,graph,encoder_output,seed_entities,sample_mask,node2nodeId):
        sub_embeddings = []


        id2embedding = {}


        heads,tails = graph.edges()[0].tolist(),graph.edges()[1].tolist()

        old_ndata = graph.ndata['nodeId']

        # print(seed_entities)
        # #print(sample_mask)
        #print('seed',seed_entities)
        # print('graph.ndata',graph.ndata["nodeId"])
     
        # print('graph.edata',graph.edata)
        # print('graph.edges()',graph.edges())
        # print('dict',node2nodeId)
        # print('heads',heads,'tails',tails)

        all_rel = graph.edata['edge_type']
        # print('all_rel',all_rel)
   
        # print('sample mask',len(sample_mask))

        for i in range(len(seed_entities)):
            seed = seed_entities[i]

            head_indices = [idx for idx,num in enumerate(heads) if num == seed]
            #print('indices',head_indices)
            rels = [graph.edata['edge_type'][idx] for idx in head_indices]

            obj_id = [tails[idx] for idx in head_indices]

            #print('rels',rels)

            triple_num = len(rels)
            
            rel_embedding = [self.relation_embeddings[key] for key in rels]
           

            sub_hidden= sample_mask[i].to(self.device)

            # Equation 3
            sub_embedding = self.sub_linear(sub_hidden) 
            sub_embedding = self.sub_tanh(sub_embedding)
            sub_embedding = sub_embedding.unsqueeze(0)

            # Equation 5
            r0 = self.path_cell(encoder_output,sub_embedding)

            sub_embeddings.append(sub_embedding.squeeze())
            id2embedding[int(seed)] = sub_embedding.squeeze()

            idx_rel = 0
            for j in range(triple_num):
                # Equation 6
                rel_embedding[j] = rel_embedding[j].unsqueeze(0).to(self.device)
                rj = self.path_cell(r0,rel_embedding[j])

                # Equation 4
                obj_embedding = self.obj_linear(rj)
                obj_embedding = self.obj_tanh(obj_embedding)

                id2embedding[int(obj_id[idx_rel])] = obj_embedding.squeeze()

                #obj_embeddings[indices[idx_rel]] = obj_embedding.squeeze()
                idx_rel += 1
        
        # print('id2embedding',id2embedding.keys())
        # print('old_ndata',old_ndata)

        new_ndata = []
        for id in old_ndata:
            reindex_id = node2nodeId[id]

            if reindex_id in id2embedding:
                new_ndata.append(id2embedding[reindex_id])
            else:
                new_ndata.append(torch.randn(768).to(self.device))

        # for key,value in node2nodeId.items():
        #     new_ndata.append(id2embedding[value])
        new_ndata = torch.stack(new_ndata)

        #print(new_ndata.shape)

        return new_ndata


