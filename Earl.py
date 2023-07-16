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
                 relation_embeddings
                 ):

        #self.sub_hidden = sub_hidden # seed_entity的embedding。维度为[len(seed_entity),num_embed_units]
        # self.triple_num = triple_num    # 采样得到的知识图谱元组数量。
        # self.encoder_batch_size = encoder_batch_size
        self.num_embed_units = num_embed_units 
        self.relation_embeddings = relation_embeddings

        self.path_cell = GRUCell(num_embed_units,num_embed_units)

        self.sub_linear = nn.Linear(num_embed_units,num_embed_units)   # Equation 3
        self.sub_tanh = nn.Tanh()

        self.obj_linear = nn.Linear()   # Equarion 4
        self.obj_tanh = nn.Tanh()

    def forward(self,graph,encoder_output,sub_hidden,rel_embedding):
        rels = graph.edata["edge_type"] # tensor形态的关系编号
        triple_num = len(rels)
        
        rel_embedding = [self.relation_embeddings[key] for key in rels]

        # Equation 3
        sub_embedding = self.sub_linear(sub_hidden) #
        sub_embedding = self.sub_tanh(sub_embedding)

        # Equation 5,6
        obj_embedding = []
        r0 = self.path_cell(encoder_output,sub_hidden)
        for j in range(triple_num):
            obj_embedding.append(self.path_cell(r0,rel_embedding[j]))


        rel_embedding.view(triple_num, self.num_embed_units) # relation的embedding.二维tensor[triple_num,num_embed_units]
        obj_embedding ,state= self.path_cell(rel_embedding,sub_embedding) # r

        # Equation 4
        obj_embedding.view(triple_num,self.num_embed_units)
        obj_embedding = self.obj_linear(obj_embedding)
        obj_embedding = self.obj_tanh(obj_embedding )

        return sub_embedding,obj_embedding 


