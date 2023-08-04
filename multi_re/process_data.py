import os
import argparse
from collections import defaultdict
import pandas as pd
import json
from ast import literal_eval
import re
import unicodedata
import dill as pickle
import random
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
import torch.nn as nn

import torch
from transformers import AlbertTokenizer, AlbertModel
from sentence_transformers import SentenceTransformer
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2", cache_dir="/root/autodl-tmp/cache")
model = AlbertModel.from_pretrained("albert-base-v2", cache_dir="/root/autodl-tmp/cache")
encoder = AlbertModel.from_pretrained('albert-base-v2',cache_dir="/root/autodl-tmp/cache")

def get_albert_word_representations(sentence_list):
    sentence = ''
    for sen in sentence_list:
        sentence += sen + ' '
    sentence_tokenized = tokenizer.tokenize(sentence)  # 对句子进行分词

    last_sentence = sentence_list[-1]
    mask_count = last_sentence.count("[MASK]")

    mask_indices = [index for index, element in enumerate(sentence_tokenized) if element == '[MASK]'][-mask_count:]

    mask_indices = mask_indices[-mask_count:]

    sentence_ids = tokenizer.convert_tokens_to_ids(sentence_tokenized)  # 将分词转换为对应的词id

    # 添加特殊标记和生成批处理张量
    input_ids = [tokenizer.cls_token_id] + sentence_ids + [tokenizer.sep_token_id]
    inputs = torch.tensor(input_ids).unsqueeze(0)  # 添加批处理维度

    with torch.no_grad():
        outputs = model(inputs)  # 获取ALBERT模型的输出
    

    word_embeddings = outputs.last_hidden_state.squeeze(0)  # 获取最后一层的隐藏状态
    word_representations = word_embeddings[1:-1]  # 去除特殊标记并保留每个单词的编码
    
    return word_representations,mask_indices

# def get_albert_word_representations(sentence):
#     sentence_tokenized = tokenizer.tokenize(sentence)  # 对句子进行分词
#     sentence_ids = tokenizer.convert_tokens_to_ids(sentence_tokenized)  # 将分词转换为对应的词id

#     # # 选择要mask的单词索引
#     # word_indices_to_mask = [1, 3]  # 示例：将第2个和第4个单词进行mask

#     # # 将选择的单词替换为[MASK]标记
#     # for index in word_indices_to_mask:
#     #     input_ids[index] = tokenizer.mask_token_id


#     # 添加特殊标记和生成批处理张量
#     input_ids = [tokenizer.cls_token_id] + sentence_ids + [tokenizer.sep_token_id]
#     inputs = torch.tensor(input_ids).unsqueeze(0)  # 添加批处理维度

#     with torch.no_grad():
#         outputs = model(inputs)  # 获取ALBERT模型的输出
    

#     word_embeddings = outputs.last_hidden_state.squeeze(0)  # 获取最后一层的隐藏状态
#     word_representations = word_embeddings[1:-1]  # 去除特殊标记并保留每个单词的编码
    
#     return word_representations

def get_albert_representations(sentence):
    sentence_tokenized = tokenizer(sentence, return_tensors="pt")

    sentence_encoding = encoder(**sentence_tokenized)[0].detach().to("cpu")
    sentence_encoding = sentence_encoding[0][0]

    sentence_word_representations = get_albert_word_representations(sentence)

    return sentence_encoding, sentence_word_representations

# # 定义双向 RNN 编码器模型
# class BiRNNEncoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers):
#         super(BiRNNEncoder, self).__init__()

#         self.hidden_dim = hidden_dim
#         self.num_layers = num_layers
        
#         # 双向 RNN
#         self.encoder = nn.RNN(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
    
#     def forward(self, input_seq, seq_lengths):
#         # 打包输入序列
#         packed_seq = nn.utils.rnn.pack_padded_sequence(input_seq, seq_lengths, batch_first=True, enforce_sorted=False)
        
#         # 前向传播
#         encoder_outputs, encoder_state = self.encoder(packed_seq)
        
#         # 解包输出序列
#         encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)

#         return encoder_outputs, encoder_state


# input_size = 100
# hidden_size = 300
# num_layers = 2

# encoder = BiRNNEncoder(input_size,hidden_size,num_layers)

# class BiRNN(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(BiRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(input_size, hidden_size)
#         self.rnn = nn.RNN(hidden_size, hidden_size, bidirectional=True)

#     def forward(self, input_seq):
#         embedded = self.embedding(input_seq)
#         outputs, hidden = self.rnn(embedded.permute(1, 0, 2))
#         encoded = torch.cat((hidden[0], hidden[1]), dim=1)
#         return encoded

# # 示例用法
# input_size = 100  # 输入序列的词汇表大小
# hidden_size = 128  # 隐层特征维度

# # 创建模型实例
# bi_rnn = BiRNN(input_size, hidden_size)

# # 定义输入句子（转换为词索引序列）
# sentence = "这是一个示例句子"
# tokenizer = torch.nn.utils.rnn.pad_sequence([torch.tensor([1, 2, 3, 4, 5])], batch_first=True)

# # 在双向RNN上进行编码
# encoded_sentence = bi_rnn(tokenizer)

# print(encoded_sentence.shape)
# print(encoded_sentence)

start_entity= ['world']

masked_sentence = ['Hello WOrld','hello world']
# masked_sentence = ' '.join(masked_sentence)


for phrase in start_entity:
    for idx,sen in enumerate(masked_sentence):
        pattern = r"\b" + re.escape(phrase) + r"\b"  # 使用正则表达式匹配整个单词
        masked_sentence[idx] = re.sub(pattern, "[MASK]", sen, flags=re.IGNORECASE)

print(masked_sentence)


encoder_output, mask_indices = get_albert_word_representations(utterance)
mask_embeddings = [encoder_output[mask] for mask in mask_indices]

print(mask_embeddings)

# 将列表中的张量堆叠起来，创建一个新的维度作为堆叠的维度
stacked_tensor = torch.stack(mask_embeddings)

# 沿着堆叠的维度求平均值
average_tensor = torch.mean(stacked_tensor, dim=0)

print(average_tensor)
