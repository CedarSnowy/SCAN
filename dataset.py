import pandas as pd 
import numpy as np 
import functools
import operator
from time import time
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertTokenizer, AlbertModel
import dgl
from dgl.sampling import sample_neighbors
from utils import load_pickle_file, _read_knowledge_graph_dialkg, _make_dgl_graph, _find_relation_entity_path

def attnIO_collate(batch): #取数据时进行堆叠
    dialogue_representations = []
    seed_entities = []
    subgraphs = []
    entity_paths = []
    for sample in batch:
        dialogue_representations.append(sample[0])
        seed_entities.append(sample[1])
        subgraphs.append(sample[2])
        entity_paths.append(sample[3])

    return [dialogue_representations, seed_entities, subgraphs, entity_paths]

#dataset里包含[previous_sentence, dialogue_history[:], starting_entities, kg_path]
class MultiReDataset():
    def __init__(self, opt):
        self.dataset = load_pickle_file(opt['dialog_samples'])
        
        self.kg_seen = load_pickle_file(opt['kg_seen'])
        self.kg_unseen = load_pickle_file(opt['kg_unseen'])

        self.entity2entityID_seen = load_pickle_file(opt['entity2entityID_seen'])
        self.entity2entityID_unseen = load_pickle_file(opt['entity2entityID_unseen'])

        self.relation2relationID = load_pickle_file(opt['relation2relationID'])

        self.entity_embeddings_seen = load_pickle_file(opt["entity_embeddings_seen"])
        self.entity_embeddings_unseen = load_pickle_file(opt["entity_embeddings_unseen"])
        self.relation_embeddings = load_pickle_file(opt["relation_embeddings"])
        self.self_loop_id = self.relation2relationID["self loop"]
        self.n_hop = opt['n_hop']
        self.n_max = opt['n_max']
        self.batch = opt['batch']

        # 划分数据集
        seen_percentage = opt['seen_percentage']
        train,valid,test_seen,test_unseen = 10583,1200,600,600
        self.train_seen = [0,int(train * seen_percentage)-1]
        self.train_unseen = [int(train * seen_percentage) ,10583 -1]
        self.valid_seen = [10583,10583+int(valid * seen_percentage)-1]
        self.valid_unseen = [10583+int(valid * seen_percentage),12983 - 1200]

        self.graph_seen,self.graph_unseen = self.get_graph()

    def get_graph(self):
        head_list,tail_list,rel_list = [],[],[]
        for key,value in self.kg_seen.items():
            head = self.entity2entityID_seen[key]
            for triple in value:
                rel = self.entity2entityID_seen[triple[1]]
                tail = self.relation2relationID[triple[2][:-4]]
                head_list.append(head)
                tail_list.append(tail)
                rel_list.append(rel)
        graph_seen = _make_dgl_graph(head_list,tail_list,rel_list)

        head_list,tail_list,rel_list = [],[],[]
        for key,value in self.kg_unseen.items():
            head = self.entity2entityID_unseen[key]
            for triple in value:
                rel = self.entity2entityID_unseen[triple[1]]
                tail = self.relation2relationID[triple[2][:-4]]
                head_list.append(head)
                tail_list.append(tail)
                rel_list.append(rel)
        graph_unseen = _make_dgl_graph(head_list,tail_list,rel_list)

        return graph_seen,graph_unseen

    def get_batch_data(self,start_index,state = 'train'):
        datas = []

        last_flag = 1
        for i in range(self.batch):
            samples_flag = self.dataset[start_index]
            samples,flag = samples_flag['samples'],samples_flag['flag']
            if i == 0:
                last_flag = flag
            else:
                if flag != last_flag:
                    break
                last_flag = flag

            if start_index - 1 >= self.valid_seen[0]:
                state = 'test'
             
            data = []
            for sample in samples:
                data.append(self.toTensor(sample,flag,state = state))
            
            datas.append(data)
            start_index += 1
        
        return {'flag':last_flag,'datas':datas}

    
    def toTensor(self,sample,flag,state):
        encoder_utterances = sample['utterances']
        startEntities = sample['seeds']
        paths = sample['paths']

        if flag == 1 or state == 'train':
            paths = torch.tensor([[self.entity2entityID_seen[path[0]], self.relation2relationID[path[1]], self.entity2entityID_seen[path[2]]] for path in paths])
        elif state == 'test': # 仅当test unseen
            paths = torch.tensor([[self.entity2entityID_unseen[path[0]], self.relation2relationID[path[1]], self.entity2entityID_unseen[path[2]]] for path in paths])


        entity_path = _find_relation_entity_path(paths, self_loop_id=(self.self_loop_id))
        entity_path = [path[1].item() for path in entity_path]

        dialogue_representation = torch.unsqueeze(encoder_utterances,0)

        if flag == 1 or state == 'train':
            seed_entities = [self.entity2entityID_seen[entity] for entity in startEntities]
            source_entities = [self.entity2entityID_seen[entity] for entity in startEntities]
        elif flag ==0 and state == 'test':
            seed_entities = [self.entity2entityID_unseen[entity] for entity in startEntities]
            source_entities = [self.entity2entityID_unseen[entity] for entity in startEntities]   

        head_entities = []
        tail_entities = []
        edge_relations = []

        if flag == 1 and state == 'train': # AttnIO
            for _ in range(self.n_hop):
                subgraph = sample_neighbors(g=(self.graph_seen), nodes=source_entities, fanout=(self.n_max), edge_dir='out')
                edges = subgraph.edges()
                head_entities.extend(edges[0].tolist())
                tail_entities.extend(edges[1].tolist())
                edge_relations.extend(subgraph.edata['edge_type'].tolist())
                source_entities = list(set(edges[1].tolist()))
        elif flag == 0 and state =='train': # Earl
            for _ in range(1):
                subgraph = sample_neighbors(g=(self.graph_seen), nodes=source_entities, fanout=(self.n_max), edge_dir='out')
                edges = subgraph.edges()
                head_entities.extend(edges[0].tolist())
                tail_entities.extend(edges[1].tolist())
                edge_relations.extend(subgraph.edata['edge_type'].tolist())
                source_entities = list(set(edges[1].tolist()))
        elif flag == 1 and state == 'test': # AttnIO
            for _ in range(self.n_hop):
                subgraph = sample_neighbors(g=(self.graph_seen), nodes=source_entities, fanout=(self.n_max), edge_dir='out')
                edges = subgraph.edges()
                head_entities.extend(edges[0].tolist())
                tail_entities.extend(edges[1].tolist())
                edge_relations.extend(subgraph.edata['edge_type'].tolist())
                source_entities = list(set(edges[1].tolist()))
        elif flag == 0 and state == 'test': # Earl
            for _ in range(1):
                subgraph = sample_neighbors(g=(self.graph_unseen), nodes=source_entities, fanout=(self.n_max), edge_dir='out')
                edges = subgraph.edges()
                head_entities.extend(edges[0].tolist())
                tail_entities.extend(edges[1].tolist())
                edge_relations.extend(subgraph.edata['edge_type'].tolist())
                source_entities = list(set(edges[1].tolist()))

        edge_presence = defaultdict(int)
        for i in range(len(head_entities)):
            label = str(head_entities[i]) + '_' + str(tail_entities[i]) + '_' + str(edge_relations[i])
            edge_presence[label] += 1

        #calculate edge present frequency
        head_entities, tail_entities, edge_relations = [], [], []

        for key, value in edge_presence.items():
            head, tail, relation = key.split('_')
            head = int(head)
            tail = int(tail)
            relation = int(relation)
            head_entities.append(head)
            tail_entities.append(tail)
            edge_relations.append(relation)

        entities = head_entities + tail_entities + entity_path
        entities = list(set(entities))
        node2nodeId = defaultdict(int)
        nodeId2node = defaultdict(int)
        idx = 0
        for entity in entities:
            node2nodeId[entity] = idx
            nodeId2node[idx] = entity
            idx += 1

        #re-encode the graph with new index
        indexed_head_entities = [node2nodeId[head_entity] for head_entity in head_entities]
        indexed_tail_entities = [node2nodeId[tail_entity] for tail_entity in tail_entities]

        entity_paths = [node2nodeId[node] for node in entity_path]
        seed_entities = [node2nodeId[node] for node in seed_entities]

        paths = [[node2nodeId[path[0].item()], path[1].item(), node2nodeId[path[2].item()]] for path in paths]

        indexed_head_entities = torch.tensor(indexed_head_entities, dtype=(torch.int64))
        indexed_tail_entities = torch.tensor(indexed_tail_entities, dtype=(torch.int64))
        edge_relations = torch.tensor(edge_relations, dtype=(torch.int64))

        subgraph = dgl.graph((torch.tensor([], dtype=(torch.int64)), torch.tensor([], dtype=(torch.int64))))
        subgraph.add_edges(indexed_head_entities, indexed_tail_entities, {'edge_type': edge_relations})
        subgraph = dgl.remove_self_loop(subgraph)
        subgraph_nodes = subgraph.nodes().tolist()
        subgraph_node_ids = [nodeId2node[node] for node in subgraph_nodes]
        subgraph_node_ids = torch.tensor(subgraph_node_ids, dtype=(torch.int64))
        subgraph.ndata['nodeId'] = subgraph_node_ids
        #ndata[nodeId]就是一个id的属性
        seed_entities = torch.tensor(seed_entities, dtype=(torch.int64))
        entity_paths = torch.tensor(entity_paths, dtype=(torch.int64))
        edges = subgraph.edges()
        heads, relations, tails = edges[0].tolist(), edge_relations.tolist(), edges[1].tolist()
        for path in paths:
            _flag = 0
            for j in range(len(heads)):
                if path[0] == heads[j] and path[1] == relations[j] and path[2] == tails[j]:
                    _flag = 1

            if not _flag:
                subgraph.add_edges(u=(torch.tensor([path[0]])), v=(torch.tensor([path[2]])), data={'edge_type': torch.tensor([path[1]])})

        if flag == 1:
            return [dialogue_representation, seed_entities, subgraph, entity_paths]
        else:
            return [dialogue_representation, seed_entities, subgraph, entity_paths, sample['MASK embedding']]



