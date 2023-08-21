import torch
from torch.utils.data import Dataset,DataLoader,Subset
from utils import load_pickle_file
from utils import _read_knowledge_graph_dialkg,_make_dgl_graph,_find_entity_path,_find_relation_entity_path
from collections import defaultdict
import dgl
from dgl.sampling import sample_neighbors
from copy import deepcopy
import random
import pandas as pd
# from association_rels import predict_rel
import itertools
import time
from association_rels_exact import predict_rel

from tqdm import tqdm


def compare_lists(list1, list2):
    # 找到相同的元素
    same_elements = []
    for sublist1 in list1:
        for sublist2 in list2:
            if sublist1 == sublist2:
                same_elements.append(sublist1)
    
    # 找到list1独有的元素
    unique_elements_list1 = [sublist for sublist in list1 if sublist not in list2]

    # 找到list2独有的元素
    unique_elements_list2 = [sublist for sublist in list2 if sublist not in list1]

    return same_elements, unique_elements_list1, unique_elements_list2

class MultiReDataset(Dataset):
    def __init__(self, opt,transform):
        self.transform = transform
        self.dataset = load_pickle_file(opt['dialog_samples'])
        
        self.kg_whole = load_pickle_file(opt['kg_whole'])
        self.kg_seen = load_pickle_file(opt['kg_seen'])
        self.kg_unseen = load_pickle_file(opt['kg_unseen'])

        self.entity2entityID_seen = load_pickle_file(opt['entity2entityID_seen'])

        self.entity2entityID_unseen = load_pickle_file(opt['entity2entityID_unseen'])

        self.entity2entityID_all = load_pickle_file(opt['entity2entityID_all'])

        self.relation2relationID = load_pickle_file(opt['relation2relationID'])

        self.entity_embeddings_seen = load_pickle_file(opt["entity_embeddings_seen"])
  
        self.entity_embeddings_unseen = load_pickle_file(opt["entity_embeddings_unseen"])

        self.entity_embeddings_all = load_pickle_file(opt['entity_embeddings_all'])
        self.relation_embeddings = load_pickle_file(opt["relation_embeddings"])
        self.self_loop_id = self.relation2relationID["self loop"]
        self.n_hop = opt['n_hop']
        self.n_max = opt['n_max']
        self.batch = opt['batch']


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        process_samples = []
        if self.transform:
            dialog_samples = self.dataset[idx]

            dialogID = dialog_samples['dialogID']
            samples_flag = dialog_samples['samples_flag']

            flag = samples_flag['flag']
            samples = samples_flag['samples']
            state = samples_flag['state']

            for sample in samples:
                sample['flag'],sample['state'] = flag,state
                process_samples.append(self.transform(sample))

        return dialogID,process_samples,flag,state


class ToTensor(object):
    def __init__(self, opt):
        self.kg_whole = load_pickle_file(opt['kg_whole'])
        self.kg_seen = load_pickle_file(opt['kg_seen'])
        self.kg_unseen = load_pickle_file(opt['kg_unseen'])

        self.entity2entityID_all = load_pickle_file(opt['entity2entityID_all'])
        self.entityID2entity_all = load_pickle_file(opt['entityID2entity_all'])

        self.entity2entityID_seen = load_pickle_file(opt['entity2entityID_seen'])
        self.entity2entityID_unseen = load_pickle_file(opt['entity2entityID_unseen'])


        self.relation_embeddings = load_pickle_file(opt["relation_embeddings"])
        self.relation2relationID = load_pickle_file(opt['relation2relationID'])

        self.self_loop_id = self.relation2relationID["self loop"]
        self.n_hop = opt['n_hop']
        self.n_max = opt['n_max']
        self.batch = opt['batch']
        self.train_unseen_rate = opt['train_unseen_rate']
        self.max_edge = opt['max_edge']
        self.do_predict = opt['do_predict']

        self.graph_train, self.graph_all = self.get_graph()



        
    def get_graph(self):
        '''
        训练时，使用kg_train,整个kg_train是已知的，不包含unseen实体；随机将其中一些节点设置为unseen
        验证、测试时，使用kg_whole,其中有seen和unseen实体
        '''
        kg_whole = deepcopy(self.kg_whole)
        entity2entityID_all = deepcopy(self.entity2entityID_all)
        relation2relationID = deepcopy(self.relation2relationID)

        kg_train = deepcopy(self.kg_seen)
        entity2entityID_train = deepcopy(self.entity2entityID_seen)

        head_list,tail_list,rel_list = [],[],[]
        for key,value in tqdm(kg_train.items()):
            head = entity2entityID_train[key]
            for triple in value:
                if triple[2] not in entity2entityID_train:
                    continue
                #rel = relation2relationID[triple[1][:-4]]
                rel = relation2relationID[triple[1]]
                tail = entity2entityID_train[triple[2]]
                head_list.append(head)
                tail_list.append(tail)
                rel_list.append(rel)
            
        graph_train = _make_dgl_graph(head_list,tail_list,rel_list)


        head_list,tail_list,rel_list = [],[],[]
        for key,value in tqdm(kg_whole.items()):
            head = entity2entityID_all[key]
            for triple in value:
                if triple[2] not in entity2entityID_all.keys():
                    continue
                #rel = relation2relationID[triple[1][:-4]]
                rel = relation2relationID[triple[1]]
                tail = entity2entityID_all[triple[2]]
                head_list.append(head)
                tail_list.append(tail)
                rel_list.append(rel)
            
        graph_all = _make_dgl_graph(head_list,tail_list,rel_list)

        return graph_train,graph_all


    def __call__(self, sample):
        #print(sample)
        start = time.time()
        flag,state = sample['flag'],sample['state']
        train_unseen = sample['train-unseen'] if 'train-unseen' in sample.keys() else 0
        encoder_utterances = sample['utterances']

        sub_embedding = sample['MASK embedding'] if 'MASK embedding' in sample.keys() else []
        startEntities = sample['starts']
        paths = sample['paths']
        
        search_rels_index = sample['search rels index'] if 'search rels index' in sample.keys() else []


        extra_dialogue_representation = torch.unsqueeze(sample['extra_utterances'],0) if 'extra_utterances' in sample.keys() else None

        #print(sample)

        dialogue_representation = torch.unsqueeze(encoder_utterances,0)
        

        do_predict = self.do_predict if len(search_rels_index) else '0'

        #do_predict = 0

        if state == 'train':
            startEntity_in_seenKG = 1
            try:
                paths = torch.tensor([[self.entity2entityID_seen[path[0]], self.relation2relationID[path[1]], self.entity2entityID_seen[path[2]]] for path in paths])
                seed_entities = [self.entity2entityID_seen[entity] for entity in startEntities]
            except:
                return {}
        elif state == 'valid' or state == 'test':
            startEntity_in_seenKG = 1 if startEntities[0] in self.entity2entityID_seen.keys() else 0 # 检查起始eneity在seen还是unseen部分，如果不在字典里面，会返回0
            try:
                paths = torch.tensor([[self.entity2entityID_all[path[0]], self.relation2relationID[path[1]], self.entity2entityID_all[path[2]]] for path in paths])
                seed_entities = [self.entity2entityID_all[entity] for entity in startEntities]
            except:
                return {}

        entity_path = _find_relation_entity_path(paths, self_loop_id=(self.self_loop_id))
        entity_path = [path[1].item() for path in entity_path]

        source_entities = deepcopy(seed_entities)

        # kg中找不到起始节点，跳过该样本
        if all(element == 0 for element in seed_entities):
            return {}
    

        # ---------------- NORMAL------------------------#

        head_entities_1_jump = []
        tail_entities_1_jump = []
        edge_relations_1_jump = []

        head_entities_2_jump = []
        tail_entities_2_jump = []
        edge_relations_2_jump = []

        # for i in range(self.n_hop):
        #     if state == 'train':
        #         subgraph = sample_neighbors(g=(self.graph_train), nodes=source_entities, fanout=(self.n_max), edge_dir='out')
        #     else:
        #         subgraph = sample_neighbors(g=(self.graph_all), nodes=source_entities, fanout=(self.n_max), edge_dir='out')
        #     edges = subgraph.edges()
        #     heads = edges[0].tolist()
        #     tails = edges[1].tolist()
        #     rels = subgraph.edata['edge_type'].tolist()
            
        #     if i == 0:
        #         if do_predict == '0' or do_predict == '2':
        #             edges = subgraph.edges()
        #             head_entities_1_jump.extend(edges[0].tolist())
        #             tail_entities_1_jump.extend(edges[1].tolist())
        #             edge_relations_1_jump.extend(subgraph.edata['edge_type'].tolist())
        #             source_entities = list(set(edges[1].tolist()))
        #         elif do_predict == '1' or do_predict == '1+2':
        #             for idx in range(len(heads)):
        #                 rel_index = int(rels[idx])
        #                 if rel_index in search_rels_index:
        #                     head_entities_1_jump.append(heads[idx])
        #                     tail_entities_1_jump.append(tails[idx])
        #                     edge_relations_1_jump.append(rels[idx])
        #             source_entities = list(set(tail_entities_1_jump))
        #     elif i == 1:
        #         if do_predict == '0' or do_predict == '1':
        #             edges = subgraph.edges()
        #             head_entities_2_jump.extend(edges[0].tolist())
        #             tail_entities_2_jump.extend(edges[1].tolist())
        #             edge_relations_2_jump.extend(subgraph.edata['edge_type'].tolist())
        #         elif do_predict == '2' or do_predict == '1+2':
        #             for idx in range(len(heads)):
        #                 rel_index = int(rels[idx])
        #                 if rel_index in search_rels_index:
        #                     head_entities_2_jump.append(heads[idx])
        #                     tail_entities_2_jump.append(tails[idx])
        #                     edge_relations_2_jump.append(rels[idx])     


        # 整理采样到的边
        # edge_presence_1_jump,edge_presence_2_jump = defaultdict(int),defaultdict(int)
        # for i in range(min(self.max_edge,len(head_entities_1_jump))):
        #     label = str(head_entities_1_jump[i]) + '_' + str(tail_entities_1_jump[i]) + '_' + str(edge_relations_1_jump[i])
        #     edge_presence_1_jump[label] += 1

        # for i in range(min(self.max_edge,len(head_entities_2_jump))):
        #     label = str(head_entities_2_jump[i]) + '_' + str(tail_entities_2_jump[i]) + '_' + str(edge_relations_2_jump[i])
        #     edge_presence_2_jump[label] += 1

        # #calculate edge present frequency
        # head_entities, tail_entities, edge_relations = [], [], []


        # for key, value in edge_presence_1_jump.items():
        #     head, tail, relation = key.split('_')
        #     head = int(head)
        #     tail = int(tail)
        #     relation = int(relation)
        #     head_entities.append(head)
        #     tail_entities.append(tail)
        #     edge_relations.append(relation)

        # for key, value in edge_presence_2_jump.items():
        #     head, tail, relation = key.split('_')
        #     head = int(head)
        #     tail = int(tail)
        #     relation = int(relation)
        #     head_entities.append(head)
        #     tail_entities.append(tail)
        #     edge_relations.append(relation)

        # ----------------- NORMAL(END) -----------------#

        # ----------------- DENSE -----------------------#

        head_entities = []
        tail_entities = []
        edge_relations = []

        successors_list = [seed_entities[0]]

        successors_list = []
        if state == 'train':
            successors_list += self.graph_train.successors(seed_entities[0]).tolist()
            subgraph = dgl.node_subgraph(self.graph_train,successors_list)

        elif state == 'valid' or state == 'test':
            successors_list += self.graph_all.successors(seed_entities[0]).tolist()
            subgraph = dgl.node_subgraph(self.graph_all,successors_list)

        node_old2new,node_new2old = {},{}

        node_new = subgraph.nodes().tolist()
        node_old = subgraph.ndata['_ID'].tolist()

        for i in range(len(node_new)):
            node_old2new[node_old[i]] = node_new[i]
            node_new2old[node_new[i]] = node_old[i]

        head_node,tail_node = subgraph.edges()[0],subgraph.edges()[1]
        for i in range(min(self.max_edge,len(head_node))):
            head_entities.append(node_new2old[head_node[i].item()])
            tail_entities.append(node_new2old[tail_node[i].item()])
            edge_relations.extend(subgraph.edata['edge_type'].tolist())

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
        # ------------------- DENSE (END) -----------------#

    

        entities = head_entities + tail_entities + entity_path

        entities = list(set(entities))
        node2nodeID = {}
        nodeID2node = {}
        
        idx = 0
        for entity in entities:
            node2nodeID[entity] = idx
            nodeID2node[idx] = entity
            idx += 1


        #re-encode the graph with new index
        indexed_head_entities = [node2nodeID[head_entity] for head_entity in head_entities]
        indexed_tail_entities = [node2nodeID[tail_entity] for tail_entity in tail_entities]

        entity_paths = [node2nodeID[node] for node in entity_path]
        seed_entities = [node2nodeID[node] for node in seed_entities]

        paths = [[node2nodeID[path[0].item()], path[1].item(), node2nodeID[path[2].item()]] for path in paths]

        indexed_head_entities = torch.tensor(indexed_head_entities, dtype=(torch.int64))
        indexed_tail_entities = torch.tensor(indexed_tail_entities, dtype=(torch.int64))
        edge_relations = torch.tensor(edge_relations, dtype=(torch.int64))

        subgraph = dgl.graph((torch.tensor([], dtype=(torch.int64)), torch.tensor([], dtype=(torch.int64))))
        subgraph.add_edges(indexed_head_entities, indexed_tail_entities, {'edge_type': edge_relations})
        subgraph = dgl.remove_self_loop(subgraph)
        subgraph_nodes = subgraph.nodes().tolist()
        subgraph_node_ids = [nodeID2node[node] for node in subgraph_nodes]
        subgraph_node_ids = torch.tensor(subgraph_node_ids, dtype=(torch.int64))
        subgraph.ndata['nodeId'] = subgraph_node_ids
        #ndata[nodeId]就是一个id的属性
        seed_entities = torch.tensor(seed_entities, dtype=(torch.int64))
        entity_paths = torch.tensor(entity_paths, dtype=(torch.int64))
        edges = subgraph.edges()
        heads, relations, tails = edges[0].tolist(), edge_relations.tolist(), edges[1].tolist()

        # 如果没有采样到ground truth路径，则加入图中
        successfully_sample = 1 # 

        for path in paths:
            _flag = 0
            for j in range(len(heads)):
                if path[0] == heads[j] and path[1] == relations[j] and path[2] == tails[j]:
                    _flag = 1

            if not _flag:
                if state == 'train':
                    subgraph.add_edges(u=(torch.tensor([path[0]])), v=(torch.tensor([path[2]])), data={'edge_type': torch.tensor([path[1]])})
                else:
                    successfully_sample = 0
                    #print(edges)
                    
        return {
                'dialogue_representation': dialogue_representation,
                'extra_dialogue_representation':extra_dialogue_representation,
                'seed_entities': seed_entities,
                'subgraph': subgraph,
                'entity_paths': entity_paths,
                'sub_embedding': sub_embedding,
                'node2nodeID': node2nodeID,
                'nodeID2node': nodeID2node,
                'entity_state_1_jump': None,
                'entity_state_2_jump': None,
                'head_entities_1_jump':head_entities_1_jump,
                'head_entities_2_jump':head_entities_2_jump,
                'tail_entities_1_jump':tail_entities_1_jump,
                'tail_entities_2_jump':tail_entities_2_jump,
                'edge_relations_1_jump':edge_relations_1_jump,
                'edge_relations_2_jump':edge_relations_2_jump,
                'unseen_rel':None,
                'startEntity_in_seenKG': startEntity_in_seenKG,
                'successfully_sample':successfully_sample
            }


def MultiRe_collate(batch): #取数据时进行堆叠
    #print('collate',batch)

    batch_datas =[]

    for data in batch:
        
        dialogID,process_samples,flag,state = data[0],data[1],data[2],data[3]

        one_dialogue_representation = {}
        one_extra_dialogue_representation = {}
        one_seed_Entities = {}
        one_subgraph = {}
        one_entity_paths = {}
        one_sub_embedding = {}
        one_node2nodeID = {}
        one_nodeID2node = {}
        one_entity_state_1_jump = {}
        one_entity_state_2_jump = {}
        one_head_entities_1_jump = {}
        one_head_entities_2_jump = {}
        one_tail_entities_1_jump = {}
        one_tail_entities_2_jump = {}
        one_edge_relations_1_jump = {}
        one_edge_relations_2_jump = {}
        one_unseen_rel = {}
        one_startEntity_in_seenKG = {}
        one_successfully_sample = {}

        count = 0
        for i,sample in enumerate(process_samples):        
            if not len(sample):
                continue 
            
            one_dialogue_representation[count] = {'dialogue_representation':sample['dialogue_representation']}
            one_extra_dialogue_representation[count] = {'extra_dialogue_representation':sample['extra_dialogue_representation']}
            one_seed_Entities[count] = {'seed_entities': sample['seed_entities']}
            one_subgraph[count] = {'subgraph': sample['subgraph']}
            one_entity_paths[count] = {'entity_paths': sample['entity_paths']}
            one_sub_embedding[count] = {'sub_embedding': sample['sub_embedding']}
            one_node2nodeID[count] = {'node2nodeID': sample['node2nodeID']}
            one_nodeID2node[count] = {'nodeID2node': sample['nodeID2node']}
            one_entity_state_1_jump[count] = {'entity_state_1_jump': sample['entity_state_1_jump']}
            one_entity_state_2_jump[count] = {'entity_state_2_jump': sample['entity_state_2_jump']}
            one_head_entities_1_jump[count] = {'head_entities_1_jump': sample['head_entities_1_jump']}
            one_head_entities_2_jump[count] = {'head_entities_2_jump': sample['head_entities_2_jump']}
            one_tail_entities_1_jump[count] = {'tail_entities_1_jump': sample['tail_entities_1_jump']}
            one_tail_entities_2_jump[count] = {'tail_entities_2_jump': sample['tail_entities_2_jump']}
            one_edge_relations_1_jump[count] = {'edge_relations_1_jump': sample['edge_relations_1_jump']}
            one_edge_relations_2_jump[count] = {'edge_relations_2_jump': sample['edge_relations_2_jump']}
            one_unseen_rel[count] = {'unseen_rel':sample['unseen_rel']}
            one_startEntity_in_seenKG[count] = {'startEntity_in_seenKG': sample['startEntity_in_seenKG']}
            one_successfully_sample[count] = {'successfully_sample':sample['successfully_sample']}
            count += 1

        one_data = {
            'one_dialogue_representation': one_dialogue_representation,
            'one_extra_dialogue_representation':one_extra_dialogue_representation,
            'one_seed_Entities': one_seed_Entities,
            'one_subgraph': one_subgraph,
            'one_entity_paths': one_entity_paths,
            'one_sub_embedding': one_sub_embedding,
            'one_node2nodeID': one_node2nodeID,
            'one_nodeID2node': one_nodeID2node,
            'one_entity_state_1_jump': one_entity_state_1_jump,
            'one_entity_state_2_jump': one_entity_state_2_jump,
            'one_head_entities_1_jump': one_head_entities_1_jump,
            'one_head_entities_2_jump': one_head_entities_2_jump,
            'one_tail_entities_1_jump': one_tail_entities_1_jump,
            'one_tail_entities_2_jump': one_tail_entities_2_jump,
            'one_edge_relations_1_jump': one_edge_relations_1_jump,
            'one_edge_relations_2_jump': one_edge_relations_2_jump,
            'one_unseen_rel': one_unseen_rel,
            'one_startEntity_in_seenKG': one_startEntity_in_seenKG,
            'one_successfully_sample':one_successfully_sample,
            'flag': flag,
            'state': state,
            'sample_num':count,
        }


        batch_datas.append(one_data)
        
    return batch_datas



