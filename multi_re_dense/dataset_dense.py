import torch
from torch.utils.data import Dataset,DataLoader,Subset
from utils import load_pickle_file
from utils import _read_knowledge_graph_dialkg,_make_dgl_graph,_find_entity_path,_find_relation_entity_path
from collections import defaultdict
import dgl
from dgl.sampling import sample_neighbors
from copy import deepcopy
import random
from tqdm import tqdm

random.seed(123)

class MultiReDataset(Dataset):
    def __init__(self, opt,transform):
        self.transform = transform
        self.dataset = load_pickle_file(opt['dialog_samples'])
        
        self.kg_whole = load_pickle_file(opt['kg_whole'])
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
                rel = relation2relationID[triple[1][:-4]]
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
                rel = relation2relationID[triple[1][:-4]]
                tail = entity2entityID_all[triple[2]]
                head_list.append(head)
                tail_list.append(tail)
                rel_list.append(rel)
            
        graph_all = _make_dgl_graph(head_list,tail_list,rel_list)

        return graph_train,graph_all


    def __call__(self, sample):
        #print(sample)
        flag,state = sample['flag'],sample['state']
        train_unseen = sample['train-unseen'] if 'train-unseen' in sample.keys() else 0
        encoder_utterances = sample['utterances']
        sub_embedding = sample['MASK embedding'] if train_unseen else []
        startEntities = sample['starts']

        paths = sample['paths']

        dialogue_representation = torch.unsqueeze(encoder_utterances,0)
        #print(paths)

        if state == 'train':
            startEntity_in_seenKG = 1
            try:
                paths = torch.tensor([[self.entity2entityID_seen[path[0]], self.relation2relationID[path[1]], self.entity2entityID_seen[path[2]]] for path in paths])
                seed_entities = [self.entity2entityID_seen[entity] for entity in startEntities]
            except:
                return [[] for _ in range(100)]
        elif state == 'valid' or state == 'test':
            startEntity_in_seenKG = 1 if startEntities[0] in self.entity2entityID_seen.keys() else 0 # 检查起始eneity在seen还是unseen部分，如果不在字典里面，会返回0
            try:
                paths = torch.tensor([[self.entity2entityID_all[path[0]], self.relation2relationID[path[1]], self.entity2entityID_all[path[2]]] for path in paths])
                seed_entities = [self.entity2entityID_all[entity] for entity in startEntities]
            except:
                return [[] for _ in range(100)]

        entity_path = _find_relation_entity_path(paths, self_loop_id=(self.self_loop_id))
        entity_path = [path[1].item() for path in entity_path]


        # kg中找不到起始节点，跳过该样本
        if all(element == 0 for element in seed_entities):
            return [[] for _ in range(100)]
        
        
        # 构建子图
        head_entities = []
        tail_entities = []
        edge_relations = []

        successors_list = [seed_entities[0]]

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
        for i in range(len(head_node)):
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

        entities = head_entities + tail_entities + entity_path
        entities = list(set(entities))
        node2nodeId = {}
        nodeId2node = {}

        entity_state = {}
        idx = 0
        for entity in entities:
            node2nodeId[entity] = idx
            nodeId2node[idx] = entity
            idx += 1

            if state == 'train':
                # 将一些object节点随机设置为unseen
                if random.random() < self.train_unseen_rate:
                    entity_state[entity] = 1
                else:
                    entity_state[entity] = 0

            elif state == 'valid' or state =='test':
                # 检查采样到的所有节点，得到每个节点是seen or unseen
                entity_name = self.entityID2entity_all[entity]
                entity_state[entity] = 1 if entity_name in self.entity2entityID_seen.keys() else 0 # 1代表在seen中找到;0代表未找到，即为unseen

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

        # 如果没有采样到ground truth路径，则加入图中
        for path in paths:
            _flag = 0
            for j in range(len(heads)):
                if path[0] == heads[j] and path[1] == relations[j] and path[2] == tails[j]:
                    _flag = 1

            if not _flag:
                subgraph.add_edges(u=(torch.tensor([path[0]])), v=(torch.tensor([path[2]])), data={'edge_type': torch.tensor([path[1]])})

        return [dialogue_representation, seed_entities,subgraph, entity_paths, sub_embedding,node2nodeId,nodeId2node,entity_state,startEntity_in_seenKG]


def MultiRe_collate(batch): #取数据时进行堆叠
    #print('collate',batch)

    batch_datas =[]

    for data in batch:
        
        dialogID,process_samples,flag,state = data[0],data[1],data[2],data[3]

        one_data = []

        one_dialogue_representation=[]
        one_seed_Entities=[]
        one_subgraph=[]
        one_entity_paths=[]
        one_mask_embedding=[]
        one_node2nodeID=[]
        one_nodeID2node = []
        one_entity_state = []
        one_startEntity_in_seenKG = []


        for sample in process_samples:         
            #print(sample)
            one_dialogue_representation.append(sample[0])
            one_seed_Entities.append(sample[1])
            one_subgraph.append(sample[2])
            one_entity_paths.append(sample[3])
            one_mask_embedding.append(sample[4])
            one_node2nodeID.append(sample[5])
            one_nodeID2node.append(sample[6])
            one_entity_state.append(sample[7])
            one_startEntity_in_seenKG.append(sample[8])

        one_data.append(one_dialogue_representation)
        one_data.append(one_seed_Entities)
        one_data.append(one_subgraph)
        one_data.append(one_entity_paths)
        one_data.append(one_mask_embedding)
        one_data.append(one_node2nodeID)
        one_data.append(one_nodeID2node)
        one_data.append(one_entity_state)
        one_data.append(one_startEntity_in_seenKG)

        one_data.append(flag)
        one_data.append(state)
        one_data.append(len(process_samples))

        batch_datas.append(one_data)
        
    return batch_datas



