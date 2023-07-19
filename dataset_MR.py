import torch
from torch.utils.data import Dataset,DataLoader,Subset
from utils import load_pickle_file
from utils import _read_knowledge_graph_dialkg,_make_dgl_graph,_find_entity_path,_find_relation_entity_path
from collections import defaultdict
import dgl
from dgl.sampling import sample_neighbors
from copy import deepcopy
from tqdm import tqdm

class MultiReDataset(Dataset):
    def __init__(self, opt,transform):
        self.transform = transform
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

        return dialogID,process_samples,flag


class ToTensor(object):
    def __init__(self, opt):
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

        self.graph_seen,self.graph_unseen = self.get_graph()

        
    def get_graph(self):
        kg_seen = deepcopy(self.kg_seen)
        kg_unseen = deepcopy(self.kg_unseen)
        entity2entityID_seen = deepcopy(self.entity2entityID_seen)
        entity2entityID_unseen = deepcopy(self.entity2entityID_unseen)
        relation2relationID = deepcopy(self.relation2relationID)

        head_list,tail_list,rel_list = [],[],[]
        for key,value in kg_seen.items():
            head = entity2entityID_seen[key]
            for triple in value:
                rel = relation2relationID[triple[1][:-4]]
                tail = entity2entityID_seen[triple[2]]
                head_list.append(head)
                tail_list.append(tail)
                rel_list.append(rel)
            
        graph_seen = _make_dgl_graph(head_list,tail_list,rel_list)

        head_list,tail_list,rel_list = [],[],[]
        for key,value in kg_unseen.items():
            head = entity2entityID_unseen[key]
            for triple in value:
                rel = relation2relationID[triple[1][:-4]]
                tail = entity2entityID_unseen[triple[2]]
                head_list.append(head)
                tail_list.append(tail)
                rel_list.append(rel)

        graph_unseen = _make_dgl_graph(head_list,tail_list,rel_list)

        return graph_seen,graph_unseen


    def __call__(self, sample):
        flag,state = sample['flag'],sample['state']
        encoder_utterances = sample['utterances']
        if flag == 1:
            startEntities = sample['seeds']
        else:
            startEntities = sample['origins'] if 'origins' in sample else []
        paths = sample['paths']

        # print('flag:',flag,'state',state)
        # print('startentity',startEntities)

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

        #print('seed_entitites',seed_entities)

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

        #print('head',head_entities,'tail',tail_entities,'rel',edge_relations)
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

        # 如果没有采样到ground truth路径，则加入图中
        for path in paths:
            _flag = 0
            for j in range(len(heads)):
                if path[0] == heads[j] and path[1] == relations[j] and path[2] == tails[j]:
                    _flag = 1

            if not _flag:
                subgraph.add_edges(u=(torch.tensor([path[0]])), v=(torch.tensor([path[2]])), data={'edge_type': torch.tensor([path[1]])})

        if flag == 1:
            return [dialogue_representation, seed_entities, subgraph, entity_paths,[],{}]
        else:
            return [dialogue_representation, seed_entities,subgraph, entity_paths, sample['MASK embedding'],node2nodeId]
        

def MultiRe_collate(batch): #取数据时进行堆叠
    #print('collate',batch)
    dialogue_representation=[]
    seed_Entities=[]
    subgraph=[]
    entity_paths=[]
    mask_embedding=[]
    node2nodeID=[]
    flags = []

    batch_datas =[]

    for data in batch:
        
        dialogID,process_samples,flag = data[0],data[1],data[2]

        one_batch_data = []

        one_batch_dialogue_representation=[]
        one_batch_seed_Entities=[]
        one_batch_subgraph=[]
        one_batch_entity_paths=[]
        one_batch_mask_embedding=[]
        one_batch_node2nodeID=[]



        for sample in process_samples:         
            one_batch_dialogue_representation.append(sample[0])
            one_batch_seed_Entities.append(sample[1])
            one_batch_subgraph.append(sample[2])
            one_batch_entity_paths.append(sample[3])
            one_batch_mask_embedding.append(sample[4])
            one_batch_node2nodeID.append(sample[5])
        

        one_batch_data.append(one_batch_dialogue_representation)
        one_batch_data.append(one_batch_seed_Entities)
        one_batch_data.append(one_batch_subgraph)
        one_batch_data.append(one_batch_entity_paths)
        one_batch_data.append(one_batch_mask_embedding)
        one_batch_data.append(one_batch_node2nodeID)
        one_batch_data.append(flag)
        one_batch_data.append(len(process_samples))

        batch_datas.append(one_batch_data)
        
    return batch_datas



if __name__ == '__main__':

    batch_size = 4
    data_directory = './dataset/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt_dataset_train = {
                        "entity2entityID_seen": data_directory+"entity2entityID_seen.pkl", 
                        "entity2entityID_unseen": data_directory+"entity2entityID_unseen.pkl", 
                        "relation2relationID": data_directory+"relation2relationID.pkl",
                        "entity_embeddings_seen": data_directory+"entity_embeddings_seen.pkl",
                        "entity_embeddings_unseen": data_directory+"entity_embeddings_unseen.pkl", 
                        "relation_embeddings": data_directory+"relation_embeddings.pkl",
                        "dialog_samples": data_directory + "dialog_samples_list.pkl", 
                        "knowledge_graph": data_directory+"opendialkg_triples.txt",
                        'kg_seen':data_directory+'kg_seen.pkl',
                        'kg_unseen':data_directory+'kg_unseen.pkl',
                        "device": device,
                        "n_hop": 1, "n_max": 20, "max_dialogue_history": 3,'batch':batch_size,'seen_percentage':0.8}

