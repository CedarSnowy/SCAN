import torch
#from dataset import MultiReDataset
from tqdm import tqdm
from MulitRe_build import MultiReModel
from torch.utils.data import DataLoader,Subset
import os
from dataset_dense import MultiReDataset,ToTensor,MultiRe_collate
#from dataset_dense import MultiReDataset,ToTensor,MultiRe_collate
from torchvision import transforms


# 参数设置
public_data_directory = '../data/'
data_directory = './dataset/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 4
n_hop = 2
n_max = 100
save_path = f'./models/dense_share-subgraph/batch-{batch_size}/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# 
opt_dataset_train = {
                    'entity2entityID_all':data_directory + 'entity2entityID_all.pkl',
                    "entity2entityID_seen": data_directory+"entity2entityID_seen.pkl", 
                    "entity2entityID_unseen": data_directory+"entity2entityID_unseen.pkl", 
                    'entityID2entity_all':data_directory + 'entityID2entity_all.pkl',
                    "relation2relationID": data_directory+"relation2relationID.pkl",
                    'relationID2relation':data_directory + 'relationID2relation.pkl',
                    "entity_embeddings_seen": data_directory+"entity_embeddings_seen.pkl",
                    "entity_embeddings_unseen": data_directory+"entity_embeddings_unseen.pkl", 
                    "relation_embeddings": data_directory+"relation_embeddings.pkl",
                    "dialog_samples": data_directory+ "dialog_samples_list.pkl", 
                    "knowledge_graph": data_directory+"opendialkg_triples.txt",
                    'rules_path':public_data_directory+'rules_exact.csv',
                    'kg_whole':data_directory + 'kg_whole.pkl',
                    'kg_seen':data_directory+'kg_seen.pkl',
                    'kg_unseen':data_directory+'kg_unseen.pkl',
                    "device": device,
                    "n_hop": n_hop, "n_max":n_max, "max_dialogue_history": 3,'batch':batch_size,'seen_percentage':0.95,'train_unseen_rate':0.95,'max_edge':10000,'prefix_size':1}

data = MultiReDataset(opt=opt_dataset_train,transform=transforms.Compose([ToTensor(opt_dataset_train)]))

opt_model = {"n_entity_seen": len(data.entity2entityID_seen), 
                'n_entity_unseen':len(data.entity2entityID_unseen), 
                "n_relation": len(data.relation2relationID),
            "entity_embeddings_seen": data.entity_embeddings_seen, 
            "relation_embeddings": data.relation_embeddings,
            "out_dim":80, "in_dim": 768, "batch_size":batch_size, "device": device, "lr": 5e-4, "lr_reduction_factor":0.1, "attn_heads": 5, "beam_size": 5,
            "epoch": 20, "n_hop": n_hop, "n_max": n_max,"model_directory":save_path, "model_name": f'MultiRe', "clip": 5, "self_loop_id": data.relation2relationID["self loop"],'max_edge':10000}

model = MultiReModel(opt_model)

if __name__ == '__main__':
    # 定义划分的索引范围
    train_indices = list(range(10583))
    valid_indices = list(range(10583, 10583+1200))

    test_indices = list(range(len(data)-1200, len(data)))

    # 使用 Subset 创建划分的子数据集
    train_data = Subset(data, train_indices)
    valid_data = Subset(data, valid_indices)
    test_data = Subset(data, test_indices)

    # 使用Dataloader加载
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,collate_fn=MultiRe_collate)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size,collate_fn=MultiRe_collate)

    model.train_model(train_dataloader,valid_dataloader)
