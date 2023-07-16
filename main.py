import torch
from dataset import MultiReDataset
from tqdm import tqdm
from MulitRe_build import MultiReModel

if __name__ == '__main__':
    data_directory = './dataset/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt_dataset_train = {
                        "entity2entityID_seen": data_directory+"entity2entityID_seen.pkl", 
                        "entity2entityID_unseen": data_directory+"entity2entityID_unseen.pkl", 
                        "relation2relationID": data_directory+"relation2relationID.pkl",
                        "entity_embeddings_seen": data_directory+"entity_embeddings_seen.pkl",
                        "entity_embeddings_unseen": data_directory+"entity_embeddings_unseen.pkl", 
                        "relation_embeddings": data_directory+"relation_embeddings.pkl",
                        "dialog_samples": data_directory + "dialog_samples.pkl", 
                        "knowledge_graph": data_directory+"opendialkg_triples.txt",
                        'kg_seen':data_directory+'kg_seen.pkl',
                        'kg_unseen':data_directory+'kg_unseen.pkl',
                        "device": device,
                        "n_hop": 1, "n_max": -1, "max_dialogue_history": 3,'batch':1,'seen_percentage':0.8}
    
    MulitRe_dataset_train = MultiReDataset(opt_dataset_train)
    
    opt_model = {"n_entity_seen": len(MulitRe_dataset_train.entity2entityID_seen), 
                 'n_entity_unseen':len(MulitRe_dataset_train.entity2entityID_unseen), 
                 "n_relation": len(MulitRe_dataset_train.relation2relationID),
                "entity2entityId": opt_dataset_train["entity2entityId"], 
                "entity_embedding_path": opt_dataset_train["entity_embeddings"],
                "entity_embeddings": MulitRe_dataset_train.entity_embeddings_seen, 
                "relation_embeddings": MulitRe_dataset_train.relation_embeddings,
                "out_dim":80, "in_dim": 768, "batch_size":8, "device": device, "lr": 5e-4, "lr_reduction_factor":0.1, "attn_heads": 5, "beam_size": 5,
                "epoch": 20, "model_directory": "./models/", "model_name": f'MultiRe_', "clip": 5, "self_loop_id": MulitRe_dataset_train.relation2relationId["self loop"]}
    
    model = MultiReModel()
    
    
    for dialogID in tqdm(range(1,12983 + 1)):
        MulitRe_dataset_train.get_batch_data(dialogID)
