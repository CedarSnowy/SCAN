import torch.nn as nn
from torch.nn import GRUCell,Linear
from utils import load_pickle_file
from dataset_rel import MultiReDataset,ToTensor,MultiRe_collate
from torch.utils.data import Subset,DataLoader
from torchvision import transforms
import torch
from torch import tensor
from tqdm import tqdm
from torch.nn import Embedding



class RelPredict(nn.Module):
    def __init__(self, opt):
        super(RelPredict, self).__init__()
        
        self.device = opt["device"]
        self.n_relation = opt["n_relation"]
        self.out_dim = opt["out_dim"]
        self.in_dim = opt["in_dim"]
        self.lr = opt["lr"]
        self.lr_reduction_factor = opt["lr_reduction_factor"]
        self.epochs = opt["epoch"]
 
        self.self_loop_id = opt["self_loop_id"]
        self.batch_size = opt['batch_size']

        self.model_directory = opt["model_directory"]
        self.model_name = opt["model_name"]
        
        self.relation_embeddings = Embedding(self.n_relation + 1, 768).to(self.device)
        self.relation_embeddings.weight.data.copy_(opt["relation_embeddings"])

        self.optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, self.parameters()), self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.hidden_size = opt['hidden_size']

        self.gru_cell = GRUCell(input_size = self.out_dim, hidden_size = self.out_dim).to(self.device)

        self.relation_linear = Linear(self.in_dim,self.out_dim).to(self.device)
        self.dialogue_linear = Linear(self.in_dim,self.out_dim).to(self.device)

        self.linear = Linear(self.out_dim*3,self.n_relation).to(self.device)


    def forward(self,last_output,current_input):

        output_1 = self.relation_linear(current_input[0])

        output_2 = self.relation_linear(current_input[1])

        output_3 = self.dialogue_linear(current_input[2])

        new_current_input = torch.zeros(3,self.out_dim).to(self.device)

        new_current_input[0] = output_1
        new_current_input[1] = output_2
        new_current_input[2] = output_3

        current_output = self.gru_cell(new_current_input,last_output)

        predict = self.linear(torch.reshape(current_output,(1,3*self.out_dim))).squeeze(0)

        return current_output,predict
    
    def process_batch(self, batch_data,train):
        batch_loss = []
        batch_loss.append(tensor(0.0).to(self.device))

        for idx,one_data in enumerate(batch_data):
            one_dialog_loss = []
            sample_num = one_data['sample_num']

            if sample_num <= 1:
                continue

            one_dialogue_representation = one_data['one_dialogue_representation']
            one_extra_dialogue_representation = one_data['one_extra_dialogue_representation']
            
            one_rels_history = one_data['one_rels_history']
            one_true_rel = one_data['one_true_rel']
    
            first_rel = one_true_rel[0]['true_rel']
            first_dialogue_rep = one_dialogue_representation[0]['dialogue_representation']

            rel_embeddings = [self.relation_embeddings(tensor(rel).to(self.device)) for rel in first_rel]
            
            init_state = torch.zeros((3,self.in_dim), requires_grad = True).to(self.device)
            last_output = torch.zeros((3,self.out_dim), requires_grad = True).to(self.device)

            if len(rel_embeddings) == 1:
                init_state[0:2] = torch.stack([rel_embeddings[0].to(self.device)] * 2)
            else:
                init_state[0:2] = torch.stack([rel_embeddings[i].to(self.device) for i in range(2)])
            
            init_state[2] = first_dialogue_rep.to(self.device)
            output_1 = self.relation_linear(init_state[0])
            output_2 = self.relation_linear(init_state[1])
            output_3 = self.dialogue_linear(init_state[2])
            last_output[0] = output_1
            last_output[1] = output_2
            last_output[2] = output_3

            for i in range(1,sample_num):
                dialogue_representation = one_dialogue_representation[i]['dialogue_representation']
                extra_dialogue_representation = one_extra_dialogue_representation[i]['extra_dialogue_representation']
                rels_history = one_rels_history[i]['rels_history']

                rel_last = rels_history
                true_rel = one_true_rel[i]['true_rel']

                true_label = torch.zeros(self.n_relation).to(self.device)
                
                for rel_idx in true_rel:
                    true_label[rel_idx] = 1

                current_input = torch.zeros((3,self.in_dim),requires_grad = True).to(self.device)

                rel_embeddings = [self.relation_embeddings(tensor(rel).to(self.device)) for rel in rel_last]

                if len(rel_embeddings) == 1:
                    current_input[0:2] = torch.stack([rel_embeddings[0].to(self.device)] * 2)
                else:
                    current_input[0:2] = torch.stack([rel_embeddings[i].to(self.device) for i in range(2)])


                current_input[2] = extra_dialogue_representation.to(self.device) if extra_dialogue_representation is not None else dialogue_representation.to(self.device)

                current_output,predict = self(last_output,current_input)
                
                one_dialog_loss.append(self.loss_fn(predict,true_label))

                last_output = current_output.to(self.device)
                
            
            # 如果没有进行训练，则暂存一个0，后续删除
            if len(one_dialog_loss) > 0:
                one_dialog_loss = torch.stack(one_dialog_loss).sum(-1)/len(one_dialog_loss)
            else:
                one_dialog_loss = tensor(0.0).to(self.device)
            batch_loss.append(one_dialog_loss)
        
        # 找到不符合要求的数据的索引
        indices_to_remove = [i for i, loss in enumerate(batch_loss) if loss.item() == 0.0]

        # 如果全部数据都不符合要求，则保留一个值为 0 的张量
        if len(indices_to_remove) == len(batch_loss):
            indices_to_remove.pop()

        # 删除不符合要求的数据
        for index in sorted(indices_to_remove, reverse=True):
            batch_loss.pop(index)

        # 计算平均损失
        batch_loss = torch.stack(batch_loss).sum(-1)/len(batch_loss)

        if train:
            self.optimizer.zero_grad()
            if batch_loss.item() != 0.0:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                self.optimizer.step()

        return batch_loss.item()

    
    def train_model(self,train_dataloader,valid_dataloader):
        self.optimizer.zero_grad()
        ins = 0 
        for epoch in range(self.epochs):
            self.train()

            # 训练阶段
            train_loss = 0
            cnt = 0
            for batch in tqdm(train_dataloader):
                train_loss += self.process_batch(batch,True)
                cnt += 1
            train_loss /= cnt

            # 验证阶段
            dev_loss,count = 0,0

            cnt = 0
            for batch in tqdm(valid_dataloader):
                batch_loss = self.process_batch(batch,False)
                dev_loss += batch_loss
                cnt += 1

            # 全部平均
            dev_loss /= cnt
            
            self.lr_scheduler.step(dev_loss)
            print("Epoch: %d, Train Loss: %f" %(epoch, train_loss))
            print("Epoch: %d, Dev All Loss: %f" %(epoch, dev_loss))
            

            ins+=1
            if (epoch+1)%20==0:
                torch.save(self.state_dict(), f'{self.model_directory}{self.model_name}_epoch-{epoch+1}')

def test_model(opt,policy_file,test_dataloader):
    pretrain_sd = torch.load(policy_file)
    model = RelPredict(opt).to(opt["device"])
    model_sd = model.state_dict()
    model_sd.update(pretrain_sd)
    model.load_state_dict(model_sd)
    model = model.to(opt["device"])

    count = 0
    success_count_single = {1:0,3:0,5:0}
    success_count_dual = {2:0,3:0,5:0}

    for batch_data in tqdm(test_dataloader):
        for idx,one_data in enumerate(batch_data):
            sample_num = one_data['sample_num']

            if sample_num <= 1:
                continue

            one_dialogue_representation = one_data['one_dialogue_representation']
            one_extra_dialogue_representation = one_data['one_extra_dialogue_representation']
            
            one_rels_history = one_data['one_rels_history']
            one_true_rel = one_data['one_true_rel']
    
            first_rel = one_true_rel[0]['true_rel']
            first_dialogue_rep = one_dialogue_representation[0]['dialogue_representation']

            rel_embeddings = [model.relation_embeddings(tensor(rel).to(model.device)) for rel in first_rel]
            
            last_output = torch.zeros((3,model.in_feats), requires_grad = True).to(model.device)

            if len(rel_embeddings) == 1:
                last_output[0:2] = torch.stack([rel_embeddings[0].to(model.device)] * 2)
            else:
                last_output[0:2] = torch.stack([rel_embeddings[i].to(model.device) for i in range(2)])
            
            last_output[2] = first_dialogue_rep.to(model.device)

            for i in range(1,sample_num):
                dialogue_representation = one_dialogue_representation[i]['dialogue_representation']
                extra_dialogue_representation = one_extra_dialogue_representation[i]['extra_dialogue_representation']
                rels_history = one_rels_history[i]['rels_history']

                rel_last = rels_history
                true_rel = one_true_rel[i]['true_rel']

                true_label = torch.zeros(model.n_relation).to(model.device)
                
                for rel_idx in true_rel:
                    true_label[rel_idx] = 1

                current_input = torch.zeros((3,model.in_feats),requires_grad = True).to(model.device)




                rel_embeddings = [model.relation_embeddings(tensor(rel).to(model.device)) for rel in rel_last]

                if len(rel_embeddings) == 1:
                    current_input[0:2] = torch.stack([rel_embeddings[0].to(model.device)] * 2)
                else:
                    current_input[0:2] = torch.stack([rel_embeddings[i].to(model.device) for i in range(2)])

                current_input[2] = extra_dialogue_representation.to(model.device) if extra_dialogue_representation is not None else dialogue_representation.to(model.device)

                current_output,predict = model(last_output,current_input)

                last_output = current_output.to(model.device)

                count += 1

                top1_values, top1_indices = torch.topk(predict, k=1)
                top2_values, top2_indices = torch.topk(predict, k=2)
                top3_values, top3_indices = torch.topk(predict, k=3)
                top5_values, top5_indices = torch.topk(predict, k=5)
                print(true_rel,top5_indices)

                if len(true_rel) == 1:
                    K = [1, 3, 5]
                    for k in K:
                        top_k_indices = top5_indices[:k]
                        if true_rel[0] in top_k_indices:
                            success_count_single[k] += 1

                elif len(true_rel) == 2:
                    K = [2, 3, 5]
                    for k in K:
                        top_k_indices = top5_indices[:k]
                        if all(label in top_k_indices for label in true_rel):
                            success_count_dual[k] += 1

    print(success_count_dual,success_count_single)
    print(count)


if __name__ == '__main__':

    public_data_directory = '.../data/'
    data_directory = '../dataset_whole/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    save_path = './models/rel_predict/'

    opt_dataset = {
                    "relation2relationID": data_directory+"relation2relationID.pkl",
                    "relation_embeddings": data_directory+"relation_embeddings.pkl",
                    "dialog_samples": data_directory+ "dialog_samples_list.pkl", 
                    "device": device,
                    'batch':batch_size}

    data = MultiReDataset(opt=opt_dataset,transform=transforms.Compose([ToTensor(opt_dataset)]))

    
    # 设置测试数据的维度和参数
    in_dim = 768
    rel_nums = len(data.relation2relationID)
    hidden_size = 100

    opt_model = {
                "n_relation": rel_nums,
                "relation_embeddings": data.relation_embeddings,
                "out_dim":80, "in_dim": in_dim, "batch_size":batch_size, "device": device, "lr": 1e-1, "lr_reduction_factor":0.1, "attn_heads": 5, "beam_size": 5,'hidden_size':hidden_size,
                "epoch": 100,"model_directory":save_path, 
                "model_name": f'MultiRe', "clip": 5, "self_loop_id": data.relation2relationID["self loop"]}


    model = RelPredict(opt_model)

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
    test_dataloader = DataLoader(test_data,batch_size = 1,collate_fn = MultiRe_collate)

    model.train_model(train_dataloader,valid_dataloader)

    # for i in range(20,100,20):
    #     opt_model['batch_size'] = 1
    #     model_path = save_path
    #     model_name = f'MultiRe_epoch-{i}'
    #     model_path += model_name
    #     test_model(opt_model,model_path,test_dataloader)






