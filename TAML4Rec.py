import torch
import numpy as np
import math
from torch.nn import functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from BaseModel import UserEmbeddingML, ItemEmbeddingML,ItemEmbeddingDB,UserEmbeddingDB,ItemEmbeddingYP,UserEmbeddingYP,NCF_RecommModule
from MetaLearner import AE_TaskEncoder,HGAN,KV_TaskMemory
from DataLoader import MetaPath_Neighbors_Loader

class Evaluation:
    def __init__(self):
        self.k = 5

    def prediction(self, real_score, pred_score):
        MAE = mean_absolute_error(real_score, pred_score)
        RMSE = math.sqrt(mean_squared_error(real_score, pred_score))
        return MAE, RMSE

    def dcg_at_k(self,scores):
        # assert scores
        return scores[0] + sum(sc / math.log(ind+1, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))

    def ranking(self, real_score, pred_score, k):

        # NDCG@K
        pred_sorted_idx = np.argsort(pred_score)[::-1][:k]
        p_s_at_k = real_score[pred_sorted_idx]
        pred_dcg = self.dcg_at_k(p_s_at_k)

        real_sorted_idx = np.argsort(real_score)[::-1][:k]
        r_s_at_k = real_score[real_sorted_idx]
        idcg = self.dcg_at_k(r_s_at_k)
        return pred_dcg/idcg

class TAML4Rec(torch.nn.Module):
    def __init__(self, config, model_name):
        super(TAML4Rec, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")
        self.model_name = model_name
        self.input_dir = config["Input_dir"]
        # Base Model
        # Embedding modules
        if self.config['base_model'] == 'NCF':
            if self.config['dataset'] == 'movielens':
                self.item_emb = ItemEmbeddingML(config)
                self.user_emb = UserEmbeddingML(config)
            elif self.config['dataset'] == 'dbook':
                self.item_emb = ItemEmbeddingDB(config)
                self.user_emb = UserEmbeddingDB(config)
            elif self.config['dataset'] == 'yelp':
                self.item_emb = ItemEmbeddingYP(config)
                self.user_emb = UserEmbeddingYP(config)

        # Recommendation modules
        if self.config['base_model'] == 'NCF':
            self.recomm_module = NCF_RecommModule(config)

        self.rec_weight_len = len(self.recomm_module.update_parameters())
        self.rec_weight_name = list(self.recomm_module.update_parameters().keys())

        print("Parameters of Base mode {}!".format(config["base_model"]))
        for name,param in self.recomm_module.update_parameters().items():
            print(name,param.shape)

        ### Meta learner
        # Task Encoder
        self.taskencoder = AE_TaskEncoder(config)

        # Task Relationship Modeling
        self.train_tasks_neighbors = MetaPath_Neighbors_Loader(self.input_dir,"train")
        self.test_tasks_neighbors = MetaPath_Neighbors_Loader(self.input_dir,"test")
        self.task_relationship = HGAN(config) # HIN-based Heterogeneous Graph Attention network

        # Global Task Memory
        self.taskmemory = KV_TaskMemory(config,self.recomm_module.update_parameters())
        self.customization_ratio = config['customization_ratio']

        # Training settting
        self.global_lr = config['global_lr']
        self.local_lr = config['local_lr']
        self.memory_lr = config['memory_lr']
        self.AE_lr = config['AE_lr']
        self.lambda_recons_loss = config['lambda_recons_loss']
        print("\nParameters for recommendation loss optimizer!")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name,param.shape)
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=self.global_lr)

        self.AE_paramters = self.taskencoder.parameters()
        print("\n Parameters for reconstruction loss optimizer!")
        for name, param in self.taskencoder.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
        self.AE_optimizer = torch.optim.Adam(self.AE_paramters, lr=self.AE_lr)

        self.cal_metrics = Evaluation()


    def global_update(self,task_ids, support_xs_u,support_xs_v, support_ys, query_xs_u,query_xs_v,query_ys, device='cpu'):

        batch_sz = len(support_xs_u)
        loss_s = []
        recons_loss_s = []
        mae_s = []
        rmse_s = []
        ndcg_at_1_s = []
        ndcg_at_3_s = []
        ndcg_at_5_s = []

        for i in range(batch_sz):  # each task in a batch
            # Local Update
            i_support_x_u = torch.tensor(support_xs_u[i])
            i_support_x_v = torch.tensor(support_xs_v[i])
            i_support_y = torch.tensor(support_ys[i])

            i_query_x_u = torch.tensor(query_xs_u[i])
            i_query_x_v = torch.tensor(query_xs_v[i])
            i_query_y = torch.tensor(query_ys[i])

            i_meta_path_neighbors = self.train_tasks_neighbors[task_ids[i]]


            _loss, _recons_loss, _mae, _rmse, _ndcg_1,_ndcg_3,_ndcg_5 = self.local_update(i_support_x_u.to(device), i_support_x_v.to(device),i_support_y.to(device)
                                                                            ,i_query_x_u.to(device), i_query_x_v.to(device), i_query_y.to(device),i_meta_path_neighbors, mode="train")
            loss_s.append(_loss)
            recons_loss_s.append(_recons_loss)
            mae_s.append(_mae)
            rmse_s.append(_rmse)
            ndcg_at_1_s.append(_ndcg_1)
            ndcg_at_3_s.append(_ndcg_3)
            ndcg_at_5_s.append(_ndcg_5)

        loss = torch.stack(loss_s).mean(0)
        recons_loss = torch.stack(recons_loss_s).mean(0)
        mae = np.mean(mae_s)
        rmse = np.mean(rmse_s)
        ndcg_at_1 = np.mean(ndcg_at_1_s)
        ndcg_at_3 = np.mean(ndcg_at_3_s)
        ndcg_at_5 = np.mean(ndcg_at_5_s)

        # global update
        self.meta_optimizer.zero_grad()
        Total_loss = loss
        Total_loss.backward()
        self.meta_optimizer.step()

        # update memory netwroks
        self.taskmemory.write_head()

        return loss.cpu().data.numpy(), mae, rmse, ndcg_at_1,ndcg_at_3,ndcg_at_5

    def local_update(self,support_set_x_u,support_set_x_v, support_set_y, query_set_x_u,query_set_x_v, query_set_y, meta_path_neighbors, mode="train",device = 'cpu'):
        rec_initial_weights = self.recomm_module.update_parameters()

        # Embeding layers
        if self.config['base_model'] == 'NCF':
            support_user_emb = self.user_emb(support_set_x_u)
            support_item_emb = self.item_emb(support_set_x_v)
            query_user_emb = self.user_emb(query_set_x_u)
            query_item_emb = self.item_emb(query_set_x_v)

        # Task-adaptive Modulation
        # Task representation learning
        _recons_loss, interaction_embs = self.taskencoder(support_user_emb, support_item_emb,support_set_y)
        task_repre = torch.mean(interaction_embs, 0)

        #  Task relationship modeling
        task_relation_repre = self.task_relationship(task_repre,meta_path_neighbors)

        #  Global Memory
        Final_task_repre = torch.cat([task_relation_repre,task_repre],0)
        memory_weights = self.taskmemory.address_head(Final_task_repre)
        initial_fast_weights = self.taskmemory.read_head(memory_weights)

        # customization
        customized_initial_weights = {}
        for para_i, para_name in enumerate(rec_initial_weights.keys()):
            customized_initial_weights[para_name] = rec_initial_weights[para_name] - self.customization_ratio * initial_fast_weights[para_i]

        # Prediction
        if self.config['base_model'] == 'NCF':
            support_set_y_pred = self.recomm_module(support_user_emb, support_item_emb ,vars_dict = customized_initial_weights)

        loss = F.mse_loss(support_set_y_pred, support_set_y.to(torch.float32))  # torch.nn.functional.mse_loss
        grad = torch.autograd.grad(loss, customized_initial_weights.values(), create_graph=True)
        stored_grad = grad

        fast_weights = {}
        for para_i,para_name in enumerate(customized_initial_weights.keys()):
            fast_weights[para_name] = customized_initial_weights[para_name] - self.local_lr * grad[para_i]

        for local_step_i in range(1,self.config["local_steps"]):
            if self.config['base_model'] == 'NCF':
                support_set_y_pred = self.recomm_module(support_user_emb, support_item_emb,
                                                        vars_dict=fast_weights)

            loss = F.mse_loss(support_set_y_pred, support_set_y.to(torch.float32))
            grad = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            for para_i,para_name in enumerate(fast_weights.keys()):
                fast_weights[para_name] = fast_weights[para_name] - self.local_lr * grad[para_i]

        if self.config['base_model'] == 'NCF':
            query_set_y_pred = self.recomm_module(query_user_emb, query_item_emb,vars_dict=fast_weights)

        query_loss = F.mse_loss(query_set_y_pred.to(torch.float32), query_set_y.to(torch.float32))

        query_y_real = query_set_y.data.cpu().numpy()
        query_y_pred = query_set_y_pred.data.cpu().numpy()

        mae, rmse = self.cal_metrics.prediction(query_y_real, query_y_pred)  # sklearn.metrics
        ndcg_1 = self.cal_metrics.ranking(query_y_real, query_y_pred, k=1)
        ndcg_3 = self.cal_metrics.ranking(query_y_real, query_y_pred, k=3)
        ndcg_5 = self.cal_metrics.ranking(query_y_real, query_y_pred, k=5)

        if mode == "train":
            #  Memory
            self.taskmemory.store_temporary_fast_weights(Final_task_repre, stored_grad, memory_weights)
        return query_loss,_recons_loss,mae, rmse, ndcg_1, ndcg_3, ndcg_5


    def evaluation(self,i_task_id, support_x_u_i, support_x_v_i, support_y_i, query_x_u_i, query_x_v_i, query_y_i, device='cpu'):
        i_support_x_u = torch.tensor(support_x_u_i)
        i_support_x_v = torch.tensor(support_x_v_i)
        i_support_y = torch.tensor(support_y_i)

        i_query_x_u = torch.tensor(query_x_u_i)
        i_query_x_v = torch.tensor(query_x_v_i)
        i_query_y = torch.tensor(query_y_i)

        i_meta_path_neighbors = self.test_tasks_neighbors[i_task_id]
        _loss, _, _mae, _rmse, _ndcg_1, _ndcg_3, _ndcg_5 = self.local_update(i_support_x_u.to(device),
                                                                             i_support_x_v.to(device),
                                                                             i_support_y.to(device),
                                                                             i_query_x_u.to(device),
                                                                             i_query_x_v.to(device),
                                                                             i_query_y.to(device),
                                                                             i_meta_path_neighbors,mode="test")
        return _mae, _rmse, _ndcg_1, _ndcg_3, _ndcg_5
