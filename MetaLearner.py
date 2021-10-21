import torch
from torch.nn import functional as F
from copy import deepcopy
from torch.autograd import Variable

class AE_TaskEncoder(torch.nn.Module):
    def __init__(self,config):
        super(AE_TaskEncoder, self).__init__()
        self.num_rating = config["num_rating"]
        self.embedding_dim = config['embedding_dim']
        self.embedding_rating = torch.nn.Embedding(
            num_embeddings=self.num_rating,
            embedding_dim=self.embedding_dim
        )

        self.encoder = torch.nn.Sequential(
            # [b, 160] => [b, 64]
            torch.nn.Linear(config["user_embedding_dim"]+config["item_embedding_dim"]+self.embedding_dim, config["Encoder_layer_1"]),
            torch.nn.ReLU(),
            # [b, 64] => [b, 32]
            torch.nn.Linear(config["Encoder_layer_1"], config["Encoder_layer_2"]),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            # [b, 32] => [b, 64]
            torch.nn.Linear(config["Encoder_layer_2"], config["Decoder_layer_1"]),
            torch.nn.ReLU(),
            # [b, 64] => [b, 160]
            torch.nn.Linear( config["Decoder_layer_1"], config["user_embedding_dim"]+config["item_embedding_dim"]+self.embedding_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, support_x_u, support_x_v, support_y):
        """
        :param [b, 1, 28, 28]:
        :return [b, 1, 28, 28]:
        """
        support_y_emb = self.embedding_rating(support_y)
        original_rep = torch.cat([support_x_u,support_x_v,support_y_emb], 1)
        # encoder 
        dense_rep = self.encoder(original_rep)  # [b, 32]
        # decoder
        recons_rep = self.decoder(dense_rep)
        recons_loss = F.mse_loss(recons_rep.to(torch.float32), original_rep.to(torch.float32))
        return recons_loss, dense_rep

class HGAN(torch.nn.Module):
    def __init__(self,config):
        super(HGAN, self).__init__()
        self.embedding_dim = config['embedding_dim']
        self.num_task = config["num_task"]
        self.meta_paths = config["metapaths"]
        self.num_metapath = config["num_metapath"]
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")

        # Task ID Embedding:
        self.embedding_taskId = torch.nn.Embedding(
            num_embeddings=self.num_task + 1 ,
            embedding_dim=self.embedding_dim
        )

        # attention
        self.Nodel_level_vars =  torch.nn.ParameterDict()
        for path_i in self.meta_paths:
            w = torch.nn.Parameter(torch.ones([1, self.embedding_dim * 2]))
            torch.nn.init.xavier_normal_(w)
            self.Nodel_level_vars[path_i]= w

        self.Semantic_level_vars = torch.nn.ParameterDict()
        w1 = torch.nn.Parameter(torch.ones([self.embedding_dim, self.embedding_dim]))
        torch.nn.init.xavier_normal_(w1)
        self.Semantic_level_vars['semantic_att_w1'] = w1

        w2 = torch.nn.Parameter(torch.ones([self.embedding_dim, self.embedding_dim]))
        torch.nn.init.xavier_normal_(w2)
        self.Semantic_level_vars['semantic_att_w2'] = w2
        self.Semantic_level_vars['semantic_att_b'] = torch.nn.Parameter(torch.zeros(self.embedding_dim))

        self.Semantic_level_vars['semantic_att_v'] = torch.nn.Parameter(torch.zeros(self.embedding_dim,1))

    def forward(self, task_repre, meta_path_neighbors):
        meta_path_repre_list = []
        for path_i in self.meta_paths:
            path_i_neighbors = meta_path_neighbors[path_i]
            path_i_neighbors_idx = torch.tensor(path_i_neighbors).long().to(self.device)
            path_i_neighbors_emb = self.embedding_taskId(path_i_neighbors_idx)
            # print("path_i",path_i_neighbors_emb.shape)

            n_neighbors = path_i_neighbors_emb.size()[0]
            task_repres = task_repre.view(1,-1).repeat(n_neighbors,1)

            # node_level attention
            key_query = torch.cat((task_repres, path_i_neighbors_emb), 1)
            # print("path_i_2", key_query.shape)
            node_att_weights = F.softmax(torch.sigmoid(F.linear(key_query,self.Nodel_level_vars[path_i])),dim = 0)
            # print("path_i_att",node_att_weights.shape)
            path_i_repre = torch.sum(path_i_neighbors_emb * node_att_weights,dim = 0)
            # print("path_i_final_repre", path_i_repre.shape)
            meta_path_repre_list.append(path_i_repre)

        # semantic_level attention
        all_paths_repre = torch.stack(meta_path_repre_list,dim=0)
        # print("semantic_repre",all_paths_repre.shape)
        semantic_keys = task_repre.view(1, -1).repeat(self.num_metapath, 1)
        semantic_logits = torch.matmul(torch.tanh(torch.matmul(semantic_keys, self.Semantic_level_vars['semantic_att_w1']) +  torch.matmul(all_paths_repre, self.Semantic_level_vars['semantic_att_w2']) + self.Semantic_level_vars['semantic_att_b']),self.Semantic_level_vars['semantic_att_v'])
        semantic_att_weights =  F.softmax(semantic_logits,dim=0)
        neighbor_repre = torch.sum(all_paths_repre * semantic_att_weights,dim = 0)
        # print("neighbor_repre",all_paths_repre.shape)
        return neighbor_repre

class KV_TaskMemory(torch.nn.Module):
    def __init__(self,config,base_model_paras):
        super(KV_TaskMemory, self).__init__()
        self.num_memory_unit = config["num_memory_unit"]
        self.key_dim = config["key_dim"]
        self.use_cuda = config["use_cuda"]
        self.write_ratio = config["memory_lr"]

        self.K_memory = torch.randn(self.num_memory_unit, self.key_dim).normal_()  # [K，task_emb_dim]
        if self.use_cuda:
            self.K_memory = self.K_memory.cuda()

        self.V_memory = []
        for mem_i in range(self.num_memory_unit):
            para_list = []
            for para_i in base_model_paras.values():
                if self.use_cuda:
                    para_list.append(torch.ones_like(para_i).normal_().cuda())
                else:
                    para_list.append(torch.ones_like(para_i).normal_())
            self.V_memory.append(para_list)
        # print("requires_grad: K_memory", self.K_memory.requires_grad)

        self.cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        self.batch_K_memory = torch.zeros_like(self.K_memory)  # [K，task_emb_dim]
        if self.use_cuda:
            self.batch_K_memory = self.batch_K_memory.cuda()

        self.batch_V_memory = []
        for mem_i in range(self.num_memory_unit):
            para_list = []
            for para_i in base_model_paras.values():
                if self.use_cuda:
                    para_list.append(torch.zeros_like(
                        para_i).cuda())
                else:
                    para_list.append(torch.zeros_like(para_i))
            self.batch_V_memory.append(para_list)

    def address_head(self, task_emb):
        repeated_task_emb = task_emb.repeat(self.num_memory_unit,1)
        atten_weights = self.cosine_sim(repeated_task_emb,self.K_memory)
        norm_weights = torch.nn.functional.softmax(atten_weights,dim=0)
        return norm_weights

    def read_head(self, memory_weights):
        rec_params = []
        for param in self.V_memory[0]:
            rec_params.append(torch.zeros_like(param))
        for i in range(self.num_memory_unit):
            for j, memory_value_ij in enumerate(self.V_memory[i]):
                # print(type(memory_value_ij))
                rec_params[j] = rec_params[j] + memory_weights[i] * memory_value_ij
        return rec_params

    def write_head(self):
        self.K_memory = (1 - self.write_ratio) * self.K_memory + self.write_ratio * self.batch_K_memory

        for i in range(self.num_memory_unit):
            for j in range(len(self.V_memory[i])):
                self.V_memory[i][j] = (1 - self.write_ratio) * self.V_memory[i][j] + self.write_ratio * self.batch_V_memory[i][j]
        self.clear_temporary_fast_weights()

    def cosine_similarity(input1, input2):
        query_norm = torch.sqrt(torch.sum(input1 ** 2 + 0.00001, 1))  #
        doc_norm = torch.sqrt(torch.sum(input2 ** 2 + 0.00001, 1))

        prod = torch.sum(torch.mul(input1, input2), 1)
        norm_prod = torch.mul(query_norm, doc_norm)

        cos_sim_raw = torch.div(prod, norm_prod)
        return cos_sim_raw

    def store_temporary_fast_weights(self, task_rep, task_fast_weights, memory_weights):
        _weights = memory_weights.reshape(self.num_memory_unit, 1)
        task_rep = task_rep.reshape(1, -1)
        product = torch.mm(_weights, task_rep)
        self.batch_K_memory = (self.batch_K_memory + product).detach()

        for i in range(self.num_memory_unit):
            for j in range(len(task_fast_weights)):
                self.batch_V_memory[i][j] = (self.batch_V_memory[i][j] +  memory_weights[i] * task_fast_weights[j]).detach()

    def clear_temporary_fast_weights(self):
        self.batch_K_memory.fill_(0)

        for i in range(len(self.batch_V_memory)):
            for j in range(len(self.batch_V_memory[i])):
                self.batch_V_memory[i][j].fill_(0)
