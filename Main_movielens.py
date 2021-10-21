import torch
import time
from DataLoader import DataLoader
from TAML4Rec import TAML4Rec
import random
import numpy as np
import os
from tqdm import tqdm

def training(model, config, model_save=True, model_file=None, device=torch.device('cpu')):
    print('training model...')
    if config['use_cuda']:
        model.cuda()
    model.train()

    print('loading train data...')
    dataset = config["dataset"]
    train_data,num_train_tasks = data_loader.load_data(dataset=dataset, state='train')
    test_data,num_test_tasks= data_loader.load_data(dataset=dataset, state='test')
    print("Training tasks: {}".format(num_train_tasks))
    print("Test tasks: {}".format(num_test_tasks))

    batch_size = config['batch_size']
    num_epoch = config['num_epoch']
    num_batch = int(num_train_tasks / batch_size)

    for epoch_i in range(num_epoch):
        loss, mae, rmse = [], [], []
        ndcg_at_1, ndcg_at_3, ndcg_at_5 = [],[],[]
        start = time.time()

        random.shuffle(train_data)
        train_task_id_all, support_x_u_all, support_x_v_all, support_y_all, query_x_u_all, query_x_v_all, query_y_all = zip(*train_data)
        for batch_i in tqdm(range(num_batch)):
            task_ids = list(train_task_id_all[batch_size * batch_i:batch_size * (batch_i + 1)])
            support_xs_u = list(support_x_u_all[batch_size * batch_i:batch_size * (batch_i + 1)])
            support_xs_v = list(support_x_v_all[batch_size * batch_i:batch_size * (batch_i + 1)])
            support_ys = list(support_y_all[batch_size * batch_i:batch_size * (batch_i + 1)])

            query_xs_u = list(query_x_u_all[batch_size * batch_i:batch_size * (batch_i + 1)])
            query_xs_v = list(query_x_v_all[batch_size * batch_i:batch_size * (batch_i + 1)])
            query_ys = list(query_y_all[batch_size * batch_i:batch_size * (batch_i + 1)])

            _loss, _mae, _rmse, _ndcg_1,_ndcg_3, _ndcg_5 = model.global_update(task_ids,support_xs_u, support_xs_v,support_ys,query_xs_u ,query_xs_v, query_ys, device)
            loss.append(_loss)
            mae.append(_mae)
            rmse.append(_rmse)
            ndcg_at_1.append(_ndcg_1)
            ndcg_at_3.append(_ndcg_3)
            ndcg_at_5.append(_ndcg_5)

        print('epoch: {}, loss: {:.6f}, cost time: {:.1f}s, mae: {:.5f}, rmse: {:.5f},  ndcg@1: {:.5f}, ndcg@3: {:.5f}, ndcg@5: {:.5f}'.
              format(epoch_i, np.mean(loss), time.time() - start, np.mean(mae), np.mean(rmse),np.mean(ndcg_at_1),np.mean(ndcg_at_3),np.mean(ndcg_at_5)))
        if epoch_i % 1 == 0:
            testing(model,test_data, device)
            model.train()


def split_testing(model,split_test_data,device='cpu'):
    print('evaluating model...')
    if config['use_cuda']:
        model.cuda()
    model.eval()

    for size_i,data_i in split_test_data.items():
        test_task_id_all, support_x_u_all, support_x_v_all, support_y_all, query_x_u_all, query_x_v_all, query_y_all = zip(*data_i)

        loss, mae, rmse = [], [], []
        ndcg_at_1, ndcg_at_3, ndcg_at_5 = [], [], []

        for i in range(len(support_x_u_all)):
            _mae, _rmse, _ndcg_1, _ndcg_3, _ndcg_5 = model.evaluation(test_task_id_all[i], support_x_u_all[i],
                                                                      support_x_v_all[i], support_y_all[i],
                                                                      query_x_u_all[i], query_x_v_all[i], query_y_all[i],
                                                                      device)
            mae.append(_mae)
            rmse.append(_rmse)
            ndcg_at_1.append(_ndcg_1)
            ndcg_at_3.append(_ndcg_3)
            ndcg_at_5.append(_ndcg_5)

        print('support_size: less {}, mae: {:.5f}, rmse: {:.5f}, ndcg@1: {:.5f}, ndcg@3: {:.5f}, ndcg@5: {:.5f}'.
              format(size_i, np.mean(mae), np.mean(rmse), np.mean(ndcg_at_1), np.mean(ndcg_at_3), np.mean(ndcg_at_5)))


def testing(model, test_data, device='cpu'):
    print('evaluating model...')
    if config['use_cuda']:
        model.cuda()
    model.eval()

    test_task_id_all, support_x_u_all, support_x_v_all, support_y_all, query_x_u_all, query_x_v_all, query_y_all = zip(*test_data)

    loss, mae, rmse = [], [], []
    ndcg_at_1,ndcg_at_3,ndcg_at_5 = [],[],[]

    for i in range(len(support_x_u_all)):  # each task
        _mae, _rmse, _ndcg_1,_ndcg_3,_ndcg_5 = model.evaluation(test_task_id_all[i],support_x_u_all[i], support_x_v_all[i], support_y_all[i], query_x_u_all[i], query_x_v_all[i], query_y_all[i], device)
        mae.append(_mae)
        rmse.append(_rmse)
        ndcg_at_1.append(_ndcg_1)
        ndcg_at_3.append(_ndcg_3)
        ndcg_at_5.append(_ndcg_5)

    print('mae: {:.5f}, rmse: {:.5f}, ndcg@1: {:.5f}, ndcg@3: {:.5f}, ndcg@5: {:.5f}'.
          format(np.mean(mae), np.mean(rmse), np.mean(ndcg_at_1),np.mean(ndcg_at_3),np.mean(ndcg_at_5)))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = "movielens"

    if dataset == "movielens":
        from Configurations_Movielens import Movielens_config as config

    print("Start time:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    print("Configurations：")
    print(config)

    Input_dir = "dataset/{}/Final_20".format(dataset)
    data_loader = DataLoader(Input_dir,config)


    model_name = "TAML4Rec"
    model = TAML4Rec(config, model_name)
    model_filename = "result/{}/hml.pkl".format(dataset)

    print('--------------- {} ---------------'.format(model_name))

    cuda_or_cpu = torch.device("cuda" if config['use_cuda'] else "cpu")
    training(model, config, model_save = True, model_file=model_filename, device=cuda_or_cpu)
