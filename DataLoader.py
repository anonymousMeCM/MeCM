import pickle
import torch
class DataLoader:
    def __init__(self, input_dir, config):
        self.input_dir = input_dir
        self.config = config

    def load_data(self, dataset, state):
        total_data = pickle.load(open(self.input_dir+"/meta_{}_tasks.pkl".format(state), "rb"))

        task_id_all, support_x_all, support_y_all, query_x_all, query_y_all = zip(*total_data)
        task_id_list = []
        support_x_u_tensor = []
        support_x_v_tensor = []
        support_y_tensor = []
        query_x_u_tensor = []
        query_x_v_tensor = []
        query_y_tensor = []

        num_tasks = len(support_x_all)

        # num_tasks = 320
        for i in range(num_tasks):

            support_x_u, support_x_v = zip(*support_x_all[i])

            query_x_u, query_x_v = zip(*query_x_all[i])

            task_id_list.append(task_id_all[i])
            support_x_u_tensor.append(support_x_u)
            support_x_v_tensor.append(support_x_v)
            support_y_tensor.append(support_y_all[i])

            query_x_u_tensor.append(query_x_u)
            query_x_v_tensor.append(query_x_v)
            query_y_tensor.append(query_y_all[i])
        total_data_tensor = list(zip(task_id_list,support_x_u_tensor,support_x_v_tensor,support_y_tensor,query_x_u_tensor,query_x_v_tensor,query_y_tensor))
        return total_data_tensor, num_tasks


    def split_test_data(self,dataset):
        total_data = pickle.load(open(self.input_dir + "/split_test/split_meta_test_tasks.pkl", "rb"))
        total_data_dict = {}
        for size_i in total_data.keys():
            task_id_all, support_x_all, support_y_all, query_x_all, query_y_all = zip(*total_data[size_i])
            task_id_list = []
            support_x_u_tensor = []
            support_x_v_tensor = []
            support_y_tensor = []
            query_x_u_tensor = []
            query_x_v_tensor = []
            query_y_tensor = []

            num_tasks = len(support_x_all)
            print("support_size {}:{}".format(size_i,num_tasks))
            # num_tasks = 320
            for i in range(num_tasks):
                support_x_u, support_x_v = zip(*support_x_all[i])

                query_x_u, query_x_v = zip(*query_x_all[i])

                task_id_list.append(task_id_all[i])
                support_x_u_tensor.append(support_x_u)
                support_x_v_tensor.append(support_x_v)
                support_y_tensor.append(support_y_all[i])

                query_x_u_tensor.append(query_x_u)
                query_x_v_tensor.append(query_x_v)
                query_y_tensor.append(query_y_all[i])

            total_data_dict[size_i] = list(zip(task_id_list, support_x_u_tensor, support_x_v_tensor, support_y_tensor, query_x_u_tensor,
                    query_x_v_tensor, query_y_tensor))
        return total_data_dict

def MetaPath_Neighbors_Loader(input_dir,state):
    total_data = pickle.load(open(input_dir + "/{}_tasks_MP_neighbors.pkl".format(state), "rb"))
    return total_data