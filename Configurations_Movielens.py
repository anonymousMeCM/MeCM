Movielens_config = {
    'dataset': 'movielens',
    "Input_dir" : "dataset/movielens/Final_20",
    'use_cuda': True,
    'model_save': False,

    ### feature dims
    # item
    'num_item': 3881,
    'num_rate': 6,
    'num_genre': 25,
    'num_actor': 7978,
    'num_director': 2186,
    'num_fea_item': 5,
    # user
    'num_user': 6040,
    'num_gender': 2,
    'num_age': 7,
    'num_occupation': 21,
    'num_zipcode': 3402,
    'num_fea_user': 5,
    # rating
    'num_rating': 6,

    ### model settings
    # embedding
    'embedding_dim': 32,
    'user_embedding_dim': 32*5,  # 5 features
    'item_embedding_dim': 32*5,  # 5 features

    # recomm module
    'first_fc_hidden_dim': 64,
    'second_fc_hidden_dim': 64,

    # TaskEncoder
    'Encoder_layer_1': 64,
    'Encoder_layer_2': 32,
    'Decoder_layer_1': 64,

    # Task Memory
    "num_memory_unit": 8,
    "key_dim": 32,
    "customization_ratio": 1e-2,

    # HIN meta-path
    "num_task" : 6040,
    "num_metapath" : 6,
    "metapaths": ['UOU','UZU','UIU_low','UIU_high','UIAIU','UIDIU'],  # ["User-Occupation-User"，"User-Zipcode-User", "User-Item-User","User-Item-Actor-item-User","User-Item-Director-item-User"]
    "n_neighbor_UOU" : 100,
    "n_neighbor_UZU" : 10,
    "n_neighbor_UIU_low" : 100,
    "n_neighbor_UIU_high" : 100,
    "n_neighbor_UIAIU" : 100,
    "n_neighbor_UIDIU" : 100,

    # Training settings
    'local_steps': 2,
    'lambda_recons_loss': 1e-3,
    'memory_lr':5e-3,
    'global_lr': 5e-4,
    'local_lr': 5e-3,
    'AE_lr': 5e-3,
    'batch_size': 32,  # for each batch, the number of tasks
    'num_epoch': 20,
    'base_model': 'NCF'
}