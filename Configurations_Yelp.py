Yelp_config = {
    'dataset': 'yelp',
    "Input_dir": "dataset/yelp/Final_10",
    'use_cuda': True,
    'model_save': False,

    ### feature dims
    # item
    # item
    'num_item': 34259,
    'num_city': 513,
    'num_postalcode': 6133,
    'num_category': 542,
    'num_stars': 9,
    'num_fea_item': 5,
    # user
    'num_user': 51670,
    'num_fans': 412,
    'num_avgrating': 359,
    'num_fea_user': 3,
    # rating
    'num_rating': 6,

    ### model settings
    # embedding
    'embedding_dim': 32,
    'user_embedding_dim': 32 * 3,  # 3 features
    'item_embedding_dim': 32 * 5,  # 5 features

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
    "num_task": 51670,
    "num_metapath": 4,
    "metapaths": ['UIU_high', 'UIU_low', 'UICiIU', 'UICaIU'],  # ["User-Location-User", "User-Item-User_high","User-Item-User_low","User-Item-Author-item-User","User-Item-Publisher-item-User"],
    "n_neighbor_UIU_high": 100,
    "n_neighbor_UIU_low": 50,
    "n_neighbor_UIAIU": 100,
    "n_neighbor_UIPIU": 100,

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