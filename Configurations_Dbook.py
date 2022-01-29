Dbook_config = {
    'dataset': 'dbook',
    "Input_dir": "dataset/dbook/Final_10",
    'use_cuda': True,
    'model_save': False,

    ### feature dims
    # item
    'num_item': 20934,
    'num_author': 10544,
    'num_publisher': 1698,
    'num_year': 64,
    'num_fea_item': 4,
    # user
    'num_user': 10592,
    'num_location': 453,
    'num_fea_user': 2,
    # rating
    'num_rating': 6,

    ### model settings
    # embedding
    'embedding_dim': 32,
    'user_embedding_dim': 32*2,  # 2 features
    'item_embedding_dim': 32*4,  # 4 features

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
    "num_task" : 13024,
    "num_metapath" : 5,
    "metapaths": ['ULU', 'UIU_high', 'UIU_low', 'UIAIU', 'UIPIU'],  # ["User-Location-User", "User-Item-User_high","User-Item-User_low","User-Item-Author-item-User","User-Item-Publisher-item-User"]
    "n_neighbor_ULU" : 4,
    "n_neighbor_UIU_high" : 100,
    "n_neighbor_UIU_low" : 50,
    "n_neighbor_UIAIU" : 100,
    "n_neighbor_UIPIU" : 100,

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