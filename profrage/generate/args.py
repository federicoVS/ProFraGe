import torch

# TODO here make checks for using CUDA

### Batch size
batch_size = 32

### GraphDataset parameters
graph_dataset = {'dist_thr': 12,
                 'max_size': 30,
                 'x_type': torch.FloatTensor,
                 'bb_type': torch.LongTensor,
                 'adj_type': torch.LongTensor,
                 'edge_type': torch.FloatTensor,
                 'mode': 'dense',
                 'weighted': False}

### RNNDataset_Feat parameter
rrn_dataset = {'dist_thr': 12,
               'aa_dim': 20,
               'max_size': 30,
               'ignore_idx': -100}

### GraphVAE optional parameters
graph_vae = {'node_dim': 10,
             'hidden_dim': 32,
             'latent_dim': 8,
             'enc_dims': [64,128,128,64],
             'x_mlp_dims': [32,64,128,128,64,32],
             'adj_mlp_dims': [32,64,128,128,64,32],
             'adj_cnn_dims': None,
             'kernel_size': 2,
             'dropout': 0.1,
             'max_size': 30,
             'aa_dim': 20,
             'ss_dim': 7,
             'atom_dim': 5,
             'l_kld': 1e-3,
             'ignore_idx': -100,
             'use_cnn': False,
             'device': 'cpu'}
graph_vae_train = {'n_epochs': 1000,
                   'lr': 1e-3,
                   'verbose': True}
graph_vae_eval = {'verbose': False}

### GraphDAE parameters
graph_dae = {'node_dim': 10,
             'hidden_dim': 32,
             'latent_dim': 8,
             'mlp_dims': [32,64,128,128,64,32],
             'cnn_dims': [40,50,40],
             'kernel_size': 2,
             'dropout': 0.1,
             'max_size': 30,
             'aa_dim': 20,
             'ss_dim': 7,
             'atom_dim': 5,
             'l_kld': 1e-3,
             'ignore_idx': -100,
             'device': 'cpu'}
graph_dae_train = {'n_epochs': 1000,
                   'lr': 1e-3,
                   'verbose': True}
graph_dae_eval = {'verbose': False}

### ProGAN parameters
pro_gan = {'max_num_nodes': 30,
           'node_dim': 10,
           'edge_dim': 2,
           'z_dim': 8,
           'conv_out_dim': 16,
           'agg_dim': 8,
           'g_mlp_dims': [32,64,128,128,64,32],
           'd_mlp_dims': [32,64,128,128,64,32],
           'dropout': 0.1,
           'device': 'cpu'}
pro_gan_train = {'n_epochs': 1000,
                 'n_critic': 5,
                 'w_clip': 0.01,
                 'l_wrl': 0.6,
                 'lr_g': 5e-5,
                 'lr_d': 5e-5,
                 'lr_r': 5e-5,
                 'verbose': True}
pro_gan_eval = {'test_batch_size': 1,
                'aa_min': 1,
                'aa_max': 20,
                'ss_min': 0,
                'ss_max': 6}

### GraphRNN parameters
graph_rnn = {'max_prev_node': 30,
             't_hidden_dim': 64,
             'o_hidden_dim': 16,
             't_embed_dim': 32,
             'o_embed_dim': 8,
             'num_layers': 4,
             'node_dim': 5,
             'edge_dim': 3,
             'dropout': 0,
             'device': 'cpu'}
graph_rnn_train = {'num_epochs': 3000,
                   'batch_size': batch_size,
                   'lr_trans': 3e-3,
                   'lr_out_x': 3e-3,
                   'lr_out_edge': 3e-3,
                   'elw': 10,
                   'milstones': [400,1000],
                   'decay': 0.3,
                   'verbose': True}
graph_rnn_eval = {'max_num_nodes': 20,
                  'test_batch_size': 1,
                  'aa_min': 1,
                  'aa_max': 20,
                  'ss_min': 0,
                  'ss_max': 6}

