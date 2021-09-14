import torch

### Device
device = 'cpu'

### Batch size
batch_size = 32

### Distance threshold for datasets
dist_thr = 12

### GraphDataset training dataset
graph_dataset = {'dist_thr': dist_thr,
                 'max_size': 30,
                 'mode': 'dense',
                 'load': True}

### GraphDataset validation dataset
graph_val_dataset = {'dist_thr': dist_thr,
                     'max_size': 30,
                     'mode': 'dense',
                     'load': True}

### RNNDataset training dataset
rrn_dataset = {'dist_thr': dist_thr,
               'max_size': 30,
               'load': True}

### RNNDataset validation dataset
rrn_val_dataset = {'dist_thr': dist_thr,
                   'max_size': 30,
                   'load': True}

### Testing dataset
test_dataset = {'dist_thr':dist_thr,
                'max_size': 30,
                'mode': 'dense',
                'load': False}

### ProVAE parameters
pro_vae_params = {'hidden_dim': 64,
                  'latent_dim': 16,
                  'gcn_dims': [4],
                  'mlp_dims': [512,512,512,512],
                  'max_size': 30,
                  'aa_dim': 20,
                  'ss_dim': 7,
                  'dropout': 0.1,
                  'device': device}
pro_vae_train = {'n_epochs': 200,
                 'lr': 2e-4,
                 'l_kld': 1e-4,
                 'betas': (0.9, 0.999),
                 'patience': 50,
                 'decay_milestones': [50,250,450],
                 'decay': 0.1,
                 'checkpoint': 500,
                 'verbose': True}
pro_vae_eval = {'verbose': True}

### ProDAAE parameters
pro_daae_params = {'hidden_dim': 32,
                   'latent_dim': 8,
                   'gcn_dims': [2],
                   'mlp_dims': [256,256],
                   'max_size': 30,
                   'aa_dim': 20,
                   'ss_dim': 7,
                   'p': 0.9,
                   'dropout': 0.1,
                   'weight_init': 5e-15,
                   'device': device}
pro_daae_train = {'n_epochs': 500,
                  'lr': 1e-7,
                  'l_adv': 10,
                  'betas': (0.9,0.999),
                  'patience': 100,
                  'decay_milestones': [150,300,450],
                  'decay': 0.1,
                  'checkpoint': 1000,
                  'verbose': True}
pro_daae_eval = {'verbose': True}

### ProGAN parameters
pro_gan_params = {'max_size': 30,
                  'z_dim': 8,
                  'gcn_dims': [],
                  'agg_dim': 4,
                  'g_mlp_dims': [512,512],
                  'dropout': 0.1,
                  'rl': False,
                  'device': device}
pro_gan_train = {'n_epochs': 8,
                 'n_critic': 5,
                 'l_wrl': 0.7,
                 'w_clamp': 0.5,
                 'lr': 5e-5,
                 'patience': 100,
                 'decay_milestones': [150,300,450],
                 'decay': 0.1,
                 'checkpoint': 500,
                 'verbose': True}
pro_gan_eval = {'aa_min': 1,
                'aa_max': 20,
                'ss_min': 1,
                'ss_max': 7,
                'verbose': True}

### ProRNN parameters
pro_rnn_params = {'max_prev_node': 30,
                  't_hidden_dim': 64,
                  'o_hidden_dim': 16,
                  't_embed_dim': 32,
                  'o_embed_dim': 8,
                  'num_layers': 4,
                  'mlp_dims': [1024,1024,1024],
                  'aa_dim': 20,
                  'ss_dim': 7,
                  'dropout': 0.1,
                  'device': device}
pro_rnn_train = {'n_epochs': 400,
                 'lr': 5e-4,
                 'betas': (0.9,0.999),
                 'decay_milestones': [150,300],
                 'decay': 0.1,
                 'patience': 800,
                 'checkpoint': 500,
                 'verbose': True}
pro_rnn_eval = {'test_batch_size': 1,
                'max_num_nodes': 20}

### Grid-CV
cv_best_n_to_show = 5

### ProVAE CV
pro_vae_cv_params = {'hidden_dim': [128],
                     'latent_dim': [32],
                     'gcn_dims': [[8,16]],
                     'mlp_dims': [[512,512,512,512]],
                     'max_size': [30],
                     'aa_dim': [20],
                     'ss_dim': [7],
                     'dropout': [0.1],
                     'device': [device]}
pro_vae_cv_train = {'n_epochs': [800],
                    'lr': [2e-4],
                    'l_kld': [1e-4],
                    'betas': [(0.9, 0.999)],
                    'patience': [50],
                    'decay_milestones': [[350,450]],
                    'decay': [0.1],
                    'checkpoint': [5000],
                    'verbose': [True]}

### ProDAAE CV
pro_daae_cv_params = {'hidden_dim': [64],
                      'latent_dim': [16],
                      'gcn_dims':[[4]],
                      'mlp_dims': [[256,256,256,256]],
                      'max_size': [30],
                      'aa_dim': [20],
                      'ss_dim': [7],
                      'p': [0.9],
                      'dropout': [0.1],
                      'weight_init': [5e-15],
                      'device': [device]}
pro_daae_cv_train = {'n_epochs': [600],
                     'lr': [1e-7],
                     'l_adv': [10],
                     'betas': [(0.9,0.999)],
                     'patience': [50],
                     'decay_milestones': [[250,450]],
                     'decay': [0.1],
                     'checkpoint': [5000],
                     'verbose': [True]}

### ProGAN CV
pro_gan_cv_params = {'max_size': [30],
                     'z_dim': [8],
                     'gcn_dims': [[2]],
                     'agg_dim': [4],
                     'g_mlp_dims': [[512,512]],
                     'dropout': [0.1],
                     'rl': [True],
                     'device': [device]}
pro_gan_cv_train = {'n_epochs': [20],
                    'n_critic': [5],
                    'l_wrl': [0.6],
                    'w_clamp': [0.5],
                    'lr': [5e-5],
                    'decay_milestones': [[150,300,450]],
                    'decay': [0.1],
                    'checkpoint': [5000],
                    'verbose': [True]}

### ProRNN CV
pro_rnn_cv_params = {'max_prev_node': [30],
                     't_hidden_dim': [64],
                     'o_hidden_dim': [16],
                     't_embed_dim': [32],
                     'o_embed_dim': [8],
                     'num_layers': [4],
                     'mlp_dims': [[1024,1024,1024]],
                     'aa_dim': [20],
                     'ss_dim': [7],
                     'dropout': [0.1],
                     'device': [device]}
pro_rnn_cv_train = {'num_epochs': [500],
                    'lr': [5e-4],
                    'betas': [(0.9,0.999)],
                    'decay_milestones': [[600,750]],
                    'decay': [0.1],
                    'checkpoint': [5000],
                    'verbose': [True]}
