import torch

### Device
device = 'cpu'

### Setting device
def set_gpu(gpu_id):
    if torch.cuda.is_available():
        global device
        device = 'cuda:' + str(gpu_id)

### Batch size
batch_size = 32

### GraphDataset parameters
graph_dataset = {'dist_thr': 12,
                 'max_size': 30,
                 'x_type': torch.FloatTensor,
                 'bb_type': torch.LongTensor,
                 'adj_type': torch.LongTensor,
                 'edge_type': torch.FloatTensor,
                 'mode': 'sparse', # should be sparse for GraphVAE, GraphVAE_Seq, GraphDAE
                 'weighted': False,
                 'probabilistic': False, # should be True for ProGAN
                 'load': False}

### RNNDataset_Feat parameter
rrn_dataset = {'dist_thr': 12,
               'aa_dim': 20,
               'max_size': 30,
               'ignore_idx': -100,
               'load': False}

### Testing dataset
test_dataset = {'dist_thr': 12,
                'max_size': 30,
                'x_type': torch.FloatTensor,
                'bb_type': torch.LongTensor,
                'adj_type': torch.LongTensor,
                'edge_type': torch.FloatTensor,
                'mode': 'dense',
                'weighted': False,
                'probabilistic': False,
                'load': False}

### GraphVAE parameters
graph_vae_params = {'node_dim': 10,
                    'edge_dim': 2,
                    'hidden_dim': 32,
                    'latent_dim': 8,
                    'mlp_dims': [32,64,128,128,64,32],
                    'dropout': 0.1,
                    'x_class_dim': 2,
                    'edge_class_dim': 3,
                    'max_size': 30,
                    'aa_dim': 20,
                    'ss_dim': 7,
                    'ignore_idx': -100,
                    'weight_init': 5e-5,
                    'device': device}
graph_vae_train = {'n_epochs': 100,
                   'lr': 1e-5,
                   'l_kld': 1e-5,
                   'checkpoint': 10,
                   'verbose': True}
graph_vae_eval = {'max_num_nodes': 20}

### GraphVAE_Seq parameters
graph_vae_seq_params = {'node_dim': 10,
                        'edge_dim': 2,
                        'hidden_dim': 32,
                        'latent_dim': 8,
                        'mlp_dims': [32,64,128,128,64,32],
                        'dropout': 0.1,
                        'x_class_dim': 2,
                        'aa_dim': 20,
                        'ss_dim': 7,
                        'ignore_idx': -100,
                        'weight_init': 5e-5,
                        'device': device}
graph_vae_seq_train = {'n_epochs': 1,
                       'lr': 1e-5,
                       'l_kld': 1e-4,
                       'checkpoint': 10,
                       'verbose': True}
graph_vae_seq_eval = {'max_num_nodes': 20}

### GraphDAE parameters
graph_dae_params = {'node_dim': 10,
                    'hidden_dim': 32,
                    'latent_dim': 8,
                    'mlp_dims': [32,64,128,128,64,32],
                    'dropout': 0.1,
                    'x_class_dim': 2,
                    'edge_class_dim': 3,
                    'max_size': 30,
                    'aa_dim': 20,
                    'ss_dim': 7,
                    'atom_dim': 5,
                    'ignore_idx': -100,
                    'weight_init': 5e-5,
                    'device': device}
graph_dae_train = {'n_epochs': 1000,
                   'lr': 1e-3,
                   'l_kld': 1e-4,
                   'checkpoint': 500,
                   'verbose': True}
graph_dae_eval = {'max_num_nodes': 20}

### ProGAN parameters
pro_gan_params = {'max_num_nodes': 30,
                  'node_dim': 10,
                  'edge_dim': 2,
                  'z_dim': 8,
                  'conv_out_dim': 16,
                  'agg_dim': 8,
                  'g_mlp_dims': [32,64,128,128,64,32],
                  'd_mlp_dims': [32,64,128,128,64,32],
                  'dropout': 0.1,
                  'device': device}
pro_gan_train = {'n_epochs': 1000,
                 'n_critic': 5,
                 'w_clip': 0.01,
                 'l_wrl': 0.6,
                 'lr_g': 5e-5,
                 'lr_d': 5e-5,
                 'lr_r': 5e-5,
                 'checkpoint': 500,
                 'verbose': True}
pro_gan_eval = {'test_batch_size': 1,
                'aa_min': 1,
                'aa_max': 20,
                'ss_min': 1,
                'ss_max': 7,
                'verbose': False}

### GraphRNN_A parameters
graph_rnn_a_params = {'max_prev_node': 30,
                      't_hidden_dim': 64,
                      'o_hidden_dim': 16,
                      't_embed_dim': 32,
                      'o_embed_dim': 8,
                      'num_layers': 4,
                      'node_dim': 5,
                      'edge_dim': 3,
                      'dropout': 0,
                      'device': device}
graph_rnn_a_train = {'num_epochs': 3000,
                     'lr_trans': 3e-3,
                     'lr_out_x': 3e-3,
                     'lr_out_edge': 3e-3,
                     'checkpoint': 500,
                     'milestones': [400,1000],
                     'decay': 0.3,
                     'verbose': True}
graph_rnn_a_eval = {'max_num_nodes': 20,
                    'test_batch_size': 1,
                    'aa_min': 1,
                    'aa_max': 20,
                    'ss_min': 1,
                    'ss_max': 7}

### GraphRNN_G parameters
graph_rnn_g_params = {'max_prev_node': 30,
                      'node_dim': 10,
                      'edge_dim': 2,
                      'hidden_dim': 64,
                      'g_latent_dim': 8,
                      'embed_dim': 16,
                      'mlp_dims': [32,64,128,128,64,32],
                      'num_layers': 4,
                      'dropout': 0,
                      'class_dim': 2,
                      'aa_dim': 20,
                      'ss_dim': 7,
                      'ignore_idx': -100,
                      'device': device}
graph_rnn_g_train = {'num_epochs': 3000,
                     'lr': 3e-3,
                     'checkpoint': 500,
                     'milestones': [400,1000],
                     'decay': 0.3,
                     'verbose': True}
graph_rnn_g_eval = {'max_num_nodes': 20}

### Grid-CV
cv_best_n_to_show = 5

### GraphVAE CV
graph_vae_cv_params = {'node_dim': [10],
                       'edge_dim': [2],
                       'hidden_dim': [32],
                       'latent_dim': [8, 16],
                       'mlp_dims': [[32,64,128,128,64,32], [32,64,128,256,128,64,32]],
                       'dropout': [0.1],
                       'x_class_dim': [2],
                       'edge_class_dim': [3],
                       'max_size': [30],
                       'aa_dim': [20],
                       'ss_dim': [7],
                       'ignore_idx': [-100],
                       'weight_init': [5e-5],
                       'device': [device]}
graph_vae_cv_train = {'n_epochs': [10],
                      'lr': [1e-5],
                      'l_kld': [1e-5, 1e-6],
                      'checkpoint': [None],
                      'verbose': [False]}

### GraphVAE_Seq CV
graph_vae_seq_cv_params = {}
graph_vae_seq_cv_train = {}

### GraphDAE CV
graph_dae_cv_params = {}
graph_dae_cv_train = {}

### ProGAN CV
pro_gan_cv_params = {}
pro_gan_cv_train = {}

### GraphRNN_A CV
graph_rnn_a_cv_params = {}
graph_rnn_a_cv_train = {}

### GraphRNN_G CV
graph_rnn_g_cv_params = {}
graph_rnn_g_cv_train = {}

### ProVisAdj visualization
pro_vis_adj = {'file_name': 'generated',
               'path': './'}

### ProViz visualization
pro_viz = {'file_name': 'generated',
           'bb_idx': 2,
           'inter_idx': 1,
           'path': './'}