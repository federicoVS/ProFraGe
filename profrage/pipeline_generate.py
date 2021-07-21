import os

import itertools

import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader

from generate import args
from generate.vae import GraphVAE, GraphVAE_Seq, GraphDAE
from generate.gan import ProGAN
from generate.rnn import GraphRNN_A, GraphRNN_G
from generate.datasets import GraphDataset, RNNDataset_Feat
from generate.quality import MMD, QCP
from generate.visualize import ProViz
from utils.io import get_files, from_pdb
from utils.ProgressBar import ProgressBar

def _grid_cv(model_type, pdb_train, pdb_val, stride_dir, dataset_dir, data_type='graph', quality='qcp', verbose=False):
    # Get the training proteins
    train_pdbs, val_pdbs = get_files(pdb_train, ext='.pdb'), get_files(pdb_val, ext='.pdb')
    train_proteins, val_proteins = [], []
    for pdb in train_pdbs:
        pdb_id = os.path.basename(pdb)[:-4]
        train_proteins.append(from_pdb(pdb_id, pdb, quiet=True))
    # Get the validation proteins
    for pdb in val_pdbs:
        pdb_id = os.path.basename(pdb)[:-4]
        val_proteins.append(from_pdb(pdb_id, pdb, quiet=True))
    # Get the training dataset
    if data_type == 'graph':
        train_dataset = GraphDataset(dataset_dir, train_proteins, pdb_train, stride_dir, **args.graph_dataset)
    elif data_type == 'rnn':
        train_dataset = RNNDataset_Feat(dataset_dir, train_proteins, pdb_train, stride_dir, **args.rrn_dataset)
    # Get the validation dataset
    val_dataset = GraphDataset(dataset_dir, val_proteins, pdb_val, stride_dir, **args.test_dataset)
    # Define the loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
    # Select model and parameters set
    Cmodel, model_params, train_params, eval_params = None, None, None, None
    if model_type == 'GraphVAE':
        Cmodel = GraphVAE
        model_params = args.graph_vae_cv_params
        train_params = args.graph_vae_cv_train
        eval_params = args.graph_vae_eval
    elif model_type == 'GraphVAE_Seq':
        Cmodel = GraphVAE_Seq
        model_params = args.graph_vae_seq_cv_params
        train_params = args.graph_vae_seq_cv_train
        eval_params = args.graph_vae_seq_eval_eval
    elif model_type == 'GraphDAE':
        Cmodel = GraphDAE
        model_params = args.graph_dae_cv_params
        train_params = args.graph_dae_cv_train
        eval_params = args.graph_dae_eval
    elif model_type == 'ProGAN':
        Cmodel = ProGAN
        model_params = args.pro_gan_cv_params
        train_params = args.pro_gan_cv_train
        eval_params = args.pro_gan_eval
    elif model_type == 'GraphRNN_A':
        Cmodel = GraphRNN_A
        model_params = args.graph_rnn_a_cv_params
        train_params = args.graph_rnn_a_cv_train
        eval_params = args.graph_rnn_a_eval
    elif model_type == 'GraphRNN_G':
        Cmodel = GraphRNN_G
        model_params = args.graph_rnn_g_cv_params
        train_params = args.graph_rnn_g_cv_train
        eval_params = args.graph_rnn_g_eval
    # Prepare the model parameter configurations
    params_total_len = 1
    for mp in model_params:
        params_total_len *= len(model_params[mp])
    param_search_space = (dict(zip(model_params, x)) for x in itertools.product(*model_params.values()))
    # Prepare the training parameters configurations
    train_total_len = 1
    for tp in train_params:
        train_total_len *= len(train_params[tp])
    train_search_space = (dict(zip(train_params, x)) for x in itertools.product(*train_params.values()))
    # Best configurations list
    best_params = []
    # Progress bar settings
    progress_bar = ProgressBar(params_total_len*train_total_len)
    if verbose:
        progress_bar.start()
    # Start the grid search
    for param_config, train_config in list(itertools.product(param_search_space, train_search_space)):
        if verbose:
            progress_bar.step()
        model = Cmodel(param_config)
        model.train(train_loader, **train_config)
        prediction = model.eval(**eval_params)
        if len(prediction) == 2:
            x_pred, adj_edge_pred = prediction
        else:
            x_pred, adj_edge_pred = prediction, None
        if quality == 'qcp':
            qcp = QCP(x_pred[:,5:8])
            scores = qcp.superimpose()
        elif quality == 'mmd':
            pred_graph = (x_pred, adj_edge_pred)
            target_graphs = []
            for _, (data) in enumerate(val_loader):
                x, adj_edge = data['x'], data['adj'] + data['edge'][:,:,:1]
                x, adj_edge = x.view(x.shape[1],x.shape[2]), adj_edge.view(adj_edge.shape[1],adj_edge.shape[2],adj_edge.shape[3])
                target_graphs.append((x, adj_edge))
            mmd = MMD(pred_graph, target_graphs)
            scores = mmd.compare_graphs()
        mean, var = np.mean(scores), np.var(scores)
        best_params.append((mean, var, param_config, train_config))
    best_configs = sorted(best_params, key=lambda x: x[0])[0:args.cv_best_n_to_show]
    for best_config in best_configs:
        print(f'Average: {best_config[0]}, Variance: {best_config[1]}, \n Model Params.: {best_config[2]}, Training Params.: {best_config[3]}')

def _full(model_type, pdb_train, pdb_test, stride_dir, dataset_dir, model_dir, model_idx=0, data_type='graph', quality='qcp', train=True):
    # Get the training proteins
    train_pdbs, test_pdbs = get_files(pdb_train, ext='.pdb'), get_files(pdb_test, ext='.pdb')
    train_proteins, test_proteins = [], []
    for pdb in train_pdbs:
        pdb_id = os.path.basename(pdb)[:-4]
        train_proteins.append(from_pdb(pdb_id, pdb, quiet=True))
    # Get the validation proteins
    for pdb in test_pdbs:
        pdb_id = os.path.basename(pdb)[:-4]
        test_proteins.append(from_pdb(pdb_id, pdb, quiet=True))
    # Get the training data
    if data_type == 'graph':
        train_dataset = GraphDataset(dataset_dir, train_proteins, pdb_train, stride_dir, **args.graph_dataset)
    elif data_type == 'rnn':
        train_dataset = RNNDataset_Feat(dataset_dir, train_proteins, pdb_train, stride_dir, **args.rrn_dataset)
    # Get the test data
    test_dataset = GraphDataset(dataset_dir, test_proteins, pdb_test, stride_dir, **args.test_dataset)
    # Define the loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    # Select the model
    Cmodel, model_params, train_params, eval_params = None, None, None, None
    if model_type == 'GraphVAE':
        Cmodel = GraphVAE
        model_params = args.graph_vae_params
        train_params = args.graph_vae_train
        eval_params = args.graph_vae_eval
    elif model_type == 'GraphVAE_Seq':
        Cmodel = GraphVAE_Seq
        model_params = args.graph_vae_seq_params
        train_params = args.graph_vae_seq_train
        eval_params = args.graph_vae_seq_eval_eval
    elif model_type == 'GraphDAE':
        Cmodel = GraphDAE
        model_params = args.graph_dae_params
        train_params = args.graph_dae_train
        eval_params = args.graph_dae_eval
    elif model_type == 'ProGAN':
        Cmodel = ProGAN
        model_params = args.pro_gan_params
        train_params = args.pro_gan_train
        eval_params = args.pro_gan_eval
    elif model_type == 'GraphRNN_A':
        Cmodel = GraphRNN_A
        model_params = args.graph_rnn_a_params
        train_params = args.graph_rnn_a_train
        eval_params = args.graph_rnn_a_eval
    elif model_type == 'GraphRNN_G':
        Cmodel = GraphRNN_G
        model_params = args.graph_rnn_g_params
        train_params = args.graph_rnn_g_train
        eval_params = args.graph_rnn_g_eval
    # Define the model
    model = Cmodel(**model_params)
    # Train the model (if requested)
    if train:
        model.train(train_loader, **train_params)
        torch.save(model.state_dict(), model_dir + '/' + model_type + '_' + str(model_idx))
    else:
        model.load_state_dict(torch.load(model_dir + '/' + model_type + '_' + str(model_idx)))
    # Evaluate the model
    prediction = model.eval(**eval_params)
    if len(prediction) == 2:
        x_pred, adj_edge_pred = prediction
    else:
        x_pred, adj_edge_pred = prediction, None
    # Assess the model performance
    if quality == 'qcp':
        qcp = QCP(x_pred[:,5:8])
        scores = qcp.superimpose()
    elif quality == 'mmd':
        pred_graph = (x_pred, adj_edge_pred)
        target_graphs = []
        for _, (data) in enumerate(test_loader):
            x, adj_edge = data['x'], data['adj'] + data['edge'][:,:,:1]
            x, adj_edge = x.view(x.shape[1],x.shape[2]), adj_edge.view(adj_edge.shape[1],adj_edge.shape[2],adj_edge.shape[3])
            target_graphs.append((x, adj_edge))
        mmd = MMD(pred_graph, target_graphs)
        scores = mmd.compare_graphs()
    score_mean, score_var = np.mean(scores), np.var(scores)
    print(f'Average: {score_mean}, Variance: {score_var}')
    # Visualize the data
    proviz = ProViz(x_pred, adj=adj_edge_pred, **args.pro_viz)
    proviz.vizify()

if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Full generation pipeline.')
    arg_parser.add_argument('mode', type=str, help='The mode of the pipeline. Valid modes are [grid_cv,full].')
    arg_parser.add_argument('model', type=str, help='The model to use. Valid models are [GraphVAE,GraphVAE_Seq,GraphDAE,ProGAN,GraphRNN_A,GraphRNN_G].')
    arg_parser.add_argument('pdb_train', type=str, help='The directory holding the PDB files from the training set.')
    arg_parser.add_argument('pdb_val', type=str, help='The directory holding the PDB files from the validation set.')
    arg_parser.add_argument('pdb_test', type=str, help='The directory holding the PDB files from the test set.')
    arg_parser.add_argument('stride_dir', type=str, help='The directory holding the Stride tool.')
    arg_parser.add_argument('dataset_dir', type=str, help='The directory holding the dataset.')
    arg_parser.add_argument('model_dir', type=str, help='The directory holding the models weights.')
    arg_parser.add_argument('--model_idx', type=int, default=0, help='The index of the model instance, which is used to save its weights. The default is 0.')
    arg_parser.add_argument('--data_type', type=str, default='graph', help='The type of the dataset. Valid types are [graph,rnn]. The default is graph.')
    arg_parser.add_argument('--quality', type=str, default='qcp', help='The quality metric to use to assess the generated proteins. Valid measurements are [mmd,qcp]. The default is qcp.')
    arg_parser.add_argument('--compute_data', type=bool, default=True, help='Whether to compute the data features. The default is True.')
    arg_parser.add_argument('--train', type=bool, default=True, help='Whether to train the model. If not, then the model weights are loaded. The default is True.')
    arg_parser.add_argument('--gpu_id', type=int, default=0, help='The GPU to use (if available). The default is 0.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args_parsed = arg_parser.parse_args()
    # Set device
    args.set_gpu(args_parsed.gpu_id)
    # Choose pipeline
    if args_parsed.mode == 'grid_cv':
        _grid_cv(args_parsed.model, args_parsed.pdb_train, args_parsed.pdb_val, args_parsed.stride_dir, args_parsed.dataset_dir,
                 data_type=args_parsed.data_type, quality=args_parsed.quality, verbose=args_parsed.verbose)
    elif args_parsed.mode == 'full':
        _full(args_parsed.model, args_parsed.pdb_train, args_parsed.pdb_test, args_parsed.stride_dir, args_parsed.dataset_dir, args_parsed.model_dir,
              model_idx=args_parsed.model_idx, data_type=args_parsed.data_type, quality=args_parsed.quality, train=args_parsed.train)
    else:
        print('Wrong mode selected.')