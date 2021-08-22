import os

import itertools

import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader as GDataLoader

from generate import args
from generate.vae import ProVAE
from generate.dae import ProDAAE
from generate.gan import ProGAN
from generate.rnn import ProRNN
from generate.datasets import GraphDataset, RNNDataset_Feat
from generate.quality import MMD, QCP
from generate.reconstruct import GramReconstruction, FragmentBuilder
from utils.io import get_files, from_pdb
from utils.ProgressBar import ProgressBar

def _grid_cv(model_type, pdb_train, pdb_val, stride_dir, dataset_dir, model_dir, dataset_id=0, model_id=0, data_type='graph', data_mode='sparse', verbose=False):
    if verbose:
        print('Processing the data...')
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
    # Get the training and validation datasets
    if data_type == 'graph':
        train_dataset = GraphDataset(dataset_dir, dataset_id, 'train', train_proteins, pdb_train, stride_dir, **args.graph_dataset)
        val_dataset = GraphDataset(dataset_dir, 0, 'val', val_proteins, pdb_val, stride_dir, **args.graph_val_dataset)
    elif data_type == 'rnn':
        train_dataset = RNNDataset_Feat(dataset_dir, dataset_id, 'train', train_proteins, pdb_train, stride_dir, **args.rrn_dataset)
        val_dataset = RNNDataset_Feat(dataset_dir, 0, 'val', val_proteins, pdb_val, stride_dir, **args.rrn_val_dataset)
    if verbose:
        print(f'Training set has {len(train_dataset)} samples.')
    # Save the datasets
    train_dataset.save()
    val_dataset.save()
    # Define the loaders
    if data_mode == 'dense' or data_type == 'rnn':
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
    elif data_mode == 'sparse':
        train_loader = GDataLoader(dataset=train_dataset.get_data(), batch_size=args.batch_size, shuffle=True)
        val_loader = GDataLoader(dataset=val_dataset.get_data(), batch_size=1, shuffle=True)
    # Select model and parameters set
    Cmodel, model_root, model_params, train_params, eval_params = None, model_dir, None, None, None
    if model_type == 'ProVAE':
        Cmodel = ProVAE
        model_root += 'ProVAE/' + str(model_id) + '/cv/'
        model_params = args.pro_vae_cv_params
        train_params = args.pro_vae_cv_train
    elif model_type == 'ProDAAE':
        Cmodel = ProDAAE
        model_root += 'ProDAAE/' + str(model_id) + '/cv/'
        model_params = args.pro_daae_cv_params
        train_params = args.pro_daae_cv_train
    elif model_type == 'ProGAN':
        Cmodel = ProGAN
        model_root += 'ProGAN/' + str(model_id) + '/cv/'
        model_params = args.pro_gan_cv_params
        train_params = args.pro_gan_cv_train
    elif model_type == 'ProRNN':
        Cmodel = ProRNN
        model_root += 'ProRNN/' + str(model_id) + '/cv/'
        model_params = args.pro_rnn_cv_params
        train_params = args.pro_rnn_cv_train
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
        model = Cmodel(model_root, **param_config).to(args.device)
        model.train()
        model.fit(train_loader, **train_config)
        model.eval()
        loss = {}
        for i, (data) in enumerate(val_loader):
            if model_type == 'ProVAE':
                x, w_adj, mask = data['x'], data['w_wadj'], data['mask']
                loss_dict = model.eval_loss(x, w_adj, mask, train_config['l_kld'])
            elif model_type == 'ProDAAE':
                x, w_adj, mask = data['x'], data['w_wadj'], data['mask']
                loss_dict = model.eval_loss(x, w_adj, mask, train_config['l_adv'])
            elif model_type == 'ProGAN':
                x, w_adj, mask = data['x'], data['w_adj'], data['mask']
                loss_dict = model.eval_loss(x, w_adj, mask, train_config['l_wrl'])
            elif model_type == 'ProRNN':
                x, y, y_feat, y_len = data['x'], data['y_edge'], data['y_feat'], data['len']
                loss_dict = model.eval_loss(x, y, y_feat, y_len)
            if i == 0:
                for key in loss_dict:
                    loss[key] = []
            for key in loss_dict:
                loss[key].append(loss_dict[key].detach().item())
        for key in loss:
            loss[key] = np.array(loss[key])
        means, vars = [], []
        for key in loss:
            means.append(np.mean(loss[key]))
            vars.append(np.var(loss[key]))
        best_params.append((np.median(np.array(means)), np.median(np.array(vars)), param_config, train_config))
    if verbose:
        progress_bar.end()
    best_configs = sorted(best_params, key=lambda x: x[0])[0:args.cv_best_n_to_show]
    for best_config in best_configs:
        print(f'Average: {best_config[0]}, Variance: {best_config[1]}, \n Model Params.: {best_config[2]}, Training Params.: {best_config[3]}')

def _full(model_type, pdb_train, pdb_test, stride_dir, dataset_dir, model_dir, dataset_id=0, model_id=0, data_type='graph', data_mode='sparse', quality='qcp', train=True, verbose=False):
    if verbose:
        print('Processing the data...')
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
        train_dataset = GraphDataset(dataset_dir, dataset_id, 'train', train_proteins, pdb_train, stride_dir, **args.graph_dataset)
    elif data_type == 'rnn':
        train_dataset = RNNDataset_Feat(dataset_dir, dataset_id, 'train', train_proteins, pdb_train, stride_dir, **args.rrn_dataset)
    if verbose:
        print(f'Training set has {len(train_dataset)} samples.')
    # Get the test data
    test_dataset = GraphDataset(dataset_dir, 0, 'test', test_proteins, pdb_test, stride_dir, **args.test_dataset)
    # Save the datasets
    train_dataset.save()
    test_dataset.save()
    # Define the loaders
    if data_mode == 'dense' or data_type == 'rnn':
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    elif data_mode == 'sparse':
        train_loader = GDataLoader(dataset=train_dataset.get_data(), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    # Select the model
    Cmodel, model_root, model_params, train_params, eval_params = None, model_dir, None, None, None
    if model_type == 'ProVAE':
        Cmodel = ProVAE
        model_root += 'ProVAE/' + str(model_id) + '/full/'
        model_params = args.pro_vae_params
        train_params = args.pro_vae_train
        eval_params = args.pro_vae_eval
    elif model_type == 'ProDAAE':
        Cmodel = ProDAAE
        model_root += 'ProDAAE/' + str(model_id) + '/full/'
        model_params = args.pro_daae_params
        train_params = args.pro_daae_train
        eval_params = args.pro_daae_eval
    elif model_type == 'ProGAN':
        Cmodel = ProGAN
        model_root += 'ProGAN/' + str(model_id) + '/full/'
        model_params = args.pro_gan_params
        train_params = args.pro_gan_train
        eval_params = args.pro_gan_eval
    elif model_type == 'ProRNN':
        Cmodel = ProRNN
        model_root += 'ProRNN/' + str(model_id) + '/full/'
        model_params = args.pro_rnn_params
        train_params = args.pro_rnn_train
        eval_params = args.pro_rnn_eval
    # Create root dir if it does not exist
    if not os.path.exists(model_root):
        os.makedirs(model_root)
    # Define the model
    model = Cmodel(model_root, **model_params).to(args.device)
    # Train the model (if requested)
    if train:
        if verbose:
            print('Training...')
        model.train()
        model.fit(train_loader, **train_params)
        torch.save(model.state_dict(), model_root + 'model_state')
    else:
        model.load_state_dict(torch.load(model_root + 'model_state'))
    # Evaluate the model
    if verbose:
        print('Generating...')
    model.eval()
    x_pred, dist_pred = model.generate(**eval_params)
    # Reconstructing the fragment
    if verbose:
        print('Reconstructing the fragment...')
    # Assess the model performance
    if verbose:
        print('Assessing quality...')
    if quality == 'qcp':
        gram = GramReconstruction(args.device)
        X = gram.reconstruct(dist_pred)
        qcp = QCP(X, test_proteins)
        scores = qcp.superimpose()
    elif quality == 'mmd':
        pred_graph = (x_pred, dist_pred)
        target_graphs = []
        for _, (data) in enumerate(test_loader):
            x, w_adj_edge = data['x'], data['adj'] + data['edge'][:,:,:,1]
            x, w_adj_edge = x.view(x.shape[1],x.shape[2]), w_adj_edge.view(w_adj_edge.shape[1],w_adj_edge.shape[2])
            target_graphs.append((x, w_adj_edge))
        mmd = MMD(pred_graph, target_graphs)
        scores = mmd.compare_graphs()
    score_mean, score_var = np.mean(scores), np.var(scores)
    if verbose:
        print(f'Average: {score_mean}, Variance: {score_var}')

def _generate(model_type, model_dir, model_id=0, n_generate=10):
    # Select the model
    Cmodel, model_root, model_params = None, model_dir, None
    if model_type == 'ProVAE':
        Cmodel = ProVAE
        model_root += 'ProVAE/' + str(model_id) + '/full/'
        model_params = args.pro_vae_params
    if model_type == 'ProDAAE':
        Cmodel = ProDAAE
        model_root += 'ProDAAE/' + str(model_id) + '/full/'
        model_params = args.pro_daae_params
    if model_type == 'ProGAN':
        Cmodel = ProGAN
        model_root += 'ProGAN/' + str(model_id) + '/full/'
        model_params = args.pro_gan_params
    if model_type == 'ProRNN':
        Cmodel = ProRNN
        model_root += 'ProRNN/' + str(model_id) + '/full/'
        model_params = args.pro_rnn_params
    # Create the model
    model = Cmodel(model_root, **model_params).to(args.device)
    # Load the weights
    model.load_state_dict(torch.load(model_root + 'model_state'))
    # Generate
    gram = GramReconstruction(args.device)
    for i in range(n_generate):
        if model_type == 'ProRNN':
            for j in range(12,30):
                x_gen, dist_gen = model.generate(j)
                coords = gram.reconstruct(dist_gen)
                fb = FragmentBuilder('fragment_' + str(j) + str(i), x_gen, coords)
                fb.build(out_dir=model_root)
        else:
            x_gen, dist_gen = model.generate()
            coords = gram.reconstruct(dist_gen)
            fb = FragmentBuilder('fragment_' + str(x_gen.shape[0]) + str(i), x_gen, coords)
            fb.build(out_dir=model_root)

if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Full generation pipeline.')
    arg_parser.add_argument('mode', type=str, help='The mode of the pipeline. Valid modes are [grid_cv,full,generate].')
    arg_parser.add_argument('model', type=str, help='The model to use. Valid models are [ProVAE,ProDAAE,ProGAN,ProRNN].')
    arg_parser.add_argument('pdb_train', type=str, help='The directory holding the PDB files from the training set.')
    arg_parser.add_argument('pdb_val', type=str, help='The directory holding the PDB files from the validation set.')
    arg_parser.add_argument('pdb_test', type=str, help='The directory holding the PDB files from the test set.')
    arg_parser.add_argument('stride_dir', type=str, help='The directory holding the Stride tool.')
    arg_parser.add_argument('dataset_dir', type=str, help='The directory holding the dataset.')
    arg_parser.add_argument('model_dir', type=str, help='The directory holding the models weights.')
    arg_parser.add_argument('--dataset_id', type=int, default=0, help='The index of the dataset instance, which is used to save it. Note it only applies to datasets associated to the training data. The default is 0.')
    arg_parser.add_argument('--model_id', type=int, default=0, help='The index of the model instance, which is used to save its weights. The default is 0.')
    arg_parser.add_argument('--data_type', type=str, default='graph', help='The type of the dataset. Valid types are [graph,rnn]. The default is graph.')
    arg_parser.add_argument('--data_mode', type=str, default='sparse', help='The mode of the dataset. Valid types are [sparse,dense]. The default is sparse.')
    arg_parser.add_argument('--quality', type=str, default='qcp', help='The quality metric to use to assess the generated proteins. Valid measurements are [mmd,qcp]. The default is qcp.')
    arg_parser.add_argument('--train', type=bool, default=True, help='Whether to train the model. If not, then the model weights are loaded. This parameter has only effect when mode=full. The default is True.')
    arg_parser.add_argument('--n_generate', type=int, default=10, help='The number of graphs to generate. This has only effect when mode=generate. The default is 10.')
    arg_parser.add_argument('--verbose', type=bool, default=False, help='Whether to print progress information. The default is False.')
    # Parse arguments
    args_parsed = arg_parser.parse_args()
    # Choose pipeline
    if args_parsed.mode == 'grid_cv':
        _grid_cv(args_parsed.model, args_parsed.pdb_train, args_parsed.pdb_val, args_parsed.stride_dir, args_parsed.dataset_dir,
                 args_parsed.model_dir, dataset_id=args_parsed.dataset_id, model_id=args_parsed.model_id, data_type=args_parsed.data_type,
                 data_mode=args_parsed.data_mode, verbose=args_parsed.verbose)
    elif args_parsed.mode == 'full':
        _full(args_parsed.model, args_parsed.pdb_train, args_parsed.pdb_test, args_parsed.stride_dir, args_parsed.dataset_dir, args_parsed.model_dir,
              dataset_id=args_parsed.dataset_id, model_id=args_parsed.model_id, data_type=args_parsed.data_type, data_mode=args_parsed.data_mode,
              quality=args_parsed.quality, train=args_parsed.train, verbose=args_parsed.verbose)
    elif args_parsed.mode == 'generate':
        _generate(args_parsed.model, args_parsed.model_dir, model_id=args_parsed.model_id, n_generate=args_parsed.n_generate)
    else:
        print('Wrong mode selected.')