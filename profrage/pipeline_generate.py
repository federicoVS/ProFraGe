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
from utils.structure import structure_length
from utils.io import get_files, from_pdb
from utils.ProgressBar import ProgressBar

def _grid_cv(model_type, pdb_train, pdb_val, stride_dir, dataset_dir, model_dir, dataset_id=0, model_id=0, data_type='graph', data_mode='sparse', verbose=False):
    # Set seed
    torch.manual_seed(1000)
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
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    elif data_mode == 'sparse':
        train_loader = GDataLoader(dataset=train_dataset.get_data(), batch_size=args.batch_size, shuffle=False)
        val_loader = GDataLoader(dataset=val_dataset.get_data(), batch_size=args.batch_size, shuffle=False)
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
        val_loss, val_epoch = model.fit(train_loader, val_loader, **train_config)
        best_params.append((val_loss, val_epoch, param_config, train_config))
    if verbose:
        progress_bar.end()
    best_configs = sorted(best_params, key=lambda x: x[0])[0:args.cv_best_n_to_show]
    for best_config in best_configs:
        print(f'Validation Loss: {best_config[0]}, Epoch: {best_config[1]}, \n Model Params.: {best_config[2]}, Training Params.: {best_config[3]}')

def _full(model_type, pdb_train, pdb_val, pdb_test, stride_dir, dataset_dir, model_dir, dataset_id=0, model_id=0, data_type='graph', data_mode='sparse', train=True, verbose=False):
    if verbose:
        print('Processing the data...')
    train_pdbs, val_pdbs, test_pdbs = get_files(pdb_train, ext='.pdb'),  get_files(pdb_val, ext='.pdb'), get_files(pdb_test, ext='.pdb')
    train_proteins, val_proteins, test_proteins = [], [], []
    # Get the training proteins
    for pdb in train_pdbs:
        pdb_id = os.path.basename(pdb)[:-4]
        train_proteins.append(from_pdb(pdb_id, pdb, quiet=True))
    # Get the validation proteins
    for pdb in val_pdbs:
        pdb_id = os.path.basename(pdb)[:-4]
        val_proteins.append(from_pdb(pdb_id, pdb, quiet=True))
    # Get the test proteins
    for pdb in test_pdbs:
        pdb_id = os.path.basename(pdb)[:-4]
        test_proteins.append(from_pdb(pdb_id, pdb, quiet=True))
    # Get the training and validation data
    if data_type == 'graph':
        train_dataset = GraphDataset(dataset_dir, dataset_id, 'train', train_proteins, pdb_train, stride_dir, **args.graph_dataset)
        val_dataset = GraphDataset(dataset_dir, 0, 'val', val_proteins, pdb_val, stride_dir, **args.graph_val_dataset)
    elif data_type == 'rnn':
        train_dataset = RNNDataset_Feat(dataset_dir, dataset_id, 'train', train_proteins, pdb_train, stride_dir, **args.rrn_dataset)
        if train:
            train_dataset = train_dataset.sample()
        val_dataset = RNNDataset_Feat(dataset_dir, 0, 'val', val_proteins, pdb_val, stride_dir, **args.rrn_val_dataset)
    if verbose:
        print(f'Training set has {len(train_dataset)} samples.')
    # Get the test data
    test_dataset = GraphDataset(dataset_dir, 0, 'test', test_proteins, pdb_test, stride_dir, **args.test_dataset)
    # Save the datasets
    if data_type == 'graph':
        train_dataset.save()
    test_dataset.save()
    # Set seed (after dataset sampling)
    if train:
        torch.manual_seed(1000)
    # Define the loaders
    if data_mode == 'dense' or data_type == 'rnn':
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)
    elif data_mode == 'sparse':
        train_loader = GDataLoader(dataset=train_dataset.get_data(), batch_size=args.batch_size, shuffle=False)
        val_loader = GDataLoader(dataset=val_dataset.get_data(), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
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
        model.fit(train_loader, val_loader, **train_params)
        torch.save(model.state_dict(), model_root + 'model_state')
    else:
        model.load_state_dict(torch.load(model_root + 'model_state'))
    # Evaluate the model
    if verbose:
        print('Generating...')
    model.eval()
    x_pred, dist_pred = model.generate(**eval_params)
    if model_type == 'ProRNN':
        x_pred, dist_pred = x_pred[0], dist_pred[0]
    # Reconstructing the fragment
    if verbose:
        print('Reconstructing the fragment...')
    # Assess the model performance with QCP
    if verbose:
        print('Assessing QCP quality...')
    gram = GramReconstruction(args.device)
    X = gram.reconstruct(dist_pred)
    n_nodes = X.shape[0]
    target_proteins = []
    for i in range(len(test_proteins)):
        sl = structure_length(test_proteins[i])
        if min(sl,x_pred.shape[0])/max(sl,x_pred.shape[0]) >= 0.6:
            target_proteins.append(test_proteins[i])
    qcp = QCP(X, target_proteins)
    qcp_scores = qcp.superimpose()
    qcp_min, qcp_score_mean, qcp_score_var = np.min(qcp_scores), np.mean(qcp_scores), np.var(qcp_scores)
    qcp_file = open('qcp_' + str(n_nodes), 'a')
    for i in range(qcp_scores.shape[0]):
        qcp_file.write(str(qcp_scores[i]))
    qcp_file.close()
    if verbose:
        print('Assessing MMD quality...')
    pred_graph = (x_pred, dist_pred)
    target_graphs = []
    for _, (data) in enumerate(test_loader):
        x, w_adj_edge, mask = data['x'], data['adj'] + data['edge'][:,:,:,1], data['mask']
        x, w_adj_edge = x.view(x.shape[1],x.shape[2]), w_adj_edge.view(w_adj_edge.shape[1],w_adj_edge.shape[2])
        if min(sum(mask[0]).item(),x_pred.shape[0])/max(sum(mask[0]).item(),x_pred.shape[0]) >= 0.6:
            target_graphs.append((x, w_adj_edge))
    mmd = MMD(pred_graph, target_graphs)
    mmd_scores = mmd.compare_graphs()
    mmd_min, mmd_mean, mmd_var = np.min(abs(mmd_scores)), np.mean(mmd_scores), np.var(mmd_scores)
    mmd_file = open('mmd_' + str(n_nodes), 'a')
    for i in range(mmd_scores.shape[0]):
        mmd_file.write(str(mmd_scores[i]))
    mmd_file.close()
    if verbose:
        print(f'QCP -> Average: {qcp_score_mean}, Variance: {qcp_score_var}, Minimum: {qcp_min}')
        print(f'MMD -> Average: {mmd_mean}, Variance: {mmd_var}, Minimum: {mmd_min}')

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
    if model_type == 'ProRNN':
        for j in range(12,30):
            x_gen, dist_gen = model.generate(1, j)
            x_gen, dist_gen = x_gen[0], dist_gen[0]
            coords = gram.reconstruct(dist_gen)
            fb = FragmentBuilder('fragment_' + str(j), x_gen, coords)
            fb.build(out_dir=model_root)
    else:
        for i in range(n_generate):
            x_gen, dist_gen = model.generate()
            coords = gram.reconstruct(dist_gen)
            fb = FragmentBuilder('fragment_' + str(x_gen.shape[0]) + str(i), x_gen, coords)
            fb.build(out_dir=model_root)

def _get_model_summary(model_type, model_dir, model_id=0):
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
    # Get summary
    summary = str(model)
    # Write summary
    s_file = open('summary.txt', 'w')
    s_file.write(summary)
    s_file.close()

if __name__ == '__main__':
    # Argument parser initialization
    arg_parser = argparse.ArgumentParser(description='Full generation pipeline.')
    arg_parser.add_argument('mode', type=str, help='The mode of the pipeline. Valid modes are [grid_cv,full,generate,summary].')
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
        _full(args_parsed.model, args_parsed.pdb_train, args_parsed.pdb_val, args_parsed.pdb_test, args_parsed.stride_dir, args_parsed.dataset_dir, args_parsed.model_dir,
              dataset_id=args_parsed.dataset_id, model_id=args_parsed.model_id, data_type=args_parsed.data_type, data_mode=args_parsed.data_mode,
              train=args_parsed.train, verbose=args_parsed.verbose)
    elif args_parsed.mode == 'generate':
        _generate(args_parsed.model, args_parsed.model_dir, model_id=args_parsed.model_id, n_generate=args_parsed.n_generate)
    elif args_parsed.mode == 'summary':
        _get_model_summary(args_parsed.model, args_parsed.model_dir, model_id=args_parsed.model_id)
    else:
        print('Wrong mode selected.')