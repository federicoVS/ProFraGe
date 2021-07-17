import os

import argparse

from torch.utils.data import DataLoader

from generate import args
from generate.vae import GraphVAE, GraphVAEEdge
from generate.gan import ProGAN
from generate.rnn import GraphRNN_A
from generate.dae import GraphDAE
from generate.datasets import GraphDataset, RNNDataset_Feat
from utils.io import get_files, from_pdb

def _generate(pdb_dir, stride_dir, dataset_dir, model_type, gen_type='graph', compute_data=True, train=True):
    # Get (or compute) the data
    if compute_data:
        pdbs = get_files(pdb_dir, ext='.pdb')
        proteins = []
        for pdb in pdbs:
            pdb_id = os.path.basename(pdb)[:-4]
            proteins.append(from_pdb(pdb_id, pdb, quiet=True))
        if gen_type == 'graph':
            dataset = GraphDataset(proteins, pdb_dir, stride_dir, **args.graph_dataset)
        elif gen_type == 'rnn':
            dataset = RNNDataset_Feat(proteins, pdb_dir, stride_dir, **args.rrn_dataset)
        dataset.save(dataset_dir) # TODO implement in dataset
    else:
        dataset = None # TODO implement loading in the dataset (with an additional parameter)
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    # Select the model
    if model_type == 'GraphVAE':
        model = GraphVAE(**args.graph_vae)
        if train:
            model.train(args.graph_vae_train)
            # TODO save weights
        else:
            # TODO load weights
            a=1
        x_pred, x_pred_atom, adj_pred = model.eval(**args.graph_vae_eval)
    elif model_type == 'ProGAN':
        model = ProGAN(**args.pro_gan)
        if train:
            model.train(args.pro_gan_train)
            # TODO save weights
        else:
            # TODO load weights
            a=1
        x_gen, adj_gen, edge_gen = model.eval(**args.pro_gan_eval)
    elif model_type == 'GraphDAE':
        model = GraphDAE(**args.graph_dae)
        if train:
            model.train(args.graph_dae_train)
            # TODO save weights
        else:
            # TODO load weights
            a=1
        x_pred, x_pred_atom, adj_pred = model.eval(**args.graph_dae_eval)
    elif model_type == 'GraphRNN':
        model = GraphRNN_A(**args.graph_rnn)
        if train:
            model.train(args.graph_rnn_train)
            # TODO save weights
        else:
            # TODO load weights
            a=1
        x_pred, adj_pred = model.eval(**args.graph_rnn_eval)