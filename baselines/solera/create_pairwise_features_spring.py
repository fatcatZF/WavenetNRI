import numpy as np
import networkx as nx
import torch
from scipy.stats import norm
import pickle
import os

import tslearn.metrics

import itertools
from itertools import combinations

import utils as u
from utils import *
from data_utils import *
from models_solera import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--suffix', type=str, default="_static_10_3_5",
                    help='suffix.')
parser.add_argument("--num-atoms", type=int, default=10, 
                    help="number of atoms.")

args = parser.parse_args()



save_folder="data/simulation/spring_simulation"

train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_spring_sim(suffix=args.suffix, 
                                                                                              normalize=False)

train_data = train_loader.dataset.tensors[0]
valid_data = valid_loader.dataset.tensors[0]
test_data = test_loader.dataset.tensors[0]

train_labels = train_loader.dataset.tensors[1][:,:,0]
valid_labels = valid_loader.dataset.tensors[1][:,:,0]
test_labels = test_loader.dataset.tensors[1]

all_data = torch.cat([train_data, valid_data, test_data], dim=0)
all_labels = torch.cat([train_labels, valid_labels, test_labels], dim=0)


ground = build_ground_spring(all_data)
sim, dissim, combined_features = compute_sims_spring(all_data, ground)

#store combined features
save_features_path = os.path.join(save_folder, "all_features"+args.suffix+".npy")
np.save(save_features_path, combined_features)


rel_rec, rel_send = u.create_edgeNode_relation(args.num_atoms, self_loops=False)


all_labels_diag = torch.diag_embed(all_labels)
all_labels_adj = torch.matmul(rel_send.t(), torch.matmul(all_labels_diag.float(), rel_rec))
all_labels_adj_numpy = all_labels_adj.numpy()



all_clusterings = []

for label in all_labels_adj_numpy:
    clusters = list(nx.connected_components(nx.from_numpy_array(label)))
    clusters = [list(c) for c in clusters]
    all_clusterings.append(clusters)
    

save_clustering_path = os.path.join(save_folder, "all_clustering"+args.suffix+".pkl")

with open(save_clustering_path, 'wb') as f:
    pickle.dump(all_clusterings, f)









