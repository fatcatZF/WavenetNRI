import numpy as np
import torch
from scipy.stats import norm
from statsmodels.tsa.stattools import grangercausalitytests
import tslearn.metrics

import itertools
from itertools import combinations

import utils
from utils import *

#Gaussian Mixture Models
N0 = norm(0, 0.5)
N1 = norm(0, 1.2)
N2 = norm(0, 3.7)
N3 = norm(0, 7.6)

GMM = [N0, N1, N2, N3]



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def create_edgeNode_relation(num_nodes, self_loops=False):
    if self_loops:
        indices = np.ones([num_nodes, num_nodes])
    else:
        indices = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
    rel_rec = np.array(encode_onehot(np.where(indices)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(indices)[1]), dtype=np.float32)
    
    return rel_rec, rel_send 




"""
GMM for distances
"""
def compute_gmm(distance, GMM=GMM):
    """
    args:
        distance:
            distance between 2 agents at one timestep
        GMM:
            Gaussian Mixture Model
    """
    num = len(GMM)
    probs = [N.pdf(distance) for N in GMM]
    return sum(probs)/num



def compute_gmmDist_example(example):
    """
    args:
      example, shape: [n_atoms, n_timesteps, n_in]
    """
    #extract locations of the example
    locs = example[:,:,:2] #extract locations, shape: [n_atoms, n_timesteps, 2]
    n_atoms = example.shape[0]
    n_timesteps = example.shape[1]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    #shape: [n_edges, n_atoms]
    locs_re = locs.reshape(locs.shape[0], -1) #shape: [n_atoms, n_timesteps*2]
    senders_locs = np.matmul(rel_send, locs_re) #shape: [n_edges, n_timesteps*2]
    receivers_locs = np.matmul(rel_rec, locs_re)
    senders_locs = senders_locs.reshape(senders_locs.shape[0], n_timesteps, -1)
    receivers_locs = receivers_locs.reshape(receivers_locs.shape[0], n_timesteps, -1)
    distances = np.sqrt(((senders_locs-receivers_locs)**2).sum(-1)) #shape: [n_edges, n_timesteps]
    #compute GMM probs
    distances_re = distances.reshape(-1)
    probs = np.array([compute_gmm(dist) for dist in distances_re])
    probs = probs.reshape(distances.shape[0], -1)
    probs = probs.mean(-1) #shape: [n_atoms*(n_atoms-1)]
    
    return probs



def compute_gmmDist_spring(sims):
    """
    compute gmm distributions of spring simulation data
    args:
        sims, shape:[n_sims, n_atoms, n_timesteps, n_in]
    """
    locs = sims[:,:,:,:2] #to extract locations, shape:[n_sims, n_atoms, n_timesteps, 2]
    locs_re = locs.reshape(locs.size(0), locs.size(1),-1)
    #shape: [n_sims, n_atoms, n_timesteps*2]
    n_sims = sims.size(0)
    n_atoms = sims.size(1)
    n_timesteps = sims.size(2)
    rel_rec, rel_send = utils.create_edgeNode_relation(n_atoms, self_loops=False)
    #shape: [n_edges, n_atoms]
    senders_locs = torch.matmul(rel_send, locs_re)
    receivers_locs = torch.matmul(rel_rec, locs_re)
    #shape: [n_sims, n_edges, n_timesteps*2]
    senders_locs = senders_locs.reshape(senders_locs.size(0), senders_locs.size(1), n_timesteps, -1)
    #shape: [n_sims, n_edges, n_timesteps, 2]
    receivers_locs = receivers_locs.reshape(receivers_locs.size(0), receivers_locs.size(1), n_timesteps, -1)
    distances = torch.sqrt(((senders_locs-receivers_locs)**2).sum(-1))
    #shape: [n_sims, n_edges, n_timesteps]
    distances_re = distances.reshape(-1)
    #shape: [n_sims*n_edges*n_timesteps]
    distances_re = distances_re.numpy()
    probs = np.array([compute_gmm(dist) for dist in distances_re])
    probs = probs.reshape(n_sims, (n_atoms-1)*n_atoms, n_timesteps)
    probs = probs.mean(-1)
    #shape: [n_sims, n_edges]
    return probs
    




"""
Granger Causality
"""
def compute_granger_p(example):
    n_atoms = example.shape[0]
    n_timesteps = example.shape[1]
    locs = example[:,:,:] #shape: [n_atoms, n_timesteps, n_in]
    locs_re = locs.reshape(n_atoms, -1) #shape: [n_atoms, n_timesteps*2]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    #shape: [n_edges, n_nodes]
    senders = np.matmul(rel_send, locs_re) #shape: [n_edges, n_timesteps*2]
    receivers = np.matmul(rel_rec, locs_re)
    senders = senders.reshape(senders.shape[0], n_timesteps, -1)
    receivers = receivers.reshape(receivers.shape[0], n_timesteps, -1)
    #shape: [n_edges, n_timesteps, 2]
    senders = np.sqrt((senders**2).sum(-1))
    receivers = np.sqrt((receivers**2).sum(-1))
    #shape: [n_edges, n_timesteps]
    senders = senders.reshape(senders.shape[0], n_timesteps, -1)
    receivers = receivers.reshape(receivers.shape[0], n_timesteps, -1)
    #shape: [n_edges, n_timesteps, 1]

    ps = []

    for i in range(senders.shape[0]):
        result_sr = grangercausalitytests(np.concatenate([senders[i], receivers[i]], axis=-1), maxlag=4, verbose=0)
        p_sr = np.array([lag[0]["ssr_ftest"][1] for lag in result_sr.values()]).mean()
        result_rs = grangercausalitytests(np.concatenate([receivers[i], senders[i]], axis=-1), maxlag=4, verbose=0)
        p_rs = np.array([lag[0]["ssr_ftest"][1] for lag in result_rs.values()]).mean()
        ps.append(max(p_sr, p_rs))
    
    ps = np.array(ps) #shape: [n_atoms*(n_atoms-1)]
    
    return ps 


def compute_granger_sim(example):
    ps = compute_granger_p(example)
    return 1-ps




"""
DTW Distances
"""
# compute DTW distances
def compute_dtw_dist(example):
    """
    args:
      example, shape: [n_atoms, n_timesteps, n_in]
    """
    n_atoms = example.shape[0]
    n_timesteps = example.shape[1]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    #shape: [n_edges, n_atoms]
    example_re = example.reshape(example.shape[0], -1)
    #shape: [n_atoms, n_timesteps*n_in]
    senders = np.matmul(rel_send, example_re)
    receivers = np.matmul(rel_rec, example_re)
    #shape: [n_edges, n_timesteps*n_in]
    
    senders = senders.reshape(senders.shape[0], n_timesteps, -1)
    receivers = receivers.reshape(receivers.shape[0], n_timesteps, -1)
    #shape: [n_edges, n_timesteps, n_in]
    
    n_edges = n_atoms*(n_atoms-1)
    distances = []
    for i in range(n_edges):
        distances.append(tslearn.metrics.dtw(senders[i], receivers[i]))
    
    return np.array(distances) #shape: [n_edges]




def compute_dtw_dist_spring(sims):
    """
    args:
        sims, shape:[n_sims, n_atoms, n_timesteps, n_in]
    """
    n_sims = sims.size(0)
    n_atoms = sims.size(1)
    n_timesteps = sims.size(2)
    rel_rec, rel_send = utils.create_edgeNode_relation(n_atoms, self_loops=False)
    #shape: [n_edges, n_atoms]
    sims_re = sims.reshape(n_sims, n_atoms, -1)
    #shape: [n_sims, n_atoms, n_timesteps*n_in]
    senders = torch.matmul(rel_send, sims_re)
    receivers = torch.matmul(rel_rec, sims_re)
    #shape: [n_sims, n_edges, n_timesteps*n_in]
    
    senders = senders.reshape(n_sims, senders.size(1), n_timesteps, -1)
    receivers = receivers.reshape(n_sims, receivers.size(1), n_timesteps, -1)
    #shape:[n_sims, n_edges, n_timesteps, n_in]
    
    n_edges = n_atoms*(n_atoms-1)
    senders = senders.reshape(n_sims*n_edges, n_timesteps, -1)
    receivers = receivers.reshape(n_sims*n_edges, n_timesteps, -1)
    
    distances = []
    for i in range(n_edges*n_sims):
        distances.append(tslearn.metrics.dtw(senders[i], receivers[i]))
    
    distances = np.array(distances)
    distances = distances.reshape(n_sims, n_edges)
    
    return distances







#compute DTW similarity
def compute_dtw_sim(example):
    """
    args:
        example, shape:[n_atoms, n_timesteps, n_in]
    """
    distances = compute_dtw_dist(example)
    
    return np.exp(-distances)



def compute_dtw_sim_spring(sims):
    """
    args:
        sims, shape: [n_sims, n_atoms, n_timesteps, n_in]
    """
    distances = compute_dtw_dist_spring(sims)
    return np.exp(-distances)




"""
Compute heatmap
"""

def build_ground(examples_train):
    """
    build ground for heatmap based on training examples
    args:
      examples_train: training examples
    """
    max_train_x = -np.inf
    min_train_x = np.inf
    max_train_y = -np.inf
    min_train_y = np.inf
    
    for example in examples_train:
        max_example_x = example[:,:,0].max()
        max_example_y = example[:,:,1].max()
        min_example_x = example[:,:,0].min()
        min_example_y = example[:,:,1].min()
        if max_example_x > max_train_x:
            max_train_x = max_example_x
        if max_example_y > max_train_y:
            max_train_y  = max_example_y
        if min_example_x < min_train_x:
            min_train_x = min_example_x
        if min_example_y < min_train_y:
            min_train_y = min_example_y
            
    Rs = np.arange(int(min_train_x)-2.5, int(max_train_x)+2.5, 1)
    Cs = np.arange(int(min_train_y)-2.5, int(max_train_y)+2.5, 1) 
    
    #build ground
    ground = np.zeros((Rs.shape[0], Cs.shape[0], 2))   
    for i in range(Rs.shape[0]):
        for j in range(Cs.shape[0]):
            ground[i,j,0] = Rs[i]
            ground[i,j,1] = Cs[j]
        
    return ground



def build_ground_spring(sims):
    """
    build ground for heatmap for spring simulation
    based on 
    args:
        sims, shape: [n_sims, n_atoms, n_timesteps, n_in]
    """
    train_data_xs = sims[:,:,:,0]
    train_data_ys = sims[:,:,:,1]
    max_train_x = train_data_xs.max()
    min_train_x = train_data_xs.min()
    max_train_y = train_data_ys.max()
    min_train_y = train_data_ys.min()
    
    Rs = np.arange(int(min_train_x)-2.5, int(max_train_x)+2.5, 1)
    Cs = np.arange(int(min_train_y)-2.5, int(max_train_y)+2.5, 1) 
    
    #build ground
    ground = np.zeros((Rs.shape[0], Cs.shape[0], 2))   
    for i in range(Rs.shape[0]):
        for j in range(Cs.shape[0]):
            ground[i,j,0] = Rs[i]
            ground[i,j,1] = Cs[j]
            
    return ground



    

def compute_heatmap_traj(traj, ground):
    """
    compute heatmap of one trajectory
    args:
      traj: trajectory of one agent
        shape: [n_timesteps, 2]
    ground:
       coordinates of heat on ground
    """
    heatmap = np.zeros((ground.shape[0], ground.shape[1]))
    for loc in traj:
        for r in range(ground.shape[0]):
            for c in range(ground.shape[1]):
                if np.sqrt(((ground[r,c]-loc)**2).sum())< 1.:
                    heatmap[r, c] = 1.
    
    return heatmap



def compute_heatmap_sim(example, ground):
    """
    compute heatmap similarities
    args:
      example: [n_atoms, n_timesteps, n_in]
      ground: [R X C]
    """
    locs = example[:,:,:2]#extract locations
    #shape: [n_atoms, n_timesteps, 2]
    n_atoms = example.shape[0]
    n_timesteps = example.shape[1]
    rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
    locs_re = locs.reshape(locs.shape[0], -1)
    #shape: [n_atoms, n_timesteps*2]
    senders = np.matmul(rel_send, locs_re)
    receivers = np.matmul(rel_rec, locs_re)
    #shape: [n_edges, n_timesteps*2]
    senders = senders.reshape(senders.shape[0], n_timesteps, -1)
    receivers = receivers.reshape(receivers.shape[0], n_timesteps, -1)
    #shape: [n_edges, n_timesteps, 2]
    n_edges = n_atoms*(n_atoms-1)
    sims = []
    for i in range(n_edges):
        traj_s = senders[i]
        traj_r = receivers[i]
        heatmap_s = compute_heatmap_traj(traj_s, ground)
        heatmap_r = compute_heatmap_traj(traj_r, ground)
        sim_sr = (heatmap_r*heatmap_s).sum()/(np.sqrt(ground.shape[0]*ground.shape[1]))
        sims.append(sim_sr)
        
    return np.array(sims) #shape: [n_edges]




def compute_heatmap_sim_spring(sims, ground):
    """
    compute heatmap similarities for spring 
    args:
        sims: [n_sims, n_atoms, n_timesteps, n_in]
    """
    locs = sims[:,:,:,:2]
    #shape: [n_sims, n_atoms, n_timesteps, 2]
    n_sims = locs.size(0)
    n_atoms = locs.size(1)
    n_timesteps = locs.size(2)
    rel_rec, rel_send = utils.create_edgeNode_relation(n_atoms, self_loops=False)
    #shape: [n_edges, n_atoms]
    locs_re = locs.reshape(n_sims, n_atoms, -1)
    #shape: [n_sims, n_atoms, n_timesteps*2]
    senders = torch.matmul(rel_send, locs_re)
    receivers = torch.matmul(rel_rec, locs_re)
    #shape: [n_sims, n_edges, n_timesteps*2]
    senders = senders.reshape(n_sims, n_atoms*(n_atoms-1), n_timesteps, -1)
    receivers = receivers.reshape(n_sims, n_atoms*(n_atoms-1), n_timesteps, -1)
    #shape: [n_sims, n_edges, n_timesteps, 2]
    
    
    n_edges = n_atoms*(n_atoms-1)
    senders = senders.reshape(n_sims*n_edges, n_timesteps, -1)
    receivers = receivers.reshape(n_sims*n_edges, n_timesteps, -1)
    senders = senders.numpy()
    receivers = receivers.numpy()
    
    simls = []
    for i in range(n_sims*n_edges):
        traj_s = senders[i]
        traj_r = receivers[i]
        heatmap_s = compute_heatmap_traj(traj_s, ground)
        heatmap_r = compute_heatmap_traj(traj_r, ground)
        sim_sr = (heatmap_r*heatmap_s).sum()/(np.sqrt(ground.shape[0]*ground.shape[1]))
        simls.append(sim_sr)
    simls = np.array(simls)
    simls = simls.reshape(n_sims, n_edges)
    return simls






"""
compute pairwise similarity for examples

"""
def compute_sims(example, ground):
    """
    compute pairwise similarity and dissimilarity for one example
    args:
      example: [n_atoms, n_timesteps, n_in]
    """
    #compute GMM distances Sim
    gmm_sim = compute_gmmDist_example(example)
    gmm_dissim = (1-gmm_sim).reshape(-1,1)
    gmm_sim = gmm_sim.reshape(-1,1)
    
    #compute Granger Causality Sim
    #granger_sim = compute_granger_sim(example)
    #granger_dissim = (1-granger_sim).reshape(-1,1)
    #granger_sim = granger_sim.reshape(-1,1)
    
    #compute DTW sim
    dtw_sim = compute_dtw_sim(example)
    dtw_dissim = (1-dtw_sim).reshape(-1,1)
    dtw_sim = dtw_sim.reshape(-1, 1)
    
    #compute heatmap sim
    heatmap_sim = compute_heatmap_sim(example, ground)
    heatmap_dissim = (1-heatmap_sim).reshape(-1,1)
    heatmap_sim = heatmap_sim.reshape(-1, 1)
    
    sim = np.concatenate([gmm_sim, dtw_sim, heatmap_sim], axis=-1)
    dissim = np.concatenate([gmm_dissim, dtw_dissim, heatmap_dissim], axis=-1)
    combined_features = np.concatenate([sim, dissim], axis=-1)
    
    return sim, dissim, combined_features





def compute_sims_spring(sims, ground):
    """
    compute pairwise similarity and dissimilarity for one example
    args:
        sims: [n_sims, n_atoms, n_timesteps, n_in]
    """
    #compute GMM distances similarities
    gmm_sim = compute_gmmDist_spring(sims)
    gmm_dissim = (1-gmm_sim).reshape(gmm_sim.shape[0], gmm_sim.shape[1], 1)
    #shape: [n_sims, n_edges, 1]
    gmm_sim = gmm_sim.reshape(gmm_sim.shape[0], gmm_sim.shape[1], 1)
    
    #compute DTW sim
    dtw_sim = compute_dtw_dist_spring(sims)
    dtw_dissim = (1-dtw_sim).reshape(dtw_sim.shape[0], dtw_sim.shape[1], 1)
    dtw_sim = dtw_sim.reshape(dtw_sim.shape[0], dtw_sim.shape[1], 1)
    
    #compute heatmap sim
    heatmap_sim = compute_heatmap_sim_spring(sims, ground)
    heatmap_dissim = (1-heatmap_sim).reshape(heatmap_sim.shape[0],
                                            heatmap_sim.shape[1], 1)
    heatmap_sim = heatmap_sim.reshape(heatmap_sim.shape[0],
                                     heatmap_sim.shape[1], 1)
    sim = np.concatenate([gmm_sim, dtw_sim, heatmap_sim], axis=-1)
    dissim = np.concatenate([gmm_dissim, dtw_dissim, heatmap_dissim], axis=-1)
    combined_features = np.concatenate([sim, dissim], axis=-1)
    
    return sim, dissim, combined_features









def compute_all_clusterings(indices):
    """
    args:
        indices: indices of items
    """
    if len(indices)==1:
        yield [indices]
        return
    first = indices[0]
    for smaller in compute_all_clusterings(indices[1:]):
        # insert "first" in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n]+[[first]+subset]+smaller[n+1:]
        yield [[first]]+smaller





def compute_clustering_score(sims, clustering):
    """
    args:
        sims: similarity matrix
        clustering: list of lists denoting clusters
    """
    score = 0.
    for cluster in clustering:
        if len(cluster)>=2:
            combs = list(combinations(cluster, 2))
            for comb in combs:
                score += sims[comb]
    return score





def merge_2_clusters(current_clustering, indices):
    """
    merge 2 clusters of current clustering
    args:
        current_clustering: list of lists denoting clusters
        indices(tuple): indices of 2 clusters of current clustering
    """
    assert len(current_clustering)>1
    num_clusters = len(current_clustering)
    cluster1 = current_clustering[indices[0]]
    cluster2 = current_clustering[indices[1]]
    merged_cluster = cluster1+cluster2
    new_clustering = [merged_cluster]
    for i in range(num_clusters):
        if i!=indices[0] and i!=indices[1]:
            new_clustering.append(current_clustering[i])
    return new_clustering






def greedy_approximate_best_clustering(sims):
    """
    args:
        sims(numpy ndarray): similarity matrices, shape:[n_atoms, n_atoms]
        current_clustering: a list of lists denoting clusters
        current_score: current clustering score
    """
    num_atoms = sims.shape[0]
    current_cluster_indices = list(range(num_atoms))
    current_clustering = [[i] for i in current_cluster_indices]
    current_score = 0.
    merge_2_indices = list(combinations(current_cluster_indices, 2))
    best_clustering = current_clustering
    
    
    while(True):
        #merge 2 clusters hierachically
        
        #if len(current_clustering)==1: #cannot be merged anymore
        #    return current_clustering, current_score
        
        best_delta = 0
        for merge_index in merge_2_indices:
            new_clustering = merge_2_clusters(current_clustering, merge_index)
            new_score = compute_clustering_score(sims, new_clustering)
            delta = new_score-current_score
            if delta>best_delta:
                best_clustering = new_clustering
                best_delta = delta
                current_score = new_score
        if best_delta==0:
            return best_clustering, current_score
        
        current_clustering = best_clustering
        current_num_clusters = len(current_clustering)
        if current_num_clusters==1:
            return current_clustering, current_score
        cluster_indices = list(range(current_num_clusters))
        merge_2_indices = list(combinations(cluster_indices, 2))
        
        



def compute_features_matrix(features, n_atoms):
    """
    compute feature matrix given edge features
    args:
      features: [n_edges, n_in]
      n_atoms: number of atoms
    """
    assert features.shape[0]==n_atoms*(n_atoms-1)
    features_matrix = features.reshape(n_atoms, n_atoms-1, -1)
    features_matrix = features_matrix.tolist()
    for i in range(len(features_matrix)):
        feature = [0]*features.shape[1]
        features_matrix[i].insert(i, feature)
    features_matrix = np.array(features_matrix)
    
    return features_matrix








def compute_combined_rep(example, ground ,clustering, features=None):
    """
    compute combined feature representation
    args:
        example: [n_atoms, n_timesteps, n_shape]
        ground: [R, C, 2]
        clustering: a list of lists denoting clusters
    return:
        combined feature representation
    """
    n_atoms = example.shape[0]
    if features is None:
        _, _ ,features = compute_sims(example, ground)
        #features shape: [n_atoms*(n_atoms-1), n_features]
    features_matrix = compute_features_matrix(features, n_atoms)
    
    combined_rep = np.zeros(features.shape[1])
    for cluster in clustering:
        combs = list(combinations(cluster, 2))
        for index in combs:
            combined_rep += features_matrix[index]
            
    return combined_rep




def compute_delta_rep(example, ground, label, predicted, features=None):
    """
    compute delta representation
    args:
      label: label clustering
      predicted: predicted clustering
      features: [n_edges, n_in]
    """
    combined_rep_label = compute_combined_rep(example, ground, label, features)
    combined_rep_pred = compute_combined_rep(example, ground, predicted, features)
    delta_rep = combined_rep_label-combined_rep_pred
    return delta_rep




def compute_structured_hinge(example, ground, label, predicted, w, features=None):
    """
    compute structured hinge loss
    args:
        example, shape: [n_atoms, n_timesteps, n_in]
        label/predicted: label and predicted clustering
    """
    gmitre = compute_gmitre_loss(label, predicted)
    delta_rep = compute_delta_rep(example, ground, label, predicted, features)
    return gmitre-np.dot(w, delta_rep), gmitre, delta_rep




def approximate_most_violated(example, ground, label, w, features=None):
    """
    greedy approximate most violated clustering
    args:
        example, shape: [n_atoms, n_timesteps, n_in]
        label: clustering, a list of lists denoting clusters
        w: weights
    """
    n_atoms = example.shape[0]
    current_cluster_indices = list(range(n_atoms))
    current_clustering = [[i] for i in current_cluster_indices]
    hinge_loss, gmitre, delta_rep = compute_structured_hinge(example, ground, label, current_clustering, w, features)
    merge_2_indices = list(combinations(current_cluster_indices, 2))
    worst_clustering = current_clustering
    
    while(True):
        #merge 2 clusters hierachically
        #if len(current_clustering)==1:
        #    return current_clustering, hinge_loss, gmitre, delta_rep
        most_delta = 0
        for merge_index in merge_2_indices:
            new_clustering = merge_2_clusters(current_clustering, merge_index)
            new_hinge_loss, new_gmitre, new_delta_rep = compute_structured_hinge(example, ground, 
                                                                                 label, new_clustering, w, features)
            delta = new_hinge_loss-hinge_loss
            if delta>most_delta:
                worst_clustering=new_clustering
                most_delta = delta
                hinge_loss = new_hinge_loss
                gmitre = new_gmitre
                delta_rep = new_delta_rep
        if most_delta==0:
            return worst_clustering, hinge_loss, gmitre, delta_rep
        
        current_clustering = worst_clustering
        current_num_clusters = len(current_clustering)
        if current_num_clusters==1:
            return current_clustering, hinge_loss, gmitre, delta_rep
        cluster_indices = list(range(current_num_clusters))
        merge_2_indices = list(combinations(cluster_indices, 2))
        




class SoleraSVM:
    def __init__(self, n_features=6):
        self.__init_weights(n_features)
        self.__init_l(n_features)
    
    def __init_weights(self, n_features):
        #initialize weights
        self.w = np.zeros(n_features)
        
        
    def __init_l(self, n_features):
        #initialize l
        self.l = 0.
    
    def fit_1_example_bcfw(self, example, ground ,label, n_examples, wi, li, C, features):
        """
        train model with BCFW algorithm
        """
        worst_clustering, hinge_loss, gmitre, delta_rep = approximate_most_violated(example, 
                                                                                    ground, label, self.w, features)
        ws = (C/n_examples)*delta_rep
        ls = (C/n_examples)*gmitre
        #print("ws: ",ws)
        #print("ls: ",ls)
        gamma = (np.dot((wi-ws), self.w)+(C/n_examples)*(ls-li))/(((wi-ws)**2).sum()+1e-6)
        #print("gamma: ",gamma)
        if gamma>1:
            gamma=1
        if gamma<0:
            gamma=0
        wi_new = (1-gamma)*wi+gamma*ws
        li_new = (1-gamma)*li+gamma*ls
        self.w = self.w+wi_new-wi
        self.l = self.l+li_new-li
        return wi_new, li_new, gmitre, hinge_loss
        
    
    def fit_minibatch_fw(self, examples, ground, labels, batch_size, wis, lis, C, batch_features):
        pass
    
    
    
    def fit(self, examples, ground ,labels , n_iters, C=10., verbose=0, verbose_iters=100 ,pairwise_features=None):
        """
        train model with BCFW algorithm
        args:
            examples: a list of training examples
            labels: a list of labels
            c: regularization parameter
        """
        n_features = self.w.shape[0]
        n_examples = len(examples)
        indices_training = np.arange(n_examples)
        
        wis = np.zeros((indices_training.shape[0], n_features))
        self.wis_dict = dict(zip(indices_training, wis))
        lis = np.zeros(len(indices_training))
        self.lis_dict = dict(zip(indices_training, lis))
        
        for i in range(n_iters):
            current_index = np.random.choice(indices_training, 1)[0]
            current_example = examples[current_index]
            current_label = labels[current_index]
            wi = self.wis_dict[current_index]
            li = self.lis_dict[current_index]
            
            if pairwise_features is not None:
                features = pairwise_features[current_index]
            else:
                features = None
            
            try:
                wi_new, li_new, gmitre, hinge_loss = self.fit_1_example_bcfw(current_example, ground,
                                                  current_label, n_examples, wi, li, C, features)
                self.wis_dict[current_index]=wi_new
                self.lis_dict[current_index]=li_new
                if verbose>0 and (i+1)%verbose_iters==0:
                    print("Iter: {:04d}".format(i+1),
                          "current example index: {:04d}".format(current_index),
                         "Group Mitre Loss: {:.10f}".format(gmitre),
                         "hinge Loss: {:.10f}".format(hinge_loss))
            
            except Exception as err:
                print("Exception: ", err)
                continue
        
        
    
    def predict(self, example, ground, features=None):
        """
        args:
          example(numpy ndarray, shape: [n_atoms, n_timesteps, n_in]
        return:
          predicted clustering: a list of lists denoting cluster
        """
        n_atoms = example.shape[0]
        rel_rec,rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
        if features is None:
            _, _ ,features = compute_sims(example, ground)
        #compute similarity matrix
        sims_values = np.matmul(features, self.w)
        sims_values = np.diag(sims_values)
        sims = np.matmul(rel_send.T, np.matmul(sims_values, rel_rec))
        best_clustering, current_score = greedy_approximate_best_clustering(sims)
        return best_clustering, current_score





