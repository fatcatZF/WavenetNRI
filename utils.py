import numpy as np
import torch
import torch.nn as nn



def create_edgeNode_relation(num_nodes, self_loops=False):
    if self_loops:
        indices = np.ones([num_nodes, num_nodes])
    else:
        indices = np.ones([num_nodes, num_nodes]) - np.eye(num_nodes)
    rel_rec = np.array(encode_onehot(np.where(indices)[0]), dtype=np.float32)
    rel_send = np.array(encode_onehot(np.where(indices)[1]), dtype=np.float32)
    rel_rec = torch.from_numpy(rel_rec)
    rel_send = torch.from_numpy(rel_send)
    
    return rel_rec, rel_send




def normalize_graph(graph, add_self_loops=False):
    """
    args:
      graph: adjacency matrix; [batch_size, n_nodes, n_nodes]
              or [batch_size, n_timesteps, n_nodes, n_nodes]
    """
    num_nodes = graph.size(-1)
    if add_self_loops:
        if len(graph.size())==3:
            I = torch.eye(num_nodes).unsqueeze(0)
            I.expand(graph.size(0),I.size(1),I.size(2))
            graph += I
        else:
            I = torch.eye(num_nodes).unsqueeze(0).unsqueeze(0)
            I.expand(graph.size(0), graph.size(1), I.size(2), I.size(3))
            graph += I
    degree = graph.sum(-1) #shape:[batch_size, num_nodes]
    degree = 1./torch.sqrt(degree) #shape:[batch_size,num_nodes]
    degree[degree==torch.inf]=0 #convert infs to 0s
    degree = torch.diag_embed(degree) #shape:[batch_size,num_nodes, num_nodes]
    return degree@graph@degree




def symmetrize(A):
    """
    args:
        A: batches of Interaction matrices
          [batch_size, num_nodes, num_nodes]
    return: A_sym: symmetric version of A
    """
    AT = A.transpose(-1,-2)
    return 0.5*(A+AT)


def laplacian_smooth(A):
    """
    args:
      A: batches of adjacency matrices of symmetric Interactions
         size of A: [batch_size, num_edgeTypes, num_nodes, num_nodes]       
    return: A_norm = (D**-0.5)A(D**-0.5), where D is the diagonal matrix of A
    """
    I = torch.eye(A.size(-1))
    I = I.unsqueeze(0).unsqueeze(1)
    I = I.expand(A.size(0), A.size(1), I.size(2), I.size(3))
    #size: [batch_size, num_edgeTypes, num_atoms, num_atoms]
    A_p = A+I
    D_values = A_p.sum(-1) #Degree values; size: [batch_size, num_nodes]
    D_values_p = torch.pow(D_values, -0.5)
    D_p = torch.diag_embed(D_values_p) #size: [batch_size, num_nodes, num_nodes]
    return torch.matmul(D_p, torch.matmul(A_p, D_p)) 


def laplacian_sharpen(A):
    """
    args:
        A; batches of adjacency matrices corresponding to edge types
          size: [batch_size, num_edgeTypes, num_nodes, num_nodes]
    """
    I = torch.eye(A.size(-1))
    I = I.unsqueeze(0).unsqueeze(1)
    I = I.expand(A.size(0), A.size(1), I.size(2), I.size(3))
    #size: [batch_size, num_edgeTypes, num_atoms, num_atoms]
    Ap = 2*I-A
    D_values = A.sum(-1)+2 #shape: [batch_size, num_edgeTypes, num_atoms]
    D_values_p = torch.pow(D_values, -0.5)
    D_p = torch.diag_embed(D_values_p)
    
    return torch.matmul(D_p, torch.matmul(Ap, D_p)) 
    
    
    
    



def nll_gaussian(preds, target, variance, add_const=False):
    """
    loss function
    copied from https://github.com/ethanfetaya/NRI/blob/master/utils.py
    """
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))
    

def get_noise(shape, noise_type="gaussian"):
    """copied from https://github.com/huang-xx/STGAT/blob/master/STGAT/models.py"""
    if noise_type == "gaussian":
        return torch.randn(*shape)
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0)
    raise ValueError('Unrecognized noise type "%s"' % noise_type)



def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    #not tested yet
    kl_div = preds*(torch.log(preds+eps)-log_prior)
    return kl_div.sum()/(num_atoms*preds.size(0))



def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds*torch.log(preds+eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum()/(num_atoms*preds.size(0))


def kl_gaussian(mu, sigma):
    return -((0.5*(1+torch.log(sigma**2)-mu**2-sigma**2)).sum()/(mu.size(0)*mu.size(1)))


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices

def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices

def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices

def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()

def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()



def edge_accuracy(preds, target):
    """compute pairwise group accuracy"""
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct) / (target.size(0) * target.size(1))

def edge_accuracy_prob(preds, target, threshold=0.5):
    """compute pairwise accuracy based on prob
    args:
        preds:[batch_size, n_edges]
        target:[batch_size, n_edges]        
    """
    preds = (preds>threshold).int()
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return np.float(correct)/(target.size(0)*target.size(1))



def edge_precision(preds, target):
    """compute pairwise group/non-group recall"""
    _, preds = preds.max(-1)
    true_possitive = ((preds[target==1]==1).cpu().sum()).item()
    total_possitive = ((preds[preds==1]).cpu().sum()).item()
    if total_possitive==true_possitive:
        group_precision = 1
    true_negative = ((preds[target==0]==0).cpu().sum()).item()
    total_negative = ((preds[preds==0]==0).cpu().sum()).item()
    if total_negative==true_negative:
        non_group_precision = 1
    if total_possitive>0:
        group_precision = true_possitive/total_possitive
    if total_negative>0:
        non_group_precision = true_negative/total_negative
       
    #group_precision = ((target[preds==1]==1).cpu().sum())/preds[preds==1].cpu().sum()
    #non_group_precision = ((target[preds==0]==0).cpu().sum())/(preds[preds==0]==0).cpu().sum()
    return group_precision, non_group_precision

def edge_precision_prob(preds, target, threshold=0.7):
    """Compute pairwise group/non-group precision"""
    preds = (preds>threshold).int()
    true_possitive = ((preds[target==1]==1).cpu().sum()).item()
    total_possitive = ((preds[preds==1]).cpu().sum()).item()
    if total_possitive==true_possitive:
        group_precision = 1
    true_negative = ((preds[target==0]==0).cpu().sum()).item()
    total_negative = ((preds[preds==0]==0).cpu().sum()).item()
    if total_negative==true_negative:
        non_group_precision = 1
    if total_possitive>0:
        group_precision = true_possitive/total_possitive
    if total_negative>0:
        non_group_precision = true_negative/total_negative
       
    #group_precision = ((target[preds==1]==1).cpu().sum())/preds[preds==1].cpu().sum()
    #non_group_precision = ((target[preds==0]==0).cpu().sum())/(preds[preds==0]==0).cpu().sum()
    return group_precision, non_group_precision
    
    
    

def edge_recall(preds, target):
    """compute pairwise group/non-group recall"""
    _,preds = preds.max(-1)
    retrived_possitive = ((preds[target==1]==1).cpu().sum()).item()
    total_possitive = ((target[target==1]).cpu().sum()).item()
    retrived_negative = ((preds[target==0]==0).cpu().sum()).item()
    total_negative = ((target[target==0]==0).cpu().sum()).item()
    
    if retrived_possitive==total_possitive:
        group_recall = 1
    if retrived_negative==total_negative:
        non_group_recall = 1
        
    if total_possitive > 0:
        group_recall = retrived_possitive/total_possitive
    if total_negative > 0:
        non_group_recall = retrived_negative/total_negative
    
    #group_recall = ((preds[target==1]==1).cpu().sum())/(target[target==1]).cpu().sum()
    #non_group_recall = ((preds[target==0]==0).cpu().sum())/(target[target==0]==0).cpu().sum()
    return group_recall, non_group_recall


def edge_recall_prob(preds, target, threshold=0.7):
    preds = (preds>threshold).int()
    retrived_possitive = ((preds[target==1]==1).cpu().sum()).item()
    total_possitive = ((target[target==1]).cpu().sum()).item()
    retrived_negative = ((preds[target==0]==0).cpu().sum()).item()
    total_negative = ((target[target==0]==0).cpu().sum()).item()
    
    if retrived_possitive==total_possitive:
        group_recall = 1
    if retrived_negative==total_negative:
        non_group_recall = 1
        
    if total_possitive > 0:
        group_recall = retrived_possitive/total_possitive
    if total_negative > 0:
        non_group_recall = retrived_negative/total_negative
    
    #group_recall = ((preds[target==1]==1).cpu().sum())/(target[target==1]).cpu().sum()
    #non_group_recall = ((preds[target==0]==0).cpu().sum())/(target[target==0]==0).cpu().sum()
    return group_recall, non_group_recall

    


def indices_to_clusters(l):
    """
    args:
        l: indices of clusters, e.g.. [0,0,1,1]
    return: clusters, e.g. [(0,1),(2,3)]
    """
    d = dict()
    for i,v in enumerate(l):
        d[v] = d.get(v,[])
        d[v].append(i)
    clusters = list(d.values())
    return clusters


def compute_mitre(a, b):
    """
    compute mitre 
    more details: https://aclanthology.org/M95-1005.pdf
    args:
      a,b: list of groups; e.g. a=[[1.2],[3],[4]], b=[[1,2,3],[4]]
    Return: 
      mitreLoss a_b
      
    """
    total_m = 0 #total missing links
    total_c = 0 #total correct links
    for group_a in a:
        pa = 0 #partitions of group_a in b
        part_group = []#partition group
        size_a = len(group_a) #size of group a
        for element in group_a:
            for group_b in b:
                if element in group_b:
                    if part_group==group_b:
                        continue
                    else:
                        part_group = group_b
                        pa+=1
        total_c += size_a-1
        total_m += pa-1
        
    return (total_c-total_m)/total_c




def create_counterPart(a):
    """
    add fake counter parts for each agent
    args:
      a: list of groups; e.g. a=[[0,1],[2],[3,4]]
    """
    a_p = []
    for group in a:
        if len(group)==1:#singleton
            element = group[0]
            element_counter = -(element+1)#assume element is non-negative
            new_group = [element, element_counter]
            a_p.append(new_group)
        else:
            a_p.append(group)
            for element in group:
                element_counter = -(element+1)
                a_p.append([element_counter])
    return a_p




def compute_groupMitre(target, predict):
    """
    compute group mitre
    args: 
      target,predict: list of groups; [[0,1],[2],[3,4]]
    return: recall, precision, F1
    """
    #create fake counter agents
    target_p = create_counterPart(target)
    predict_p = create_counterPart(predict)
    recall = compute_mitre(target_p, predict_p)
    precision = compute_mitre(predict_p, target_p)
    if recall==0 or precision==0:
        F1 = 0
    else:
        F1 = 2*recall*precision/(recall+precision)
    return recall, precision, F1


def compute_gmitre_loss(target, predict):
    _,_, F1 = compute_groupMitre(target, predict)
    return 1-F1





def compute_groupMitre_labels(target, predict):
    """
    compute group mitre given indices
    args: target, predict: list of indices of groups
       e.g. [0,0,1,1]
    """
    target = indices_to_clusters(target)
    predict = indices_to_clusters(predict)
    recall, precision, F1 = compute_groupMitre(target, predict)
    return recall, precision, F1










class FocalLoss(nn.Module):
    """Implementation of Facal Loss"""
    def __init__(self, weight=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weighted_cs = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.cs = nn.CrossEntropyLoss(reduction="none")
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predicted, target):
        """
        predicted: [batch_size, n_classes]
        target: [batch_size]
        """
        pt = 1/torch.exp(self.cs(predicted,target))
        #shape: [batch_size]
        entropy_loss = self.weighted_cs(predicted, target)
        #shape: [batch_size]
        focal_loss = ((1-pt)**self.gamma)*entropy_loss
        #shape: [batch_size]
        if self.reduction =="none":
            return focal_loss
        elif self.reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss.sum()
        

def focal_loss(predicted, target, weight=None, gamma=2, reduction="mean"):
    focalLoss = FocalLoss(weight=weight, gamma=gamma, reduction=reduction)
    return focalLoss(predicted, target)
























