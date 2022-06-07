"""
Train NRI supervised way 
for pedestrian dataset
@author: z fang
"""

from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from data_utils import *
from models_NRI import *

from sknetwork.topology import get_connected_components
from sknetwork.clustering import Louvain
from scipy import sparse


parser = argparse.ArgumentParser()
parser.add_argument("--no-cuda", action="store_true", default=False, 
                    help="Disables CUDA training.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--no-seed", action="store_true", default=False,
                    help="don't use seed.")
parser.add_argument("--epochs", type=int, default=200, 
                    help="Number of epochs to train.")
parser.add_argument("--batch-size", type=int, default=128,
                    help="Number of samples per batch.")
parser.add_argument("--lr", type=float, default=0.005,
                    help="Initial learning rate.")
parser.add_argument("--encoder-hidden", type=int, default=128,
                    help="Number of hidden units.")
parser.add_argument("--encoder", type=str, default="wavenet",
                    help="Type of encoder model.")
parser.add_argument("--no-factor", action="store_true", default=False,
                    help="Disables factor graph model.")
parser.add_argument("--suffix", type=str, default="ETH",
                    help="Suffix for training data ")
parser.add_argument("--split", type=str, default="split00",
                    help="Split of the dataset.")
parser.add_argument("--use-motion", action="store_true", default=False,
                    help="use increments")
parser.add_argument("--encoder-dropout", type=float, default=0.3,
                    help="Dropout rate (1-keep probability).")
parser.add_argument("--save-folder", type=str, default="logs/nripedsu",
                    help="Where to save the trained model, leave empty to not save anything.")
parser.add_argument("--load-folder", type=str, default='', 
                    help="Where to load the trained model.")
parser.add_argument("--edge-types", type=int, default=2,
                    help="The number of edge types to infer.")
parser.add_argument("--dims", type=int, default=2,
                    help="The number of feature dimensions.")
parser.add_argument("--kernel-size", type=int, default=5,
                    help="Kernel size of WavenetNRI Encoder")

parser.add_argument("--depth", type=int, default=1,
                    help="depth of Wavenet CNN res blocks.")

parser.add_argument("--use-focal", action="store_true", default=False,
                    help="use focal loss.")

parser.add_argument("--timesteps", type=int, default=15,
                    help="The number of time steps per sample.")
parser.add_argument("--lr-decay", type=int, default=100,
                    help="After how epochs to decay LR factor of gamma.")
parser.add_argument("--gamma", type=float, default=0.5,
                    help="LR decay factor.")

parser.add_argument("--group-weight", type=float, default=0.5,
                    help="group weight.")
parser.add_argument("--ng-weight", type=float, default=0.5,
                    help="Non-group weight.")

parser.add_argument("--grecall-weight", type=float, default=0.65,
                    help="group recall.")



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)



if not args.no_seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        

log = None
#Save model and meta-data
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = "{}/{}_{}_{}/".format(args.save_folder,args.encoder, timestamp, args.suffix+args.split)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, "metadata.pkl")
    encoder_file = os.path.join(save_folder, "nri_encoder.pt")
    
    log_file = os.path.join(save_folder, "log.txt")
    log = open(log_file, 'w')
    pickle.dump({"args":args}, open(meta_file, 'wb'))
    
else:
    print("WARNING: No save_folder provided!"+
          "Testing (within this script) will throw an error.")
    

#Load data
data_folder = os.path.join("data/pedestrian/", args.suffix)
data_folder = os.path.join(data_folder, args.split)


with open(os.path.join(data_folder, "tensors_train.pkl"), 'rb') as f:
    examples_train = pickle.load(f)
with open(os.path.join(data_folder, "labels_train.pkl"), 'rb') as f:
    labels_train = pickle.load(f)
with open(os.path.join(data_folder, "tensors_valid.pkl"), 'rb') as f:
    examples_valid = pickle.load(f)
with open(os.path.join(data_folder, "labels_valid.pkl"), 'rb') as f:
    labels_valid = pickle.load(f)
with open(os.path.join(data_folder, "tensors_test.pkl"),'rb') as f:
    examples_test = pickle.load(f)
with open(os.path.join(data_folder, "labels_test.pkl"), 'rb') as f:
    labels_test = pickle.load(f)





if args.encoder == "mlp":
    encoder = MLPEncoder(args.timesteps*args.dims, args.encoder_hidden, 
                         args.edge_types, args.encoder_dropout, args.factor)

elif args.encoder == "cnn":
    encoder = CNNEncoder(args.dims, args.encoder_hidden, args.edge_types, 
                         args.encoder_dropout, args.factor, use_motion=args.use_motion)
    
elif args.encoder == "cnnsym":
    encoder = CNNEncoderSym(args.dims, args.encoder_hidden, args.edge_types,
                        do_prob=args.encoder_dropout, factor=args.factor,
                        use_motion=args.use_motion)

elif args.encoder == "rescnn":
    encoder = ResCausalCNNEncoder(args.dims, args.encoder_hidden, args.edge_types,
                                  do_prob=args.encoder_dropout, factor=args.factor,
                                  use_motion=args.use_motion)
    
elif args.encoder == "wavenet":
    encoder = WavenetEncoder(args.dims, args.encoder_hidden, args.edge_types, 
                             kernel_size = args.kernel_size, depth=args.depth,
                             do_prob=args.encoder_dropout, factor=args.factor,
                             use_motion=args.use_motion)
    
elif args.encoder=="wavenetraw":
    encoder = WavenetEncoderRaw(args.dims, args.encoder_hidden, args.edge_types,
                        do_prob=args.encoder_dropout, factor=args.factor,
                        use_motion=False)
    
elif args.encoder=="waveneteuc":
    encoder = WavenetEncoderEuc(args.dims, args.encoder_hidden, args.edge_types,
                        do_prob=args.encoder_dropout, factor=args.factor,
                        use_motion=args.use_motion)

elif args.encoder=="wavenetsym":
    encoder = WavenetEncoderSym(args.dims, args.encoder_hidden, args.edge_types,
                        do_prob=args.encoder_dropout, factor=args.factor,
                        use_motion=args.use_motion)


cross_entropy_weight = torch.tensor([args.ng_weight, args.group_weight])    

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, "nri_encoder.pt")
    encoder.load_state_dict(torch.load(encoder_file))
    args.save_folder = False
    
    
if args.cuda:
    encoder.cuda()
    cross_entropy_weight = cross_entropy_weight.cuda()
    

#optimizer = optim.Adam(list(encoder.parameters()),lr=args.lr)
optimizer = optim.SGD(list(encoder.parameters()), lr=args.lr, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, gamma=args.gamma)



def train(epoch, best_val_recall):
    t = time.time()
    loss_train = []
    acc_train = []
    gp_train = []
    ngp_train = []
    gr_train = []
    ngr_train = []
    loss_val = []
    acc_val = []
    gp_val = []
    ngp_val = []
    gr_val = []
    ngr_val = []
    F1_val = []
    recall_val = []
    
    encoder.train()
    
    training_indices = np.arange(len(examples_train))
    np.random.shuffle(training_indices)
    
    optimizer.zero_grad()
    idx_count = 0
    accumulation_steps = min(args.batch_size, len(examples_train)) #Initialization of accumulation steps
    
    for idx in training_indices:
        
        example = examples_train[idx]
        label = labels_train[idx]
        #add batch dimension
        example = example.unsqueeze(0)
        label = label.unsqueeze(0)
        num_atoms = example.size(1) #get number of atoms
        rel_rec, rel_send = create_edgeNode_relation(num_atoms, self_loops=False)
        
        if args.cuda:
            example = example.cuda()
            label = label.cuda()
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            
        example = example.float()
        logits = encoder(example, rel_rec, rel_send)
        #shape: [batch_size, n_edges, n_edgetypes]
        
        output = logits.view(logits.size(0)*logits.size(1),-1)
        target = label.view(-1)
        
        if args.use_focal:
            loss = focal_loss(output, target.long(), weight=cross_entropy_weight)
        else:
            loss = F.cross_entropy(output, target.long(), weight=cross_entropy_weight)
            
        loss_train.append(loss.item())
        loss = loss/accumulation_steps #average by dividing accumulation steps
        loss.backward()
        idx_count+=1
        
        if idx_count%args.batch_size==0 or idx_count==len(examples_train):
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accumulation_steps = min(args.batch_size, len(examples_train)-idx_count)
        
        
    
        acc = edge_accuracy(logits, label)
        acc_train.append(acc)
        gp, ngp = edge_precision(logits, label)
        gp_train.append(gp)
        ngp_train.append(ngp)
        
        gr,ngr = edge_recall(logits, label)
        gr_train.append(gr)
        ngr_train.append(ngr)
        
        
    
    
    encoder.eval()
    
    valid_indices = np.arange(len(examples_valid))
    
    with torch.no_grad():
        for idx in valid_indices:
            example = examples_valid[idx]
            label = labels_valid[idx]
            example = example.unsqueeze(0)
            label = label.unsqueeze(0)
            num_atoms = example.size(1)
            rel_rec, rel_send = create_edgeNode_relation(num_atoms, self_loops=False)
            
            if args.cuda:
                example = example.cuda()
                label = label.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            
            example = example.float()
            logits = encoder(example, rel_rec, rel_send)
            
            output = logits.view(logits.size(0)*logits.size(1),-1)
            target = label.view(-1)
            
            if args.use_focal:
                loss_current = focal_loss(output, target.long(), weight=cross_entropy_weight)
            else:
                loss_current = F.cross_entropy(output, target.long(), weight=cross_entropy_weight)
            
            #move tensors back to cpu
            example = example.cpu()
            rel_rec, rel_send = rel_rec.cpu(), rel_send.cpu()
            
            acc = edge_accuracy(logits, label)
            acc_val.append(acc)
            gp, ngp = edge_precision(logits, label)
            gp_val.append(gp)
            ngp_val.append(ngp)
            
            gr,ngr = edge_recall(logits, label)
            gr_val.append(gr)
            ngr_val.append(ngr)
            
            loss_val.append(loss_current.item())
            
            if gr==0 or gp==0:
                F1_g = 0.
            else:
                F1_g = 2*(gr*gp)/(gr+gp)
                
            if ngr==0 or ngp==0:
                F1_ng = 0.
            else:
                F1_ng = 2*(ngr*ngp)/(ngr+ngp)
                
            #F1 = args.group_weight*F1_g+(1-args.group_weight)*F1_ng
            
            ave_recall = args.grecall_weight*gr+(1-args.grecall_weight)*ngr
            recall_val.append(ave_recall)
                
            F1_val.append(F1_g)
            
    
    print("Epoch: {:04d}".format(epoch),
          "loss_train: {:.10f}".format(np.mean(loss_train)),
          "acc_train: {:.10f}".format(np.mean(acc_train)),
          "gp_train: {:.10f}".format(np.mean(gp_train)),
          "ngp_train: {:.10f}".format(np.mean(ngp_train)),
          "gr_train: {:.10f}".format(np.mean(gr_train)),
          "ngr_train: {:.10f}".format(np.mean(ngr_train)),
          "loss_val: {:.10f}".format(np.mean(loss_val)),
          "acc_val: {:.10f}".format(np.mean(acc_val)),
          "gp_val: {:.10f}".format(np.mean(gp_val)),
          "ngp_val: {:.10f}".format(np.mean(ngp_val)),
          "gr_val: {:.10f}".format(np.mean(gr_val)),
          "ngr_val: {:.10f}".format(np.mean(ngr_val)),
          "F1_val: {:.10f}".format(np.mean(F1_val)),
          "recall_val: {:.10f}".format(np.mean(recall_val)))
    if args.save_folder and np.mean(recall_val) > best_val_recall:
        torch.save(encoder, encoder_file)
        print("Best model so far, saving...")
        print("Epoch: {:04d}".format(epoch),
              "loss_train: {:.10f}".format(np.mean(loss_train)),
              "acc_train: {:.10f}".format(np.mean(acc_train)),
              "gp_train: {:.10f}".format(np.mean(gp_train)),
              "ngp_train: {:.10f}".format(np.mean(ngp_train)),
              "gr_train: {:.10f}".format(np.mean(gr_train)),
              "ngr_train: {:.10f}".format(np.mean(ngr_train)),
              "loss_val: {:.10f}".format(np.mean(loss_val)),
              "acc_val: {:.10f}".format(np.mean(acc_val)),
              "gp_val: {:.10f}".format(np.mean(gp_val)),
              "ngp_val: {:.10f}".format(np.mean(ngp_val)),
              "gr_val: {:.10f}".format(np.mean(gr_val)),
              "ngr_val: {:.10f}".format(np.mean(ngr_val)),
              "F1_val: {:.10f}".format(np.mean(F1_val)),
              "recall_val: {:.10f}".format(np.mean(recall_val)),
              file=log)
        log.flush()
        
    return np.mean(recall_val)



def test():
    t = time.time()
    loss_test = []
    acc_test = []
    gp_test = []
    ngp_test = []
    gr_test = []
    ngr_test = []
    
    encoder = torch.load(encoder_file)
    encoder.eval()
    
    test_indices = np.arange(len(examples_test))
    
    with torch.no_grad():
        for idx in test_indices:
            example = examples_test[idx]
            label = labels_test[idx]
            example = example.unsqueeze(0)
            label = label.unsqueeze(0)
            num_atoms = example.size(1) #get number of atoms
            rel_rec, rel_send = create_edgeNode_relation(num_atoms, self_loops=False)
            if args.cuda:
                example = example.cuda()
                label = label.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            example = example.float()
            logits = encoder(example, rel_rec, rel_send)
            
            output = logits.view(logits.size(0)*logits.size(1),-1)
            target = label.view(-1)
            
            acc = edge_accuracy(logits, label)
            acc_test.append(acc)
            gp, ngp = edge_precision(logits, label)
            gp_test.append(gp)
            ngp_test.append(ngp)
            
            gr,ngr = edge_recall(logits, label)
            gr_test.append(gr)
            ngr_test.append(ngr)
            
    
    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('acc_test: {:.10f}'.format(np.mean(acc_test)),
          "gp_test: {:.10f}".format(np.mean(gp_test)),
          "ngp_test: {:.10f}".format(np.mean(ngp_test)),
          "gr_test: {:.10f}".format(np.mean(gr_test)),
          "ngr_test: {:.10f}".format(np.mean(ngr_test))
          )
    
    

def test_gmitre():
    """
    test group mitre recall and precision   
    """
    louvain = Louvain()
    
    encoder = torch.load(encoder_file)
    encoder.eval()    
    test_indices = np.arange(len(examples_test))
    

    gIDs = []
    predicted_gr = []
    
    precision_all = []
    recall_all = []
    F1_all = []
    
    
    with torch.no_grad():
        for idx in test_indices:
            example = examples_test[idx]
            label = labels_test[idx] #shape: [n_edges]
            example = example.unsqueeze(0) #shape: [1, n_atoms, n_timesteps, n_in]
            n_atoms = example.size(1)
            rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
            rel_rec, rel_send = rel_rec.float(), rel_send.float()
            example = example.float()
            
            label = torch.diag_embed(label) #shape: [n_edges, n_edges]
            label = label.float()
            label_converted = torch.matmul(rel_send.t(), 
                                           torch.matmul(label, rel_rec))
            label_converted = label_converted.cpu().detach().numpy()
            #shape: [n_atoms, n_atoms]
            
            if label_converted.sum()==0:
                gID = list(range(label_converted.shape[1]))
                gIDs.append(gID)
            else:
                gID = list(get_connected_components(label_converted))
                gIDs.append(gID)
            
            if args.cuda:
                example = example.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
                
            Z = encoder(example, rel_rec, rel_send)
            Z = F.softmax(Z, dim=-1)
            #shape: [1, n_edges, 2]
            
            group_prob = Z[:,:,1] #shape: [1, n_edges]
            group_prob = group_prob.squeeze(0) #shape: [n_edges]
            group_prob = torch.diag_embed(group_prob) #shape: [n_edges, n_edges]
            group_prob = torch.matmul(rel_send.t(), torch.matmul(group_prob, rel_rec))
            #shape: [n_atoms, n_atoms]
            group_prob = 0.5*(group_prob+group_prob.T)
            group_prob = (group_prob>0.5).int()
            group_prob = group_prob.cpu().detach().numpy()
            
            if group_prob.sum()==0:
                pred_gIDs = np.arange(n_atoms)
            else:
                group_prob = sparse.csr_matrix(group_prob)
                pred_gIDs = louvain.fit_transform(group_prob)
                
            predicted_gr.append(pred_gIDs)
            
            recall, precision, F1 = compute_groupMitre_labels(gID, pred_gIDs)
            
            recall_all.append(recall)
            precision_all.append(precision)
            F1_all.append(F1)
            
        average_recall = np.mean(recall_all)
        average_precision = np.mean(precision_all)
        average_F1 = np.mean(F1_all)
        
    print("Average recall: ", average_recall)
    print("Average precision: ", average_precision)
    print("Average F1: ", average_F1)
    
    print("Average recall: {:.10f}".format(average_recall),
          "Average precision: {:.10f}".format(average_precision),
          "Average_F1: {:.10f}".format(average_F1),
          file=log)
            
            
        
    
    
#Train model

t_total = time.time()
best_val_recall = 0.
best_epoch = 0

for epoch in range(args.epochs):
    val_recall = train(epoch, best_val_recall)
    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_epoch = epoch
        
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))

test()

test_gmitre()

            
            
                
            
    
            
            
    
    
    
    
        










