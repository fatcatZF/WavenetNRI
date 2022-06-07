"""Train NRI supervised way"""

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
parser.add_argument("--lr", type=float, default=0.0005,
                    help="Initial learning rate.")
parser.add_argument("--encoder-hidden", type=int, default=256,
                    help="Number of hidden units.")
parser.add_argument("--num-atoms", type=int, default=10,
                    help="Number of atoms.")
parser.add_argument("--encoder", type=str, default="wavenet",
                    help="Type of encoder model.")
parser.add_argument("--no-factor", action="store_true", default=False,
                    help="Disables factor graph model.")
parser.add_argument("--suffix", type=str, default="_static_10_3_3",
                    help="Suffix for training data ")
parser.add_argument("--use-motion", action="store_true", default=False,
                    help="use increments")
parser.add_argument("--encoder-dropout", type=float, default=0.3,
                    help="Dropout rate (1-keep probability).")
parser.add_argument("--save-folder", type=str, default="logs/nrisu",
                    help="Where to save the trained model, leave empty to not save anything.")
parser.add_argument("--load-folder", type=str, default='', 
                    help="Where to load the trained model.")
parser.add_argument("--edge-types", type=int, default=2,
                    help="The number of edge types to infer.")
parser.add_argument("--dims", type=int, default=4,
                    help="The number of feature dimensions.")
parser.add_argument("--timesteps", type=int, default=49,
                    help="The number of time steps per sample.")
parser.add_argument("--lr-decay", type=int, default=200,
                    help="After how epochs to decay LR factor of gamma.")
parser.add_argument("--gamma", type=float, default=0.5,
                    help="LR decay factor.")
parser.add_argument("--group-weight", type=float, default=0.5,
                    help="group weight.")
parser.add_argument("--ng-weight", type=float, default=0.5,
                    help="Non-group weight")
parser.add_argument("--gweight-auto", action="store_true", default=False,
                    help = "automatically determine group/non-group weights.")


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
#Save model and meta-data. Always saves in a new folder
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = "{}/su_{}_{}_{}".format(args.save_folder, args.encoder, args.suffix ,timestamp)
    if args.use_motion:
        save_folder += "_use_motion"
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, "metadata.pkl")
    encoder_file = os.path.join(save_folder, "nri_encoder.pt")
    
    log_file = os.path.join(save_folder, "log.txt")
    log = open(log_file, 'w')
    pickle.dump({"args":args}, open(meta_file, 'wb'))
    
else:
    print("WARNING: No save_folder provided!"+
          "Testing (within this script) will throw an error.")
    

train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_spring_sim(
    args.batch_size, args.suffix)

print("Number of training examples: ", len(train_loader.dataset))
print("Number of validation examples: ", len(valid_loader.dataset))
print("Number of test examples: ", len(test_loader.dataset))


off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.from_numpy(rel_rec)
rel_send = torch.from_numpy(rel_send)

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

    
if args.load_folder:
    encoder_file = os.path.join(args.load_folder, "nri_encoder.pt")
    encoder.load_state_dict(torch.load(encoder_file))
    args.save_folder = False
    

triu_indices = get_triu_offdiag_indices(args.num_atoms)
tril_indices = get_tril_offdiag_indices(args.num_atoms)



def compute_label_weights(train_loader, eps=1e-4):
    total_gr = 0
    total_ngr = 0
    for batch_idx, (data, relations) in enumerate(train_loader):
        relations = relations.float()
        relations, relations_masked = relations[:,:,0], relations[:,:,1]
        n_gr = relations.sum()
        n_ngr = (relations==0).sum()
        total_gr += n_gr
        total_ngr += n_ngr
    group_weight = (total_gr+total_ngr)/(2*total_gr+eps)
    ng_weight = (total_gr+total_ngr)/(2*total_ngr+eps)
    
    return ng_weight, group_weight


if args.gweight_auto:
    ng_weight, g_weight = compute_label_weights(train_loader)
    print("Group Label Weight: ", g_weight.item())
    print("Non-Group Label Weight: ", ng_weight.item())
    cross_entropy_weight = torch.tensor([ng_weight.item(), g_weight.item()])
else:
    cross_entropy_weight = torch.tensor([args.ng_weight, args.group_weight])
    




if args.cuda:
    encoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()
    cross_entropy_weight = cross_entropy_weight.cuda()
    

optimizer = optim.Adam(list(encoder.parameters()),lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)


print(args, file=log)







def train(epoch, best_val_F1):
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
    
    encoder.train()
    
    for batch_idx, (data, relations) in enumerate(train_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        relations = relations.float()
        relations, relations_masked = relations[:,:,0], relations[:,:,1]
        data = data.float()
        optimizer.zero_grad()
        logits = encoder(data, rel_rec, rel_send)
        #logits shape: [batch_size, n_edges, edge_types]
        
        #Flatten batch dim
        output = logits.view(logits.size(0)*logits.size(1),-1)
        target = relations.view(-1)
        
        loss = F.cross_entropy(output, target.long(), weight=cross_entropy_weight)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        acc = edge_accuracy(logits, relations)
        acc_train.append(acc)
        gp, ngp = edge_precision(logits, relations) #Precision of group and non_group
        gp_train.append(gp)
        ngp_train.append(ngp)
        
        gr,ngr = edge_recall(logits, relations)
        gr_train.append(gr)
        ngr_train.append(ngr)
        
        loss_train.append(loss.item())
        
    
    encoder.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        relations = relations.float()
        relations, relations_masked = relations[:,:,0], relations[:,:,1]
        
        data = data.float()
        with torch.no_grad():
            logits = encoder(data, rel_rec, rel_send)
            #Shape: [batch_size, n_edges, n_edgetypes]
            
            #Flatten batch dim
            output = logits.view(logits.size(0)*logits.size(1),-1)
            target = relations.view(-1)
            loss = F.cross_entropy(output, target.long(), weight=cross_entropy_weight)
            
            acc = edge_accuracy(logits, relations)
            acc_val.append(acc)
            gp, ngp = edge_precision(logits, relations) #Precision of group and non_group
            gp_val.append(gp)
            ngp_val.append(ngp)
            
            gr,ngr = edge_recall(logits, relations)
            gr_val.append(gr)
            ngr_val.append(ngr)
            
            loss_val.append(loss.item())
            
            if gr==0 or gp==0:
                F1_g = 0
            else:
                F1_g = 2*(gr*gp)/(gr+gp)
                
            if ngr==0 or ngp==0:
                F1_ng = 0.
            else:
                F1_ng = 2*(ngr*ngp)/(ngr+ngp)
                
            F1 = args.group_weight*F1_g+(1-args.group_weight)*F1_ng
                
            
                
            F1_val.append(F1)
            
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
          "F1_val: {:.10f}".format(np.mean(F1_val)))
    if args.save_folder and np.mean(F1_val) > best_val_F1:
        #torch.save(encoder.state_dict(), encoder_file)
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
              file=log)
        log.flush()
        
    return np.mean(F1_val)





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
    #encoder.load_state_dict(torch.load(encoder_file))
    
    
    for batch_idx, (data, relations) in enumerate(test_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        relations = relations.float()
        
        logits = encoder(data, rel_rec, rel_send)
        
        acc = edge_accuracy(logits, relations)
        acc_test.append(acc)
        
        gp, ngp = edge_precision(logits, relations) #Precision of group and non_group
        gp_test.append(gp)
        ngp_test.append(ngp)
        
        gr,ngr = edge_recall(logits, relations)
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
    
    rel_rec, rel_send = create_edgeNode_relation(args.num_atoms, self_loops=False)
    rel_rec, rel_send = rel_rec.float(), rel_send.float()
    if args.cuda:
        rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
    
    encoder = torch.load(encoder_file)
    encoder.eval()    
    
    

    gIDs = []
    predicted_gr = []
    
    precision_all = []
    recall_all = []
    F1_all = []
    
    
    with torch.no_grad():
        for batch_idx, (data, relations) in enumerate(test_loader):
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()
                #data shape: [1, n_atoms, n_timesteps, n_in]
                #relations, shape: [1, n_edges]
            
            relations = relations.squeeze(0) #shape: [n_edges]
            label = torch.diag_embed(relations) #shape: [n_edges, n_edges]
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
                data = data.cuda()
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
                
            Z = encoder(data, rel_rec, rel_send)
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
                pred_gIDs = np.arange(args.num_atoms)
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
best_val_F1 = -1.
best_epoch = 0

for epoch in range(args.epochs):
    val_F1 = train(epoch, best_val_F1)
    if val_F1 > best_val_F1:
        best_val_F1 = val_F1
        best_epoch = epoch
        
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))


test()
test_gmitre()
        
    
    
        
            
    
    
            
            
            
            
    
        
    
    

    
    



































