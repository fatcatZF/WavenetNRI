"""Train GD-GAN"""

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
from models_gdgan import *

from sknetwork.topology import get_connected_components
from sklearn.cluster import DBSCAN

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument("--no-seed", action="store_true", default=False,
                    help="don't use seed")

parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='Initial learning rate.')

parser.add_argument("--n-in", type=int, default=4, help="Input dimensions.")
parser.add_argument('--n-emb', type=int, default=16, help='Dimension of embedding')
parser.add_argument("--n-hid", type=int, default=32, 
                    help="Dimensions of hidden states.")
parser.add_argument("--n-noise", type=int, default=4, 
                    help="Dimensions of noise.")


parser.add_argument('--suffix', type=str, default='_static_10_3_3',
                    help='Suffix for training data (e.g. "_charged".')

parser.add_argument('--save-folder', type=str, default='logs/gdgan',
                    help='Where to save the trained model, leave empty to not save anything.')

parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')

parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')


parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument("--sc-weight", type=float, default=0.2,
                    help="Sparse Constraint Weight.")

parser.add_argument('--var', type=float, default=0.1,
                    help='Output variance.')

parser.add_argument("--teaching", action="store_true", default=True, 
                    help="Whether use teaching force.")


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)


if not args.no_seed:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        

#Save model and meta-data
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/{}_{}/'.format(args.save_folder, args.suffix, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    generator_file = os.path.join(save_folder, 'generator.pt')    
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')
    pickle.dump({'args': args}, open(meta_file, "wb"))
    
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")
    

train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_spring_sim(
    args.batch_size, args.suffix, normalize=False)

#off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
#rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
#rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
#rel_rec = torch.from_numpy(rel_rec)
#rel_send = torch.from_numpy(rel_send)



generator = LSTMGenerator(args.n_in, args.n_emb, args.n_hid, args.n_noise)

if args.cuda:
    generator = generator.cuda()
    

optimizer = optim.Adam(list(generator.parameters()), lr=args.lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr, gamma=args.gamma)


def train_generator():
    loss_train = []
    mse_train = []
    loss_val = []
    mse_val = []
    sc_train = []
    sc_val = []
    generator.train()
    
    for batch_idx, (data, relations) in enumerate(train_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        relations = relations.float()
        relations, relations_masked = relations[:,:,0], relations[:,:,1]
        data = data.float()
        #shape:[n_sims, n_atoms, n_timesteps, n_in]
        optimizer.zero_grad()
        batch_size = data.size(0)
        n_atoms = data.size(1)
        n_timesteps = data.size(2)
        T_obs = int(n_timesteps/2)
        rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
        rel_rec_t, rel_send_t = create_edgeNode_relation(T_obs, self_loops=True)
        
        if args.cuda:
            rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
            rel_rec_t, rel_send_t = rel_rec_t.cuda(), rel_send_t.cuda()
            
        #generate random noise
        noise = torch.randn(batch_size, n_atoms, args.n_noise)
        if args.cuda:
            noise = noise.cuda()
            
        #generate fake examples and hidden states
        x_fake, hs = generator(data, noise, rel_rec, rel_send, rel_rec_t, rel_send_t)
        #x_fake: [batch_size, n_atoms, n_timesteps, n_in]
        #hs: [batch_size, n_atoms, n_h*pred_timesteps]
        
        #compute L1 norm of hidden state hs
        loss_sc = args.sc_weight*(torch.norm(hs, p=1, dim=-1).sum())/(hs.size(0)*hs.size(1))
        sc_train.append(loss_sc.item())
        
        #compute reconstruction error
        out = x_fake[:,:,T_obs+1:,:] #shape: [batch_size, n_atoms, n_timesteps-T_obs, n_in]
        target = data[:,:,T_obs+1:,:] 
        loss_nll = nll_gaussian(out, target, args.var)
        loss_mse = F.mse_loss(out, target)
        
        loss = loss_nll+loss_mse
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        mse_train.append(loss_mse.item())
        loss_train.append(loss_nll.item())
        
        
    
    generator.eval()
    
    with torch.no_grad():
        for batch_idx, (data, relations) in enumerate(valid_loader):
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()
            relations = relations.float()
            relations, relations_masked = relations[:,:,0], relations[:,:,1]
            
            data = data.float()
            #shape: [batch_size, n_atoms, n_timesteps, n_in]
            batch_size = data.size(0)
            n_atoms = data.size(1)
            n_timesteps = data.size(2)
            T_obs = int(n_timesteps/2)
            rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
            rel_rec_t, rel_send_t = create_edgeNode_relation(T_obs, self_loops=True)
            
            if args.cuda:
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
                rel_rec_t, rel_send_t = rel_rec_t.cuda(), rel_send_t.cuda()
                
            #generate random noise
            noise = torch.randn(batch_size, n_atoms, args.n_noise)
            if args.cuda:
                noise = noise.cuda()
                
            #generate fake examples and hidden states
            x_fake, hs = generator(data, noise, rel_rec, rel_send, rel_rec_t, rel_send_t)
            #x_fake: [batch_size, n_atoms, n_timesteps, n_in]
            #hs: [batch_size, n_atoms, n_h*pred_timesteps]
            
            #compute L1 norm of hidden state hs
            loss_sc = args.sc_weight*(torch.norm(hs, p=1, dim=-1).sum())/(hs.size(0)*hs.size(1))
            sc_val.append(loss_sc.item())
            
            #compute reconstruction error
            out = x_fake[:,:,T_obs+1:,:] #shape: [batch_size, n_atoms, n_timesteps-T_obs, n_in]
            target = data[:,:,T_obs+1:,:] 
            loss_nll = nll_gaussian(out, target, args.var)
            loss_mse = F.mse_loss(out, target)
            
            mse_val.append(loss_mse.item())
            loss_val.append(loss_nll.item())
            
    return np.mean(loss_train), np.mean(mse_train), np.mean(sc_train), np.mean(loss_val), np.mean(mse_val), np.mean(sc_val)
            
            
            




def test_generator():
    loss_test = []
    mse_test = []
    sc_test = []
    
    generator = torch.load(generator_file)
    generator.eval()
    
    with torch.no_grad():
        for batch_idx, (data, relations) in enumerate(test_loader):
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()
            relations = relations.float()
            
            data = data.float()
            #[batch_size, n_atoms, n_timesteps, n_in]
            batch_size = data.size(0)
            n_atoms = data.size(1)
            n_timesteps = data.size(2)
            T_obs = int(n_timesteps/2)
            rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
            rel_rec_t, rel_send_t = create_edgeNode_relation(T_obs, self_loops=True)
            
            if args.cuda:
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
                rel_rec_t, rel_send_t = rel_rec_t.cuda(), rel_send_t.cuda()
                
            #generate random noise
            noise = torch.randn(batch_size, n_atoms, args.n_noise)
            if args.cuda:
                noise = noise.cuda()
                
            #generate fake examples and hidden states
            x_fake, hs = generator(data, noise, rel_rec, rel_send, rel_rec_t, rel_send_t)
            #x_fake: [batch_size, n_atoms, n_timesteps, n_in]
            #hs: [batch_size, n_atoms, n_h*pred_timesteps]
            
            #compute L1 norm of hidden state hs
            loss_sc = args.sc_weight*(torch.norm(hs, p=1, dim=-1).sum())/(hs.size(0)*hs.size(1))
            sc_test.append(loss_sc.item())
            
            
            #compute reconstruction error
            out = x_fake[:,:,T_obs+1:,:] #shape: [batch_size, n_atoms, n_timesteps-T_obs, n_in]
            target = data[:,:,T_obs+1:,:] 
            loss_nll = nll_gaussian(out, target, args.var)
            loss_mse = F.mse_loss(out, target)
            
            mse_test.append(loss_mse.item())
            loss_test.append(loss_nll.item())
            
        print("loss_test: ", np.mean(loss_test),
              "mse_test: ", np.mean(mse_test),
              "sc_test: ", np.mean(sc_test))
        
    return np.mean(loss_test), np.mean(mse_test), np.mean(sc_test)



def train(epoch, best_val_loss):
    loss_train, mse_train, sc_train, loss_val, mse_val, sc_val = train_generator()
    print("Epoch: {:04d}".format(epoch+1),
          "loss_train: {:.10f}".format(loss_train),
          "mse_train: {:.10f}".format(mse_train),
          "sc_train: {:.10f}".format(sc_train),
          "loss_val: {:.10f}".format(loss_val),
          "mse_val: {:.10f}".format(mse_val),
          "sc_val: {:.10f}".format(sc_val))
    if args.save_folder and loss_val+sc_val < best_val_loss:
        torch.save(generator, generator_file)
        print("Best model so far, saving...")
        print("Epoch: {:04d}".format(epoch+1),
              "loss_train: {:.10f}".format(loss_train),
              "mse_train: {:.10f}".format(mse_train),
              "sc_train: {:.10f}".format(sc_train),
              "loss_val: {:.10f}".format(loss_val),
              "mse_val: {:.10f}".format(mse_val),
              "sc_val: {:.10f}".format(sc_val),
              file=log)
        log.flush()
    
    return loss_train, mse_train, sc_train, loss_val, mse_val, sc_val




def test_gmitre():
    generator = torch.load(generator_file)
    generator.eval()
    hiddens = []
    gIDs = []
    
    with torch.no_grad():
        for batch_idx, (data, relations) in enumerate(test_loader):
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()
            
                
            data = data.float()
            #[1, n_atoms, n_timesteps, n_in]
            batch_size = data.size(0)
            n_atoms = data.size(1)
            n_timesteps = data.size(2)
            T_obs = int(n_timesteps/2)
            
            rel_rec, rel_send = create_edgeNode_relation(n_atoms, self_loops=False)
            rel_rec_t, rel_send_t = create_edgeNode_relation(T_obs, self_loops=True)
            if args.cuda:
                rel_rec, rel_send = rel_rec.cuda(), rel_send.cuda()
                rel_rec_t, rel_send_t = rel_rec_t.cuda(), rel_send_t.cuda()
            
            relations = relations.float()
            #data shape: [1, n_atoms, n_timesteps, n_in]
            #relations, shape: [1, n_edges]
            relations = relations.squeeze(0) #shape: [n_edges]
            label = torch.diag_embed(relations) #shape: [n_edges]
            label = label.float()
            label_converted = torch.matmul(rel_send.t(), 
                                           torch.matmul(label, rel_rec))
            label_converted = label_converted.cpu().detach().numpy()
            
            if label_converted.sum()==0:
                gID = list(range(label_converted.shape[1]))
                gIDs.append(gID)
            else:
                gID = list(get_connected_components(label_converted))
                gIDs.append(gID)
            
        
                
            #generate random noise
            noise = torch.randn(batch_size, n_atoms, args.n_noise)
            if args.cuda:
                noise = noise.cuda()
                
            #generate fake examples and hidden states
            x_fake, hs = generator(data, noise, rel_rec, rel_send, rel_rec_t, rel_send_t)
            #x_fake: [1, n_atoms, n_timesteps, n_in]
            #hs: [1, n_atoms, n_h*pred_timesteps]
            hs = hs.squeeze(0)
            #shape: [n_atoms, n_h*pred_timesteps]
            hs_numpy = hs.cpu().detach().numpy()
            hiddens.append(hs_numpy)
            
        epss = np.arange(0.01, 3.01, 0.005)
        max_average_recall = 0.
        max_average_precision = 0.
        max_average_F1 = 0.
            
        for eps in epss:
            precision_test = []
            recall_test = []
            F1_test = []
            for i in range(len(hiddens)):
                target = gIDs[i]
                hidden = hiddens[i]
                clustering = DBSCAN(eps=eps, min_samples=1).fit(hidden)
                predicted_labels = clustering.labels_
                recall, precision, F1 = compute_groupMitre_labels(target, predicted_labels)
                recall_test.append(recall)
                precision_test.append(precision)
                F1_test.append(F1)
            average_recall = np.mean(recall_test)
            average_precision = np.mean(precision_test)
            average_F1 = np.mean(F1_test)
            
            if average_F1 > max_average_F1:
                max_average_F1 = average_F1
                max_average_recall = average_recall
                max_average_precision = average_precision
                
        print("Average Recall: ", max_average_recall)
        print("Average Precision: ", max_average_precision)
        print("Average F1: ", max_average_F1)
        
        print("Average recall: {:.10f}".format(max_average_recall),
              "Average precision: {:.10f}".format(max_average_precision),
              "Average_F1: {:.10f}".format(max_average_F1),
              file=log)
                






#Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    loss_train, mse_train, sc_train, loss_val, mse_val, sc_val = train(epoch, best_val_loss)
    if loss_val < best_val_loss:
        best_val_loss = loss_val+sc_val
        best_epoch = epoch+1
        

print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()
    
test_generator()

test_gmitre()
            
            
            
            
            
            
            
                
            
            
        
            
            
            
            
            
            
            
                
            
            
            
            
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    








