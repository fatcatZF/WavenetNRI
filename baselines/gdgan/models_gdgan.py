"""
GDGAN models
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import math




class LSTMCell(nn.Module):
    def __init__(self, n_in, n_hid):
        """
        args:
          n_in: dimensions of input features
          n_hid: hidden dimensions
        """
        super(LSTMCell, self).__init__()
        self.lstm_cell = nn.LSTMCell(n_in, n_hid)
        self.n_hid = n_hid
        
    def forward(self, inputs, hc=None):
        """
        args:
          inputs, shape: [batch_size, num_atoms, num_features]
          hc, a tuple include hidden state and cell state
        """
        x = inputs.view(inputs.size(0)*inputs.size(1),-1)
        #shape: [total_atoms, num_features]
        if hc is None:
            hidden = torch.zeros(x.size(0),self.n_hid)
            cell = torch.zeros_like(hidden)
            if inputs.is_cuda:
                hidden = hidden.cuda()
                cell = cell.cuda()
            hc = (hidden, cell)
        h, c = self.lstm_cell(x, hc)
        #shape: [batch_size*n_atoms, n_hid]
    
        return h, c
    
    
   
class LSTMEncoder(nn.Module):
    """LSTM Encoder"""
    def __init__(self, n_in, n_emb=16, n_h=32):
        super(LSTMEncoder, self).__init__()
        self.fc_emb = nn.Linear(n_in, n_emb)
        self.lstm_cell = LSTMCell(n_emb, n_h)
        
    def forward(self, inputs, init_hidden=None ,rel_rec=None, rel_send=None):
        """
        args:
            inputs: [batch_size, n_atoms, n_timesteps, n_in]
            init_hidden: [batch_size, n_atoms, n_h]
        return: latents of trajectories of atoms
        """
        batch_size = inputs.size(0)
        num_atoms = inputs.size(1)
        num_timesteps = inputs.size(2)
        if init_hidden is None:
            hc = None
        else:
            h = init_hidden
            h_re = h.view(batch_size*num_atoms, -1)
            #shape: [batch_size*n_atoms, n_h]
            cell = torch.zeros_like(h_re)
            if inputs.is_cuda:
                cell = cell.cuda()
            hc = (h_re, cell)
            
        hs = []
        for i in range(num_timesteps):
            inputs_i = inputs[:,:,i,:]
            #shape: [batch_size, n_atoms, n_in]
            inputs_i = self.fc_emb(inputs_i)
            #shape: [batch_size, n_atoms, n_emb]
            h,c = self.lstm_cell(inputs_i, hc)
            #shape: h:[batch_size*n_atoms, n_h]
            h_re = h.view(batch_size, num_atoms, -1)
            hs.append(h_re)
            hc = (h,c)
        hs = torch.stack(hs) #shape: [n_timesteps, batch_size, n_atoms, n_h]
        hs = torch.permute(hs, (1,2,0,-1)) #shape: [batch_size, n_atoms, n_timesteps, n_h]
        return hs
    



class SoftAttention(nn.Module):
    """Soft Attention"""
    def __init__(self, n_h=32):
        super(SoftAttention, self).__init__()
        self.soft_att = nn.Linear(2*n_h, 1)
        
    def forward(self, inputs, rel_rec_t, rel_send_t):
        """
        inputs: [batch_size, n_atoms, n_timesteps, n_h]
        rel_rec_t: [n_timesteps*n_timesteps, n_timesteps]
        rel_send_t: [n_timesteps*n_timesteps, n_timesteps]
        """
        batch_size = inputs.size(0)
        n_atoms = inputs.size(1)
        n_timesteps = inputs.size(2)
        hs = inputs.view(batch_size*n_atoms, n_timesteps, -1)
        #shape: [batch_size*n_atoms, n_timesteps, n_h]
        senders = torch.matmul(rel_send_t, hs)
        #shape: [batch_size*n_atoms, n_timesteps**2, n_h]
        receivers = torch.matmul(rel_rec_t, hs)
        #shape: [batch_size*n_atoms, n_timesteps**2, n_h]
        edges = torch.cat([senders, receivers], dim=-1)
        #shape: [batch_size*n_atoms, n_timesteps**2, 2*n_h]
        scores = self.soft_att(edges) 
        #shape: [batch_size*n_atoms, n_timesteps**2, 1]
        scores = scores.squeeze(-1)
        #shape: [batch_size*n_atoms, n_timesteps**2]
        scores_diag = torch.diag_embed(scores)
        #shape: [batch_size*n_atoms, n_timesteps**2, n_timesteps**2]
        adj = torch.matmul(rel_send_t.t(), torch.matmul(scores_diag, rel_rec_t))
        #shape: [batch_size*n_atoms, n_timesteps, n_timesteps]
        adj_normalized = F.softmax(adj, dim=-1)
        cs = torch.matmul(adj_normalized, hs)
        #shape: [batch_size*n_atoms, n_timesteps, n_h]
        cs = cs.view(batch_size, n_atoms, n_timesteps, -1)
        #Soft Attention Context: [batch_size, n_atoms, n_timesteps, n_h]
        
        return cs
    
    
class HardwiredAttention(nn.Module):
    """Hardwired Attention"""
    def __init__(self):
        super(HardwiredAttention, self).__init__()
        
    def forward(self, locs, hidden, rel_rec, rel_send, eps=1e-5):
        """
        locs: locations, shape: [batch_size, n_atoms, n_timesteps, 2]
        hidden, shape: [batch_size, n_atoms, n_timesteps, n_h]
        rel_rec,rel_send; shape: [n_atoms*(n_atoms-1), n_atoms]
        """
        batch_size = locs.size(0)
        n_atoms = locs.size(1)
        n_timesteps = locs.size(2)
        locs_re = locs.reshape(batch_size, n_atoms, -1)
        senders = torch.matmul(rel_send, locs_re)
        receivers = torch.matmul(rel_rec, locs_re)
        #shape: [batch_size, n_atoms*(n_atoms-1), n_timesteps*2]
        senders = senders.view(batch_size, n_atoms*(n_atoms-1), n_timesteps, -1)
        receivers = receivers.view(batch_size, n_atoms*(n_atoms-1), n_timesteps, -1)
        
        distances = torch.sqrt(((senders-receivers)**2).sum(-1))
        #shape: [batch_size, n_atoms*(n_atoms-1), n_timestpes]
        weights = 1/(distances+eps)
        
        weights = weights.permute(0,2,1)
        #shape: [batch_size, n_timesteps, n_atoms*(n_atoms-1)]
        weights_diag = torch.diag_embed(weights)
        #shape: [batch_size, n_timesteps, n_edges, n_edges]
        
        adj = torch.matmul(rel_send.t(), torch.matmul(weights_diag, rel_rec))
        
        hidden_permute = hidden.permute(0,2,1,-1)
        
        ch = torch.matmul(adj, hidden_permute)
        #shape: [batch_size, n_timesteps, n_atoms, n_h]
        
        ch = ch.permute(0, 2, 1, -1)
        #shape: [batch_size, n_atoms, n_timesteps, n_h]
        
        return ch
    
    
class LSTMContextEncoder(nn.Module):
    """
    LSTM Context Encoder
    """
    def __init__(self, n_in, n_emb, n_hid):
        super(LSTMContextEncoder, self).__init__()
        self.lstm_encoder = LSTMEncoder(n_in, n_emb, n_hid)
        self.soft_att = SoftAttention(n_hid)
        self.hard_att = HardwiredAttention()
        
    def forward(self, inputs, rel_rec, rel_send, rel_rec_t, rel_send_t):
        """
        args:
            inputs: [batch_size, n_atoms, n_timesteps, n_in]
            rel_rec, rel_send: [n_atoms*(n_atoms-1), n_atoms]
            rel_rec_t, rel_send_t: [n_atoms**2, n_atoms]
        """
        locs = inputs[:,:,:,:2] #shape: [batch_size, n_atoms, n_timesteps, 2]
        hs = self.lstm_encoder(inputs) 
        # hs shape: [batch_size, n_atoms, n_timesteps, n_hid]
        cs = self.soft_att(hs, rel_rec_t, rel_send_t)
        # cs shape: [batch_size, n_atoms, n_timesteps, n_hid]
        ch = self.hard_att(locs, hs, rel_rec, rel_send)
        # ch shape: [batch_size, n_atoms, n_timesteps, n_hid]
        
        ch_t = ch.sum(2) #shape: [batch_size, n_atoms, n_hid]
        cs_t = cs[:,:,-1,:] #shape: [batch_size, n_atoms, n_hid]
        
        c = torch.tanh(torch.cat([cs_t, ch_t], dim=-1))
        #shape: [batch_size, n_atoms, 2*n_hid]
        
        return c
 
    
 

class LSTMDecoder(nn.Module):
    """
    LSTM Decoder
    """
    def __init__(self, n_in, n_emb, n_context, n_noise=4):
        super(LSTMDecoder, self).__init__()
        self.fc_emb = nn.Linear(n_in, n_emb)
        self.lstm_cell = LSTMCell(n_emb, n_context+n_noise)
        self.fc_out = nn.Linear(n_context+n_noise, n_in)
        
        
    def forward(self, inputs, init_hidden, teaching=True):
        """
        args:
            inputs: sequences to predict
              shape: [batch_size, n_atoms, n_timesteps, n_in]
            init_hidden: initial hidden state
              shape: [batch_size, n_atoms, n_context+n_noise]
        """
        x_current = inputs[:,:,0,:] #shape: [batch_size, n_atoms, n_in]
        batch_size = x_current.size(0)
        n_atoms = x_current.size(1)
        h_current = init_hidden #shape: [batch_size, n_atoms, n_context+n_noise]
        h_current = h_current.view(batch_size*n_atoms, -1)
        c_current = torch.zeros_like(h_current)
        if inputs.is_cuda:
            c_current = c_current.cuda()
        hc = (h_current, c_current)
        n_timesteps = inputs.size(2)
        pred_timesteps = n_timesteps-1
        predicted = [x_current]
        hs = []
        for t in range(pred_timesteps):
            x_emb = self.fc_emb(x_current) 
            #shape: [batch_size, n_atoms, n_emb]
            h, c = self.lstm_cell(x_emb, hc)
            hc = (h, c)
            h_re = h.view(batch_size, n_atoms, -1)
            hs.append(h_re)
            
            x_pred = self.fc_out(h_re)
            #shape: [batch_size, n_atoms, n_in]
            
            predicted.append(x_pred)
            
            if not teaching:
                x_current = x_pred
            else:
                x_current = inputs[:,:,t+1,:]
            
        predicted = torch.stack(predicted)
        #shape: [n_timesteps, batch_size, n_atoms, n_in]
        predicted = torch.permute(predicted, (1,2,0,-1))
        
        hs = torch.stack(hs)
        #shape: [pred_timesteps, batch_size, n_atoms, n_hid]
        hs = torch.permute(hs, (1,2,0, -1))
        #shape: [batch_size, n_atoms, pred_timesteps, n_hid]
        
        #reshape hs
        hs = hs.reshape(batch_size, n_atoms, -1)
        #shape: [batch_size, n_atoms, pred_timesteps*n_hid]
        
        return predicted, hs
    



class LSTMGenerator(nn.Module):
    """
    LSTM Generator
    """
    def __init__(self, n_in, n_emb, n_hid=32, n_noise=16):
        super(LSTMGenerator, self).__init__()
        self.lstm_contextEncoder= LSTMContextEncoder(n_in, n_emb, n_hid)
        self.lstm_decoder=LSTMDecoder(n_in, n_emb, 2*n_hid, n_noise)
        
    def forward(self, inputs, noise, rel_rec, rel_send, rel_rec_t, rel_send_t, teaching=True):
        """
        args:
            inputs: [batch_size, n_atoms, n_timesteps, n_in]
            noise: [batch_size, n_atoms, n_noise]
            rel_rec, rel_send: [n_atoms*(n_atoms-1), n_atoms]
            rel_rec_t, rel_send_t: [pred_timesteps**2, n_atoms]
        """
        n_timesteps = inputs.size(2)
        T_obs = int(n_timesteps/2)
        x_obs = inputs[:,:,:T_obs,:] 
        x_pred = inputs[:,:,T_obs:,:]
        
        #compute context vector
        c = self.lstm_contextEncoder(x_obs, rel_rec, rel_send,
                                     rel_rec_t, rel_send_t)
        #context vector, c: [batch_size, n_atoms, n_context=2*n_hid]
        init_hidden = torch.cat([c, noise], dim=-1)
        #shape: [batch_size, n_atoms, 2*n_hid+n_noise]
        predicted, hs = self.lstm_decoder(x_pred, init_hidden, teaching)
        #predicted shape: [batch_size, n_atoms, n_timesteps-T_obs, n_in]
        #hs: [batch_size, n_atoms, n_timesteps-T_obs-1, n_in]
        
        x_fake = torch.cat([x_obs, predicted], dim=2)
        #fake sequences: [batch_size, n_atoms, n_timesteps, n_in]
        
        return x_fake, hs
    
    



class LSTMDiscriminator(nn.Module):
    """
    LSTM Discriminator
    """
    def __init__(self, n_in, n_emb, n_hid):
        super(LSTMDiscriminator, self).__init__()
        self.lstm_contextEncoder = LSTMContextEncoder(n_in, n_emb, n_hid)
        self.lstm_encoder = LSTMEncoder(n_in, n_emb, 2*n_hid)
        self.fc_out = nn.Linear(2*n_hid, 1)
        
    def forward(self, x, rel_rec, rel_send, rel_rec_t, rel_send_t):
        """
        args:
            x, input sequences: [batch_size, n_atoms, n_timesteps, n_in]
        """
        n_timesteps = x.size(2)
        T_obs = int(n_timesteps/2)
        x_obs = x[:,:,:T_obs,:]
        x_pred = x[:,:,T_obs:,:]
        
        #compute context vector
        c = self.lstm_contextEncoder(x_obs, rel_rec, rel_send,
                                     rel_rec_t, rel_send_t)
        #context vector, c: [batch_size, n_atoms, n_context=2*n_hid]
        
        hs = self.lstm_encoder(x_pred, c)
        #shape: [batch_size, n_atoms, n_timesteps-T_obs, 2*n_hid]
        
        h_T = hs[:,:,-1,:]
        
        out = self.fc_out(h_T)
        #shape: [batch_size, n_atoms, 1]
        
        return out
        
        
        
    
    
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        










    





