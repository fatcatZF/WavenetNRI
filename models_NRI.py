"""
copied from https://github.com/ethanfetaya/NRI

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import *

_EPS = 1e-10

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        """
         n_in: #units of input layers
         n_hid: #units of hidden layers
         n_out: #units of output layers
         do_prob: dropout probability
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def batch_norm(self, inputs):
        """
        inputs.size(0): batch size
        inputs.size(1): number of channels
        """
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)
    
    
    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        #print(type(inputs))
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)
    

class CNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        """
        n_in: number of input channels
        n_hid, n_out: number of output channels
        """
        super(CNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                 dilation=1, return_indices=False, 
                                 ceil_mode=False)
        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.dropout_prob = do_prob
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, inputs):
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]       
        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        pred = self.conv_predict(x)
        attention = F.softmax(self.conv_attention(x), dim=-1)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob
    
    
class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        """
        n_in: num_timesteps*num_dimensions
        n_out: num_edgetypes
        """
        super(MLPEncoder, self).__init__()
        self.factor = factor
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)#Equation in the paper:h^1_j=femb(x_j)
        self.mlp2 = MLP(2*n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid*3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid*2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.fill_(0.1)
                
    
    def edge2node(self, x, rel_rec, rel_send):
        """
        update node embeddings according to incoming edges
        G = (V, E)
        x: embedding vectors of edges: |E|*he (he:dimensions of each edge embedding)
        rel_rec: Matrix denoting incomming edges of nodes: |E|*|V|
        rel_send: Matrix denoting outcomming edges of nodes: |E|*|V|
        """
        incomming = torch.matmul(rel_rec.t(), x)
        return incomming/incomming.size(1)
    
    def node2edge(self, x, rel_rec, rel_send):
        """
        Update edge embeddings according to the nodes connected to the edges
        G = (V, E)
        x: embedding vectors of nodes
        """
        receivers = torch.matmul(rel_rec, x) #vectors of embeddings of incomming nodes (receivers):|V|*hn
        senders = torch.matmul(rel_send, x)  #vectors of embeddings of outcomming nodes (senders): |V|*hn
        edges = torch.cat([senders, receivers], dim=2)
        return edges
    
    def forward(self, inputs, rel_rec, rel_send):
        """
        Input shape: [batch_size,num_atoms,num_timesteps,num_dims]
        """
        x = inputs.view(inputs.size(0), inputs.size(1),-1)
        #New shape: [batch_size, num_atoms, num_timesteps*num_dims]
        x = self.mlp1(x)
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x
        
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2) #Skip connection
            x = self.mlp4(x) #Edge embedding
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2) #skip connection
            x = self.mlp4(x)
            
        return self.fc_out(x)
    

class CNNEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True, use_motion=False):
        super(CNNEncoder, self).__init__()
        self.dropout_prob = do_prob
        self.factor = factor
        self.use_motion = use_motion
        self.cnn = CNN(n_in*2, n_hid, n_hid, do_prob)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid*3, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        
        if self.factor:
            print("Using factor graph CNN encoder")
        else:
            print("Using CNN encoder.")
        self.init__weights()
        
    def init__weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
                
    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        #Note: Assume that we have the same graph across all samples
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        #shape: [batch_size, num_atoms, num_timesteps*num_features]
        
        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(inputs.size(0)*receivers.size(1),
                                   inputs.size(2), inputs.size(3))
        #shape: [batch_size*num_edges, num_timesteps, num_features]
        receivers = receivers.transpose(2,1)
        #shape: [batch_size*num_edges, num_features, num_timesteps]
        
        senders = torch.matmul(rel_send, x)
        senders = senders.view(inputs.size(0)*senders.size(1),
                               inputs.size(2), inputs.size(3))
        senders = senders.transpose(2,1)
        
        edges = torch.cat([senders, receivers], dim=1)
        #shape: [batch_size*num_edges, 2*num_features, num_timesteps]
        
        return edges
    
    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming/incoming.size(1)
    
    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges
    
    def forward(self, inputs, rel_rec, rel_send):
        #inputs shape: [batch_size, num_atoms, num_timesteps, num_dims]
        if self.use_motion:
            inputs = inputs[:,:,1:,:]-inputs[:,:,:-1,:]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        #shape: [batch_size*num_edges, 2*num_dims, num_timesteps]
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1)-1)*inputs.size(1), -1)
        x = self.mlp1(x) #[batch_size, num_edges, n_hid]
        x_skip = x
        
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp2(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat([x, x_skip], dim=2) #Skip connection
            x = self.mlp3(x)
            
        return self.fc_out(x)
    
    
class MLPDecoder(nn.Module):
    """MLP Decoder module"""
    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=True):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2*n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first
        
        self.out_fc1 = nn.Linear(n_in_node+msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)
        
        print("Using learned interaction net decoder.")
        self.dropout_prob = do_prob
        
    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):
        #single_timestep_inputs has shape:
        #[batch_size, num_timesteps, num_atoms, num_dims]
        #single_timestep_rel_type has shape:
        #[batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]
        
        #node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        #shape: [batch_size, num_timesteps, num_edges, num_dims]
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)
        #shape: [batch_size, num_timesteps, num_edges, 2*num_dims]
        all_msgs = torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape)
        #requires_grad=True?
        #shape: [batch_size, num_timesteps, num_edges, msg_out]
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()
        if self.skip_first_edge_type:
            start_idx = 1
        else: start_idx = 0
        
        #Run separate MLP for every edge type
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg*single_timestep_rel_type[:,:,:,i:i+1]
            all_msgs += msg
            
        #Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2,-1).matmul(rel_rec).transpose(-2,-1)
        agg_msgs = agg_msgs.contiguous()
        #shape: [batch_size, num_timesteps, num_atoms, msg_out]
        
        #Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)
        
        #Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)
        
        # Predict position/velocity difference
        return single_timestep_inputs + pred
    
    
    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        #Note: Assumes that we have the same graph across all samples
        #inputs shape: [batch_size, num_atoms, num_timesteps, num_dim]
        
        inputs = inputs.transpose(1,2).contiguous()
        #shape: [batch_size, num_timesteps, num_atoms, num_dims]
        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1), 
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)
        #shape: [batch_size, num_timesteps, num_edges, num_edgetypes]
        
        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []
        
        #Only take n-th timesteps as starting points (n:pred_steps)
        last_pred = inputs[:,0::pred_steps, :,:]
        curr_rel_type = rel_type[:,0::pred_steps, :,:]
        #Note: Assumes rel_type is constant (i.e. same across all time steps)
        
        #Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
            preds.append(last_pred)
            
        sizes = [preds[0].size(0), preds[0].size(1)*pred_steps,
                 preds[0].size(2), preds[0].size(3)]
        
        output = torch.zeros(sizes)
        #requires_grad = True?
        
        if inputs.is_cuda:
            output = output.cuda()
            
        #Re_assemble correct timeline
        for i in range(len(preds)):
            output[:,i::pred_steps,:,:] = preds[i]
            
        pred_all = output[:,:(inputs.size(1)-1),:,:]
        
        return pred_all.transpose(1,2).contiguous()
    
        
    
       
    
        
class CausalConv1d(nn.Module):
    """
    causal conv1d layer
    return the sequence with the same length after
    1D causal convolution
    Input: [B, in_channels, L]
    Output: [B, out_channels, L]
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                dilation):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = dilation*(kernel_size-1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                          padding=self.padding, dilation=dilation)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    
    def forward(self, x):
        """
        shape of x: [total_seq, num_features, num_timesteps]
        """
        x = self.conv(x)
        if self.kernel_size==1:
            return x
        return x[:,:,:-self.padding]            
    
    


class GatedCausalConv1d(nn.Module):
    """
    Gated Causal Conv1d Layer
    h_(l+1)=tanh(Wg*h_l)*sigmoid(Ws*h_l)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                dilation):
        super(GatedCausalConv1d, self).__init__()
        self.convg = CausalConv1d(in_channels, out_channels, kernel_size,
                                  dilation) #gate
        self.convs = CausalConv1d(in_channels, out_channels, kernel_size,
                                  dilation)
        
    def forward(self, x):
        return torch.sigmoid(self.convg(x))*torch.tanh(self.convs(x))




                
  


class ResCausalConvBlock(nn.Module):
    """
    Residual convolutional block, composed sequentially of 2 causal 
    convolutions with Leaky ReLU activation functions, and a parallel 
    residual connection.
    """
    def __init__(self, n_in, n_out, kernel_size, dilation):
        super(ResCausalConvBlock, self).__init__()
        self.conv1 = CausalConv1d(n_in, n_out, kernel_size, dilation)
        self.conv2 = CausalConv1d(n_out, n_out, kernel_size, dilation*2)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.skip_conv = CausalConv1d(n_in, n_out, 1, 1)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        x_skip = self.skip_conv(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x+x_skip
        return F.leaky_relu(x)
        

        


class GatedResCausalConvBlock(nn.Module):
    """
    Gated Residual Convolutional block
    """     
    def __init__(self, n_in, n_out, kernel_size, dilation):
        super(GatedResCausalConvBlock, self).__init__()
        self.conv1 = GatedCausalConv1d(n_in, n_out, kernel_size, dilation)
        self.conv2 = GatedCausalConv1d(n_out, n_out, kernel_size, dilation*2)
        self.bn1 = nn.BatchNorm1d(n_out)
        self.bn2 = nn.BatchNorm1d(n_out)
        self.skip_conv = CausalConv1d(n_in, n_out, 1, 1)
        
    def forward(self, x):
        x_skip = self.skip_conv(x)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = x+x_skip
        return x

        


class ResCausalCNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, kernel_size=5, depth=2,
                 do_prob=0.):
        """
        n_in: number of input channels
        n_hid,n_out: number of output channels
        """
        super(ResCausalCNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                dilation=1, return_indices=False, ceil_mode=False)
        res_layers = []#residual convolutional layers
        for i in range(depth):
            in_channels = n_in if i==0 else n_hid
            res_layers += [ResCausalConvBlock(in_channels, n_hid, kernel_size,
                                              dilation=2**(2*i))]
        self.res_blocks = torch.nn.Sequential(*res_layers)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid,1, kernel_size=1)
        self.dropout_prob = do_prob
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, inputs):
        #inputs shape:[batch_size*num_edges, num_dims, num_timesteps]
        x = self.res_blocks(inputs)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        pred = self.conv_predict(x)
        attention = F.softmax(self.conv_attention(x), dim=-1)
        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob
        
        

class GatedResCausalCNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, kernel_size=5, depth=2, 
                 do_prob=0.):
        """
        n_in: number of input channels
        n_hid, n_out: number of output channels
        """
        super(GatedResCausalCNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                dilation=1, return_indices=False, ceil_mode=False)
        res_layers = []#residual convolutional layers
        for i in range(depth):
            in_channels = n_in if i==0 else n_hid
            res_layers += [GatedResCausalConvBlock(in_channels, n_hid, kernel_size, 
                                                   dilation=2**(2*i))]
        self.res_blocks = torch.nn.Sequential(*res_layers)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid,1, kernel_size=1)
        self.dropout_prob = do_prob
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    def forward(self, inputs):
        #inputs shape:[batch_size*num_edges, num_dims, num_timesteps]
        x = self.res_blocks(inputs)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        pred = self.conv_predict(x)
        attention = F.softmax(self.conv_attention(x), dim=-1)
        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob
                
    
        
    

        
        
class ResCausalCNNEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, kernel_size=5,  depth=1, do_prob=0.,
                factor=True, use_motion=False):
        super(ResCausalCNNEncoder,self).__init__()
        self.dropout_prob = do_prob
        self.factor = factor
        self.use_motion = use_motion
        self.cnn = ResCausalCNN(n_in*3, n_hid, n_hid, kernel_size, depth, do_prob)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid*3, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        if self.factor:
            print("Using factor graph ResCausalCNN encoder")
        else:
            print("Using ResCausalCNN encoder.")
        self.init__weights()
        
    def init__weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
        
    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        #Note: Assume that we have the same graph across all samples
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        #shape: [batch_size, num_atoms, num_timesteps*num_features]
        
        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(inputs.size(0)*receivers.size(1),
                                   inputs.size(2), inputs.size(3))
        #shape: [batch_size*num_edges, num_timesteps, num_features]
        receivers = receivers.transpose(2,1)
        #shape: [batch_size*num_edges, num_features, num_timesteps]
        
        senders = torch.matmul(rel_send, x)
        senders = senders.view(inputs.size(0)*senders.size(1),
                               inputs.size(2), inputs.size(3))
        senders = senders.transpose(2,1)
        
        edges = torch.cat([senders, receivers], dim=1)
        #shape: [batch_size*num_edges, 2*num_features, num_timesteps]
        
        return edges
    
    def node2edgediff_temporal(self, inputs, rel_rec, rel_send):
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        #shape: [batch_size, num_atoms, num_timesteps*num_features]
        
        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(inputs.size(0)*receivers.size(1),
                                   inputs.size(2), inputs.size(3))
        #shape: [batch_size*num_edges, num_timesteps, num_features]
        receivers = receivers.transpose(2,1)
        #shape: [batch_size*num_edges, num_features, num_timesteps]
        
        senders = torch.matmul(rel_send, x)
        senders = senders.view(inputs.size(0)*senders.size(1),
                               inputs.size(2), inputs.size(3))
        senders = senders.transpose(2,1)
        edge_diffs = receivers-senders
        #shape: [batch_size*num_edges, num_features, num_timesteps]
        return edge_diffs
        
    
    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming/incoming.size(1)
    
    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges
    
    def forward(self, inputs, rel_rec, rel_send):
        #inputs shape: [batch_size, num_atoms, num_timesteps, num_dims]
        inputs_origin = inputs
        if self.use_motion:
            inputs = inputs[:,:,1:,:]-inputs[:,:,:-1,:]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        #shape: [batch_size*num_edges, 2*num_dims, num_timesteps]
        edge_diffs = self.node2edgediff_temporal(inputs_origin, rel_rec, rel_send)
        #shape: [batch_size*num_edges, num_dims, num_timesteps]
        if self.use_motion:
            edge_diffs = edge_diffs[:,:,:-1]
        
        edges = torch.cat([edge_diffs, edges], dim=1)
        #shape: [batch_size*num_edges, 3*num_dims, num_timesteps]
        
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1)-1)*inputs.size(1), -1)
        x = self.mlp1(x) #[batch_size, num_edges, n_hid]
        x_skip = x
        
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp2(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat([x, x_skip], dim=2) #Skip connection
            x = self.mlp3(x)
            
        return self.fc_out(x)     
    
    
    
class WavenetEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, kernel_size=5,  depth=1, do_prob=0.,
                factor=True, use_motion=False):
        super(WavenetEncoder,self).__init__()
        self.dropout_prob = do_prob
        self.factor = factor
        self.use_motion = use_motion
        self.cnn = GatedResCausalCNN(n_in*3, n_hid, n_hid, kernel_size, depth, do_prob)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid*3, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        if self.factor:
            print("Using factor graph Wavenet encoder")
        else:
            print("Using Wavenet encoder.")
        self.init__weights()
        
    def init__weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
        
    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        #Note: Assume that we have the same graph across all samples
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        #shape: [batch_size, num_atoms, num_timesteps*num_features]
        
        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(inputs.size(0)*receivers.size(1),
                                   inputs.size(2), inputs.size(3))
        #shape: [batch_size*num_edges, num_timesteps, num_features]
        receivers = receivers.transpose(2,1)
        #shape: [batch_size*num_edges, num_features, num_timesteps]
        
        senders = torch.matmul(rel_send, x)
        senders = senders.view(inputs.size(0)*senders.size(1),
                               inputs.size(2), inputs.size(3))
        senders = senders.transpose(2,1)
        
        edges = torch.cat([senders, receivers], dim=1)
        #shape: [batch_size*num_edges, 2*num_features, num_timesteps]
        
        return edges
    
    def node2edgediff_temporal(self, inputs, rel_rec, rel_send):
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        #shape: [batch_size, num_atoms, num_timesteps*num_features]
        
        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(inputs.size(0)*receivers.size(1),
                                   inputs.size(2), inputs.size(3))
        #shape: [batch_size*num_edges, num_timesteps, num_features]
        receivers = receivers.transpose(2,1)
        #shape: [batch_size*num_edges, num_features, num_timesteps]
        
        senders = torch.matmul(rel_send, x)
        senders = senders.view(inputs.size(0)*senders.size(1),
                               inputs.size(2), inputs.size(3))
        senders = senders.transpose(2,1)
        edge_diffs = receivers-senders
        #shape: [batch_size*num_edges, num_features, num_timesteps]
        return edge_diffs
        
    
    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming/incoming.size(1)
    
    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges
    
    def forward(self, inputs, rel_rec, rel_send):
        #inputs shape: [batch_size, num_atoms, num_timesteps, num_dims]
        inputs_origin = inputs
        if self.use_motion:
            inputs = inputs[:,:,1:,:]-inputs[:,:,:-1,:]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        #shape: [batch_size*num_edges, 2*num_dims, num_timesteps]
        edge_diffs = self.node2edgediff_temporal(inputs_origin, rel_rec, rel_send)
        #shape: [batch_size*num_edges, num_dims, num_timesteps]
        if self.use_motion:
            edge_diffs = edge_diffs[:,:,:-1]
        
        edges = torch.cat([edge_diffs, edges], dim=1)
        #shape: [batch_size*num_edges, 3*num_dims, num_timesteps]
        
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1)-1)*inputs.size(1), -1)
        x = F.leaky_relu(x)
        x = x+self.mlp1(x) #[batch_size, num_edges, n_hid]
        x_skip = x
        
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = x+self.mlp2(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat([x, x_skip], dim=2) #Skip connection
            x = self.mlp3(x)
            
        return self.fc_out(x)   
    
    





class WavenetEncoderEuc(nn.Module):
    """
    Wavenet Encoder using euclidean distance with temporal increments as edge features
    """
    def __init__(self, n_in, n_hid, n_out, kernel_size=5,  depth=1, do_prob=0.,
                factor=True, use_motion=False):
        super(WavenetEncoderEuc,self).__init__()
        self.dropout_prob = do_prob
        self.factor = factor
        self.use_motion = use_motion
        self.cnn = GatedResCausalCNN(n_in*2+1, n_hid, n_hid, kernel_size, depth, do_prob)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid*3, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        if self.factor:
            print("Using factor graph Euclidean Wavenet encoder")
        else:
            print("Using Euclidean Wavenet encoder.")
        self.init__weights()
        
        
    def init__weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
    
                
    def node2edge_temporal(self, inputs, rel_rec, rel_send):
         #Note: Assume that we have the same graph across all samples
         x = inputs.view(inputs.size(0), inputs.size(1), -1)
         #shape: [batch_size, num_atoms, num_timesteps*num_features]
         
         receivers = torch.matmul(rel_rec, x)
         receivers = receivers.view(inputs.size(0)*receivers.size(1),
                                    inputs.size(2), inputs.size(3))
         #shape: [batch_size*num_edges, num_timesteps, num_features]
         receivers = receivers.transpose(2,1)
         #shape: [batch_size*num_edges, num_features, num_timesteps]
         
         senders = torch.matmul(rel_send, x)
         senders = senders.view(inputs.size(0)*senders.size(1),
                                inputs.size(2), inputs.size(3))
         senders = senders.transpose(2,1)
         
         edges = torch.cat([senders, receivers], dim=1)
         #shape: [batch_size*num_edges, 2*num_features, num_timesteps]
         
         return edges
     
                
    def node2edgediff_temporal(self, inputs, rel_rec, rel_send):
         x = inputs.view(inputs.size(0), inputs.size(1), -1)
         #shape: [batch_size, num_atoms, num_timesteps*num_features]
         
         receivers = torch.matmul(rel_rec, x)
         receivers = receivers.view(inputs.size(0)*receivers.size(1),
                                    inputs.size(2), inputs.size(3))
         #shape: [batch_size*num_edges, num_timesteps, num_features]
         receivers = receivers.transpose(2,1)
         #shape: [batch_size*num_edges, num_features, num_timesteps]
         
         senders = torch.matmul(rel_send, x)
         senders = senders.view(inputs.size(0)*senders.size(1),
                                inputs.size(2), inputs.size(3))
         senders = senders.transpose(2,1)
    
         edge_diffs = torch.sqrt(((senders-receivers)**2).sum(1)).unsqueeze(1)
         #shape: [batch_size*num_edges, 1, num_timesteps]
         return edge_diffs
     
        
    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming/incoming.size(1)
    
    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges
    
    def forward(self, inputs, rel_rec, rel_send):
        #inputs shape: [batch_size, num_atoms, num_timesteps, num_dims]
        inputs_origin = inputs
        if self.use_motion:
            inputs = inputs[:,:,1:,:]-inputs[:,:,:-1,:]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        #shape: [batch_size*num_edges, 2*num_dims, num_timesteps]
        edge_diffs = self.node2edgediff_temporal(inputs_origin, rel_rec, rel_send)
        #shape: [batch_size*num_edges, num_dims, num_timesteps]
        if self.use_motion:
            edge_diffs = edge_diffs[:,:,:-1]
        
        edges = torch.cat([edge_diffs, edges], dim=1)
        #shape: [batch_size*num_edges, 3*num_dims, num_timesteps]
        
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1)-1)*inputs.size(1), -1)
        x = F.leaky_relu(x)
        x = x+self.mlp1(x) #[batch_size, num_edges, n_hid]
        x_skip = x
        
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = x+self.mlp2(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat([x, x_skip], dim=2) #Skip connection
            x = self.mlp3(x)
            
        return self.fc_out(x) 
     
    
                
                
        




class WavenetEncoderSym(nn.Module):
    """
    Wavenet Encoder using symmetric way building edge features
    """
    def __init__(self, n_in, n_hid, n_out, kernel_size=5,  depth=1, do_prob=0.,
                factor=True, use_motion=False):
        super(WavenetEncoderSym,self).__init__()
        self.dropout_prob = do_prob
        self.factor = factor
        self.use_motion = use_motion
        self.cnn = GatedResCausalCNN(n_in+1, n_hid, n_hid, kernel_size, depth, do_prob)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid*2, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        if self.factor:
            print("Using factor graph Wavenet encoder with symmetric Features")
        else:
            print("Using Wavenet encoder with symmetric Features.")
        self.init__weights()
        
    def init__weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
                
    def node2edge_temporal(self, inputs, rel_rec, rel_send):
         #Note: Assume that we have the same graph across all samples
         x = inputs.view(inputs.size(0), inputs.size(1), -1)
         #shape: [batch_size, num_atoms, num_timesteps*num_features]
         
         receivers = torch.matmul(rel_rec, x)
         receivers = receivers.view(inputs.size(0)*receivers.size(1),
                                    inputs.size(2), inputs.size(3))
         #shape: [batch_size*num_edges, num_timesteps, num_features]
         receivers = receivers.transpose(2,1)
         #shape: [batch_size*num_edges, num_features, num_timesteps]
         
         senders = torch.matmul(rel_send, x)
         senders = senders.view(inputs.size(0)*senders.size(1),
                                inputs.size(2), inputs.size(3))
         senders = senders.transpose(2,1)
         #shape: [batch_size*num_edges, num_features, num_timesteps]
         
         edges = senders*receivers
         #shape: [batch_size*num_edges, num_features, num_timesteps]
         
         return edges
     
    def node2edgediff_temporal(self, inputs, rel_rec, rel_send):
          x = inputs.view(inputs.size(0), inputs.size(1), -1)
          #shape: [batch_size, num_atoms, num_timesteps*num_features]
          
          receivers = torch.matmul(rel_rec, x)
          receivers = receivers.view(inputs.size(0)*receivers.size(1),
                                     inputs.size(2), inputs.size(3))
          #shape: [batch_size*num_edges, num_timesteps, num_features]
          receivers = receivers.transpose(2,1)
          #shape: [batch_size*num_edges, num_features, num_timesteps]
          
          senders = torch.matmul(rel_send, x)
          senders = senders.view(inputs.size(0)*senders.size(1),
                                 inputs.size(2), inputs.size(3))
          senders = senders.transpose(2,1)
     
          edge_diffs = torch.sqrt(((senders-receivers)**2).sum(1)).unsqueeze(1)
          #shape: [batch_size*num_edges, 1, num_timesteps]
          return edge_diffs
      
    def edge2node(self, x, rel_rec, rel_send):
         incoming = torch.matmul(rel_rec.t(), x)
         return incoming/incoming.size(1)
     
    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        #edges = torch.cat([senders, receivers], dim=2)
        edges = senders*receivers
        return edges #shape: [batch_size, n_edges, n_hid]
    
    def forward(self, inputs, rel_rec, rel_send):
        #inputs shape: [batch_size, num_atoms, num_timesteps, num_dims]
        inputs_origin = inputs
        if self.use_motion:
            inputs = inputs[:,:,1:,:]-inputs[:,:,:-1,:]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        #shape: [batch_size*num_edges, 2*num_dims, num_timesteps]
        edge_diffs = self.node2edgediff_temporal(inputs_origin, rel_rec, rel_send)
        #shape: [batch_size*num_edges, num_dims, num_timesteps]
        if self.use_motion:
            edge_diffs = edge_diffs[:,:,:-1]
        
        edges = torch.cat([edge_diffs, edges], dim=1)
        #shape: [batch_size*num_edges,num_dims+1, num_timesteps]
        
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1)-1)*inputs.size(1), -1)
        #shape: [batch_size, n_edges, n_hid]
        x = F.leaky_relu(x)
        x = x+self.mlp1(x) #[batch_size, num_edges, n_hid]
        x_skip = x
        
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            #shape: [batch_size, n_nodes, n_hid]
            x = x+self.mlp2(x)
            x = self.node2edge(x, rel_rec, rel_send)
            #shape: [batch_size, n_edges, n_hid]
            x = torch.cat([x, x_skip], dim=2) #Skip connection
            x = self.mlp3(x)
            
        return self.fc_out(x) 
    




class CNNEncoderSym(nn.Module):
    """
    CNN Encoder using symmetric way building edge features
    """
    def __init__(self, n_in, n_hid, n_out, kernel_size=5,  depth=1, do_prob=0.,
                factor=True, use_motion=False):
        super(CNNEncoderSym,self).__init__()
        self.dropout_prob = do_prob
        self.factor = factor
        self.use_motion = use_motion
        self.cnn = CNN(n_in+1, n_hid, n_hid, do_prob)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid*2, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        if self.factor:
            print("Using factor graph CNN encoder with symmetric Features")
        else:
            print("Using CNN encoder with symmetric Features.")
        self.init__weights()
        
    def init__weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
                
    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        #inputs: [batch_size, n_atoms, n_timesteps, n_features]
        
         x = inputs.view(inputs.size(0), inputs.size(1), -1)
         #shape: [batch_size, num_atoms, num_timesteps*num_features]
         
         receivers = torch.matmul(rel_rec, x)
         receivers = receivers.view(inputs.size(0)*receivers.size(1),
                                    inputs.size(2), inputs.size(3))
         #shape: [batch_size*num_edges, num_timesteps, num_features]
         receivers = receivers.transpose(2,1)
         #shape: [batch_size*num_edges, num_features, num_timesteps]
         
         senders = torch.matmul(rel_send, x)
         senders = senders.view(inputs.size(0)*senders.size(1),
                                inputs.size(2), inputs.size(3))
         senders = senders.transpose(2,1)
         #shape: [batch_size*num_edges, num_features, num_timesteps]
         
         edges = senders*receivers
         #shape: [batch_size*num_edges, num_features, num_timesteps]
         
         return edges
     
    def node2edgediff_temporal(self, inputs, rel_rec, rel_send):
          #inputs shape: [batch_size, n_atoms, n_timesteps, n_features]
          x = inputs.view(inputs.size(0), inputs.size(1), -1)
          #shape: [batch_size, n_atoms, n_timesteps*n_features]
          
          receivers = torch.matmul(rel_rec, x)
          #shape: [batch_size, n_edges, n_timesteps*n_features]
          receivers = receivers.view(inputs.size(0)*receivers.size(1),
                                     inputs.size(2), inputs.size(3))
          #shape: [batch_size*n_edges, n_timesteps, n_features]
          receivers = receivers.transpose(2,1)
          #shape: [batch_size*n_edges, n_features, n_timesteps]
          
          senders = torch.matmul(rel_send, x)
          senders = senders.view(inputs.size(0)*senders.size(1),
                                 inputs.size(2), inputs.size(3))
          senders = senders.transpose(2,1)
     
          edge_diffs = torch.sqrt(((senders-receivers)**2).sum(1)).unsqueeze(1)
          #shape: [batch_size*num_edges, 1, n_timesteps]
          return edge_diffs
      
    def edge2node(self, x, rel_rec, rel_send):
         incoming = torch.matmul(rel_rec.t(), x)
         return incoming/incoming.size(1)
     
    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        #edges = torch.cat([senders, receivers], dim=2)
        edges = senders*receivers
        return edges #shape: [batch_size, n_edges, n_hid]
    
    def forward(self, inputs, rel_rec, rel_send):
        #inputs shape: [batch_size, n_atoms, n_timesteps, n_dims]
        inputs_origin = inputs
        if self.use_motion:
            inputs = inputs[:,:,1:,:]-inputs[:,:,:-1,:]
            #shape: [batch_size, n_atoms, n_timesteps-1, n_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        #shape: [batch_size*num_edges, 2*num_dims, num_timesteps]
        edge_diffs = self.node2edgediff_temporal(inputs_origin, rel_rec, rel_send)
        #shape: [batch_size*num_edges, num_dims, num_timesteps]
        if self.use_motion:
            edge_diffs = edge_diffs[:,:,:-1]
        
        edges = torch.cat([edge_diffs, edges], dim=1)
        #shape: [batch_size*n_edges,n_dims+1, n_timesteps]
        
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1)-1)*inputs.size(1), -1)
        #shape: [batch_size, n_edges, n_hid]
        x = F.leaky_relu(x)
        x = x+self.mlp1(x) #[batch_size, num_edges, n_hid]
        x_skip = x
        
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            #shape: [batch_size, n_nodes, n_hid]
            x = x+self.mlp2(x)
            x = self.node2edge(x, rel_rec, rel_send)
            #shape: [batch_size, n_edges, n_hid]
            x = torch.cat([x, x_skip], dim=2) #Skip connection
            x = self.mlp3(x)
            
        return self.fc_out(x) 
    
    
     
    
     
        
           
     
    
                
    
                
    
    



class WavenetEncoderRaw(nn.Module):
    """
    Wavenet Encoder using concatenated raw features
    """
    def __init__(self, n_in, n_hid, n_out, kernel_size=5,  depth=1, do_prob=0.,
                factor=True, use_motion=False):
        super(WavenetEncoderRaw,self).__init__()
        self.dropout_prob = do_prob
        self.factor = factor
        self.use_motion = use_motion
        self.cnn = GatedResCausalCNN(n_in*2, n_hid, n_hid, kernel_size, depth, do_prob)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid*3, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)
        if self.factor:
            print("Using factor graph Wavenet encoder with raw Features")
        else:
            print("Using Wavenet encoder with raw Features.")
        self.init__weights()
        
    def init__weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
                
    def node2edge_temporal(self, inputs, rel_rec, rel_send):
          #Note: Assume that we have the same graph across all samples
          x = inputs.view(inputs.size(0), inputs.size(1), -1)
          #shape: [batch_size, num_atoms, num_timesteps*num_features]
          
          receivers = torch.matmul(rel_rec, x)
          receivers = receivers.view(inputs.size(0)*receivers.size(1),
                                     inputs.size(2), inputs.size(3))
          #shape: [batch_size*num_edges, num_timesteps, num_features]
          receivers = receivers.transpose(2,1)
          #shape: [batch_size*num_edges, num_features, num_timesteps]
          
          senders = torch.matmul(rel_send, x)
          senders = senders.view(inputs.size(0)*senders.size(1),
                                 inputs.size(2), inputs.size(3))
          senders = senders.transpose(2,1)
          
          edges = torch.cat([senders, receivers], dim=1)
          #shape: [batch_size*num_edges, 2*num_features, num_timesteps]
          
          return edges
      
    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming/incoming.size(1)
    
    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges
    
    
    def forward(self, inputs, rel_rec, rel_send):
        #inputs shape: [batch_size, num_atoms, num_timesteps, num_dims]
        
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        #shape: [batch_size*num_edges, 2*num_dims, num_timesteps]
        
        
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1)-1)*inputs.size(1), -1)
        x = F.leaky_relu(x)
        x = x+self.mlp1(x) #[batch_size, num_edges, n_hid]
        x_skip = x
        
        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = x+self.mlp2(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat([x, x_skip], dim=2) #Skip connection
            x = self.mlp3(x)
            
        return self.fc_out(x) 
    
    
      
     
                
    
        
class ZeroEncoder(nn.Module):
    """
    Encoder always return zero
    """
    def __init__(self, n_in, n_hid, n_out, kernel_size=5,  depth=1, do_prob=0.,
                factor=True, use_motion=False):
        super(ZeroEncoder,self).__init__()
        
    def forward(self, inputs, rel_rec, rel_send):
        #inputs: [batch_size, n_atoms, n_timesteps, n_dims]
        
        batch_size = inputs.size(0)
        n_atoms = inputs.size(1)
        
        zeros = torch.zeros(batch_size, n_atoms*(n_atoms-1), 2)
        if inputs.is_cuda:
            zeros = zeros.cuda()
            
        return zeros
    
    



















    
   

        
    
