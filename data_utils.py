import os
import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader


"""
def load_spring_sim(batch_size=1 , suffix="_static_5", folder="data/simulation/spring_simulation/"):
    loc_train = np.load(folder+"loc_sampled_all_train_group"+suffix+".npy")
    vel_train = np.load(folder+"vel_sampled_all_train_group"+suffix+".npy")
    groups_train = np.load(folder+"gr_train_group"+suffix+".npy")
    
    loc_valid = np.load(folder+"loc_sampled_all_valid_group" + suffix + '.npy')
    vel_valid = np.load( folder+"vel_sampled_all_valid_group"+ suffix + '.npy')
    groups_valid = np.load(folder+"gr_valid_group" + suffix + '.npy')
    
    loc_test = np.load(folder+"loc_sampled_all_test_group" + suffix + '.npy')
    vel_test = np.load(folder+"vel_sampled_all_test_group" + suffix + '.npy')
    groups_test = np.load(folder+"gr_test_group" + suffix + '.npy')
    
    #[num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]
    
    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()
    
    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1
    
    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1
    
    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1
    
    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    groups_train = np.reshape(groups_train, [-1, num_atoms ** 2])
    groups_train = np.array((groups_train + 1) / 2, dtype=np.int64)
    
    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    groups_valid = np.reshape(groups_valid, [-1, num_atoms ** 2])
    groups_valid = np.array((groups_valid + 1) / 2, dtype=np.int64)
       
    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    groups_test = np.reshape(groups_test, [-1, num_atoms ** 2])
    groups_test = np.array((groups_test + 1) / 2, dtype=np.int64)
    
    feat_train = torch.from_numpy(feat_train)
    groups_train = torch.from_numpy(groups_train)
    feat_valid = torch.from_numpy(feat_valid)
    groups_valid = torch.from_numpy(groups_valid)
    feat_test = torch.from_numpy(feat_test)
    groups_test = torch.from_numpy(groups_test)
    
    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    
    
    groups_train = groups_train[:, off_diag_idx]
    groups_valid = groups_valid[:, off_diag_idx]
    groups_test = groups_test[:, off_diag_idx]
    
    train_data = TensorDataset(feat_train, groups_train)
    valid_data = TensorDataset(feat_valid, groups_valid)
    test_data = TensorDataset(feat_test, groups_test)
    
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min
    
 """

def load_spring_sim(batch_size=1, suffix='', label_rate=0.02, save_folder="data/simulation/spring_simulation",
              load_folder=None, normalize=True):
    if load_folder is not None:
        #load saved data
        train_loader_path = os.path.join(load_folder, "train_data_loader"+suffix+".pth")
        valid_loader_path = os.path.join(load_folder, "valid_data_loader"+suffix+".pth")
        test_loader_path = os.path.join(load_folder, "test_data_loader"+suffix+".pth")
        datainfo_file = "datainfo"+suffix+".npy"
        datainfo_path = os.path.join(load_folder, datainfo_file)
        
        train_data_loader = torch.load(train_loader_path)
        valid_data_loader = torch.load(valid_loader_path)
        test_data_loader = torch.load(test_loader_path)
        datainfo = np.load(datainfo_path)
        return train_data_loader, valid_data_loader, test_data_loader,\
               datainfo[0], datainfo[1], datainfo[2], datainfo[3]
    
    
    
    loc_all = np.load('data/simulation/spring_simulation/loc_sampled_all_sim_group' + suffix + '.npy')
    vel_all = np.load('data/simulation/spring_simulation/vel_sampled_all_sim_group' + suffix + '.npy')
    edges_all = np.load('data/simulation/spring_simulation/gr_sim_group' + suffix + '.npy')
    
    num_sims = loc_all.shape[0]
    indices = np.arange(num_sims)
    np.random.shuffle(indices)
    train_idx = int(num_sims*0.6)
    valid_idx = int(num_sims*0.8)
    train_indices = indices[:train_idx]
    valid_indices = indices[train_idx:valid_idx]
    test_indices = indices[valid_idx:]
    
    
    
    loc_train = loc_all[train_indices]
    vel_train = vel_all[train_indices]
    edges_train = edges_all[train_indices]

    loc_valid = loc_all[valid_indices]
    vel_valid = vel_all[valid_indices]
    edges_valid = edges_all[valid_indices]

    loc_test = loc_all[test_indices]
    vel_test = vel_all[test_indices]
    edges_test = edges_all[test_indices]

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()
    
    datainfo = np.array([loc_max, loc_min, vel_max, vel_min])
    datainfo_file = "datainfo"+suffix+".npy"

    if normalize:
      # Normalize to [-1, 1]
      loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
      vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

      loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
      vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

      loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
      vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]
    
    #create mask for training and validation data
    edges_train_masked = edges_train.clone()
    edges_train_masked[edges_train_masked==0]=-1
    edges_valid_masked = edges_valid.clone()
    edges_valid_masked[edges_valid_masked==0]=-1
    mask_train = np.random.choice(a=[1,0], size=edges_train_masked.size(),
                       p=[label_rate, 1-label_rate])
    mask_valid = np.random.choice(a=[1,0], size=edges_valid_masked.size(),
                       p=[label_rate, 1-label_rate])
    mask_train = torch.LongTensor(mask_train)
    mask_valid = torch.LongTensor(mask_valid)
    edges_train_masked = edges_train_masked*mask_train
    edges_valid_masked = edges_valid_masked*mask_valid
    
    edges_train_stack = torch.stack([edges_train, edges_train_masked], dim=-1)
    edges_valid_stack = torch.stack([edges_valid, edges_valid_masked], dim=-1)
    
    

    train_data = TensorDataset(feat_train, edges_train_stack)
    valid_data = TensorDataset(feat_valid, edges_valid_stack)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=1)
    
    train_loader_path = os.path.join(save_folder, "train_data_loader"+suffix+".pth")
    valid_loader_path = os.path.join(save_folder, "valid_data_loader"+suffix+".pth")
    test_loader_path = os.path.join(save_folder, "test_data_loader"+suffix+".pth")
    datainfo_path = os.path.join(save_folder, datainfo_file)
    
    #save dataloader and datainfo array
    torch.save(train_data_loader, train_loader_path)
    torch.save(valid_data_loader, valid_loader_path)
    torch.save(test_data_loader, test_loader_path)
    np.save(datainfo_path, datainfo)
    

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min   
    


def load_gordon(batch_size=1, suffix="acc" ,label_rate=0.02,
                save_folder="data/gordon/preprocessed", load_folder=None, normalize=True):
    if load_folder is not None:
        #load saved data
        train_loader_path = os.path.join(load_folder, "train_data_loader_"+suffix+".pth")
        valid_loader_path = os.path.join(load_folder, "valid_data_loader_"+suffix+".pth")
        test_loader_path = os.path.join(load_folder, "test_data_loader_"+suffix+".pth")
        
        train_data_loader = torch.load(train_loader_path)
        valid_data_loader = torch.load(valid_loader_path)
        test_data_loader = torch.load(test_loader_path)
       
        return train_data_loader, valid_data_loader, test_data_loader
    
        
    #features shape: [batch_size, num_atoms, num_timesteps, num_features]
    features_train = np.load('data/gordon/preprocessed/' + suffix + '/examples_train.npy')
    edges_train = np.load('data/gordon/preprocessed/' + suffix + '/labels_train.npy')

    features_valid = np.load('data/gordon/preprocessed/' + suffix + '/examples_valid.npy')
    edges_valid = np.load('data/gordon/preprocessed/' + suffix + '/labels_valid.npy')

    features_test = np.load('data/gordon/preprocessed/' + suffix + '/examples_test.npy')
    edges_test = np.load('data/gordon/preprocessed/' + suffix + '/labels_test.npy')
        
    num_atoms = features_train.shape[1]
        
    if normalize:
        if suffix == "acc":
            features_max = features_train.max()
            features_min = features_train.min()
            features_train = (features_train - features_min) * 2 / (features_max - features_min) - 1
            features_valid = (features_valid - features_min) * 2 / (features_max - features_min) - 1
            features_test = (features_test - features_min) * 2 / (features_max - features_min) - 1
            
        elif suffix=="orien":
            features_train = np.cos(features_train/2)
            features_valid = np.cos(features_valid/2)
            features_test = np.cos(features_test/2)
                
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)
        
    features_train = torch.FloatTensor(features_train)
    edges_train = torch.LongTensor(edges_train)
    features_valid = torch.FloatTensor(features_valid)
    edges_valid = torch.LongTensor(edges_valid)
    features_test = torch.FloatTensor(features_test)
    edges_test = torch.LongTensor(edges_test)
        
    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
            [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]
        
        
    #create mask for training and validation data
    edges_train_masked = edges_train.clone()
    edges_train_masked[edges_train_masked==0]=-1
    edges_valid_masked = edges_valid.clone()
    edges_valid_masked[edges_valid_masked==0]=-1
    mask_train = np.random.choice(a=[1,0], size=edges_train_masked.size(),
                           p=[label_rate, 1-label_rate])
    mask_valid = np.random.choice(a=[1,0], size=edges_valid_masked.size(),
                           p=[label_rate, 1-label_rate])
    mask_train = torch.LongTensor(mask_train)
    mask_valid = torch.LongTensor(mask_valid)
    edges_train_masked = edges_train_masked*mask_train
    edges_valid_masked = edges_valid_masked*mask_valid
        
        
    edges_train_stack = torch.stack([edges_train, edges_train_masked], dim=-1)
    edges_valid_stack = torch.stack([edges_valid, edges_valid_masked], dim=-1)
        
    train_data = TensorDataset(features_train, edges_train_stack)
    valid_data = TensorDataset(features_valid, edges_valid_stack)
    test_data = TensorDataset(features_test, edges_test)
        
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)
        
    train_loader_path = os.path.join(save_folder, "train_data_loader_"+suffix+".pth")
    valid_loader_path = os.path.join(save_folder, "valid_data_loader_"+suffix+".pth")
    test_loader_path = os.path.join(save_folder, "test_data_loader_"+suffix+".pth")
        
    #save dataloader 
    torch.save(train_data_loader, train_loader_path)
    torch.save(valid_data_loader, valid_loader_path)
    torch.save(test_data_loader, test_loader_path)
        
    return train_data_loader, valid_data_loader, test_data_loader

        
        
        
        
        
        
        
        
        
        
        
        
                
        
                
        
        
        
        
    




    
    