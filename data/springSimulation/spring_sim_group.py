import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time

def group_initialize(n, ga_values_factor=3):
    """
    Initialize a group
    n: number of atoms
    """
    #an array denoting groups corresponding to each individual
    group_initial = np.random.choice(np.arange(n), size=(n,))
    #a dictionary denoting groups corresponding to each individual
    ga_dict = dict({i:g for i,g in enumerate(group_initial)})
    # group assignment matrix: ga_matrix[i,k]=1 denotes node i belongs to kth group
    ga_matrix = np.zeros((n,n))
    ga_matrix[list(ga_dict.keys()), list(ga_dict.values())]=1
    ga_values = ga_values_factor*ga_matrix.copy()
    #group relation matrix: gr_matrix[i,j]=1 denotes i and j are in one group
    gr_matrix = np.zeros((n,n))
    for i in range(gr_matrix.shape[0]):
        for j in range(gr_matrix.shape[1]):
            if ga_dict[i]==ga_dict[j]:
                gr_matrix[i,j]=1
    return group_initial, ga_dict, ga_matrix, ga_values, gr_matrix


def compute_ga_values(ga_matrix, ga_ages, age_factor=0.01, ga_factor=3):
    """
    compute group assignment values 
    according to group assignment ages
    params:
      ga_matrix: group assignment matrix
      ga_ages: group assignment ages
      age_factor: to adjust the change of the 
    """
    return ga_factor*(ga_matrix+((-1)**ga_matrix)*age_factor*ga_ages)

def softmax(m, axis=1):
    """
    Compute the ga probabilities given ga_values
    parmas:
    m: ga_values 
    """
    m_exp = np.exp(m)
    m_exp_sum = np.diag(m_exp.sum(axis))
    m_exp_sum_inv = inv(m_exp_sum)
    m_prob = np.matmul(m_exp_sum_inv, m_exp)
    return m_prob


def sample_ga(ga_values):
    """
    Sample new group assignment based on ga values
    """
    n = ga_values.shape[0] #number of atoms
    atoms = np.arange(n)
    groups = np.arange(ga_values.shape[1])
    ga_probs = softmax(ga_values, axis=1)
    ga_dict = dict()
    for atom in atoms:
            ga_dict[atom]= np.random.choice(groups, p=ga_probs[atom])
    # group assignment matrix: ga_matrix[i,k]=1 denotes node i belongs to kth group
    ga_matrix = np.zeros((n,n))
    ga_matrix[list(ga_dict.keys()), list(ga_dict.values())]=1
    #group relation matrix: gr_matrix[i,j]=1 denotes i and j are in one group
    gr_matrix = np.zeros((n,n))
    for i in range(gr_matrix.shape[0]):
        for j in range(gr_matrix.shape[1]):
            if ga_dict[i]==ga_dict[j]:
                gr_matrix[i,j]=1
                
    ga = list(ga_dict.values())
    return ga, ga_dict, ga_matrix, gr_matrix



def group_to_interaction(Group, k=3, b=0.1):
    """
    P(I[i,j]=1|G[i,j])=1-exp(-k(G[i,j]+b))
    """
    I = np.zeros_like(Group)
    for i in range(Group.shape[0]):
        for j in range(Group.shape[1]):
            I[i,j] = np.random.choice([1,0],p=[1-np.exp(-k*(Group[i,j]+b)),np.exp(-k*(Group[i,j]+b))])
    np.fill_diagonal(I, 0)
    #Symmetric
    I = np.tril(I)+np.tril(I).T
    return I


class SpringSim(object):
    """
    copied from https://github.com/ethanfetaya/NRI/blob/master/data/synthetic_sim.py
    adapted for Dynamic Graphs
    """
    def __init__(self, n_balls=5, box_size=10.,loc_std=0.5,vel_norm=0.5,
                 interaction_strength=0.1, noise_var=0., age_factor=0.01, ga_values_factor=3,
                 K=3, b=0.001, dynamic=False):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var
        
        self._delta_T = 0.004
        self._max_F = 0.1/self._delta_T
        self.age_factor = age_factor
        self.ga_values_factor = ga_values_factor
        self.K = K
        self.b = b
        self.dynamic = dynamic
        
    def _energy(self, loc, vel, edges):
        with np.errstate(divide="ignore"):
            K = 0.5*(vel**2).sum() #kinetic energy
            U = 0 #potential energy
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i!=j:
                        r = loc[:,i]-loc[:,j]
                        dist = np.sqrt((r**2).sum())
                        U += 0.5*self.interaction_strength*edges[
                            i,j]*(dist**2)/2
            return U+K
        
    def _clamp(self, loc, vel):
        """
        args:
          loc: 2xN locations at one time step
          vel: 2xN velocity at one time step
        return: location and velocity after hitting walls and returning after elastically colliding with walls
        """
        assert (np.all(loc<self.box_size*3))
        assert (np.all(loc>-self.box_size*3))
        
        over = loc > self.box_size
        loc[over] = 2*self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))
        
        vel[over] = -np.abs(vel[over])
        
        under = loc < -self.box_size
        loc[under] = -2*self.box_size - loc[under]
        
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])
        
        return loc, vel
    
    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matrix
        Output: dist is a NxM matrix where dist[i,j] is the square norm
                between A[i,:] and B[j,:]
        i.e. dist[i,j] = |A[i,:]-B[j,:]|^2
        """
        A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
        B_norm = (B**2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm+B_norm-2*A.dot(B.transpose())
        return dist
    
    def sample_trajectory(self, T=10000, sample_freq=100):
        """
        Interaction edges may change at each timestep
        if dynamic, the group assignment will be re-evaluated at each sampled 
            timestep according to ga_values
            ga_values will change at each time according to current group assignment ga_matrix 
            and group assignment ages ga_ages
            if the group assignment ga_matrix[i] changes at one sampled timestep, 
            the corresponding ga_ages[i] will be reset to 0 and the ga_values[i] will be reset
            to ga_values_factor*ga_matrix[i]
        """
        n = self.n_balls
        age_factor = self.age_factor
        ga_values_factor = self.ga_values_factor
        K = self.K
        b = self.b
        assert (T % sample_freq == 0)
        T_save = int(T/sample_freq-1)
        diag_mask = np.ones((n,n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        
        #Initialize groups
        ga, ga_dict, ga_matrix, ga_values, gr = group_initialize(n, ga_values_factor)
        #Initialize group assignment ages
        ga_ages = np.zeros_like(ga_matrix)
        #Initialize Interaction matrix 
        edges = group_to_interaction(gr, K, b)
        
        sampled_indices = []
        loc = np.zeros((T_save,2,n))
        vel = np.zeros((T_save,2,n))
        loc_all = np.zeros((T,2, n))
        vel_all = np.zeros((T,2,n))
        loc_next = np.random.randn(2,n)*self.loc_std
        vel_next = np.random.randn(2,n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next =  vel_next*self.vel_norm/v_norm
        loc[0,:,:], vel[0,:,:] = self._clamp(loc_next, vel_next)
        loc_all[0,:,:], vel_all[0,:,:] = self._clamp(loc_next, vel_next)
        all_edges_sampled = np.zeros((T_save, n, n))
        all_edges = np.zeros((T, n, n))
        all_edges_sampled[0,:,:] = edges
        all_edges[0,:,:] = edges
        
        
        if self.dynamic:
            ga_sampled = np.zeros((T_save, n)) #group labels of each atom at all sampled time steps
            gr_sampled = np.zeros((T_save, n, n)) #group Initialization
            ga_sampled[0,:] = ga
            gr_sampled[0,:,:] = gr
            
            
        
        with np.errstate(divide="ignore"):
            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size, 0)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)))).sum(
                axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F
            
            vel_next += self._delta_T*F
            
            #run leapfrog
            for i in range(1, T):
                #Assumption: the next states(loc and vel) are determined by
                #current states and current interaction edges
                loc_next += self._delta_T*vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)
                loc_all[i,:,:], vel_all[i,:,:] = loc_next, vel_next
                #compute current interaction edges based on group relationship
                edges = group_to_interaction(gr, K, b)
                all_edges[i,:,:] = edges
                
                
                if i%sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    sampled_indices.append(i)
                    if self.dynamic:
                        new_ga, new_ga_dict, new_ga_matrix, new_gr = sample_ga(ga_values)
                        ga_sampled[counter,:] = new_ga
                        gr_sampled[counter,:,:] = new_gr                       
                        #find the changed group assignment
                        changed_atoms = np.unique(np.where(ga!=new_ga)[0])
                        #reset the ga_values and ga_ages
                        ga_values[changed_atoms] = ga_values_factor*new_ga_matrix[changed_atoms]
                        ga_ages[changed_atoms] = 0
                        ga = new_ga
                        ga_dict = new_ga_dict
                        ga_matrix = new_ga_matrix
                        gr = new_gr
                        #reevaluate edges
                        edges = group_to_interaction(gr, K, b)
                        
                        
                    all_edges_sampled[counter,:,:] = edges
                    all_edges[i,:,:] = edges
                    counter+=1
                    
                
                forces_size = -self.interaction_strength*edges
                np.fill_diagonal(forces_size, 0)
                
                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n),
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n,
                                                                   n)))).sum(
                    axis=-1)
                                                                       
                
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
                
                
                if self.dynamic:
                    #increase ga_ages and compute ga_values
                    ga_values = compute_ga_values(ga_matrix, ga_ages, age_factor, ga_values_factor)
                    ga_ages+=1
                
            loc += np.random.randn(T_save, 2, n) * self.noise_var
            vel += np.random.randn(T_save, 2, n) * self.noise_var
            sampled_indices = np.array(sampled_indices)
            
            if self.dynamic:
                ga = ga_sampled
                gr = gr_sampled
            
            
            return loc, vel, loc_all, vel_all, all_edges_sampled, all_edges, ga, gr, sampled_indices
