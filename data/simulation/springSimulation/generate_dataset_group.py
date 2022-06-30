from spring_sim_group import *
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-sim", type=int, default=2500)
#parser.add_argument("--num-valid", type=int, default=600)
#parser.add_argument("--num-test", type=int, default=600)
parser.add_argument("--length", type=int, default=5000, help="length of trajectory.")
parser.add_argument("--sample-freq", type=int, default=100, 
                    help="How often to sample the trajectory.")
parser.add_argument("--n-balls", type=int, default=10, help="Number of balls in the simulation.")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--dynamic", action="store_true", default=False, help="whether generate dynamic groups")
parser.add_argument("--age-factor", type=int, default=0.0001, help="age factor")
parser.add_argument("--ga-values-factor", type=int, default=5, help="group assignment value factor")
parser.add_argument("--K", type=float, default=3.0, help="K")
parser.add_argument("--b", type=float, default= 0.05, help="b")

args = parser.parse_args()
print(args)

suffix = "_group_static_"
if args.dynamic:
    suffix = "_group_dynamic_"
    
sim = SpringSim(n_balls = args.n_balls, age_factor=args.age_factor, 
                ga_values_factor=args.ga_values_factor, K = args.K, b=args.b,
                dynamic=args.dynamic)

suffix += str(args.n_balls)
suffix += '_'
suffix += str(int(args.K))
suffix += '_'
suffix += str(int(args.b*100))


np.random.seed(args.seed)

print(suffix)


def generate_dataset(num_sims, length, sample_freq):
    loc_sampled_all = list() #shape: [num_sims, num_sampledTimesteps, num_features, num_atoms]
    vel_sampled_all = list() #shape: [num_sims, num_sampledTimesteps, num_features, num_atoms]
    loc_all = list() #shape: [num_sims, num_timesteps, num_features, num_atoms]
    vel_all = list() #shape: [num_sims, num_timesteps, num_features, num_atoms]
    sampled_indices_all = list() #shape: [num_sims, num_sampledTimesteps]
    edges_sampled_all = list() #shape: [num_sims, num_sampledTimesteps, num_atoms, num_atoms]
    edges_all = list() #shape: [num_sims, num_timesteps, num_atoms, num_atoms]
    #group assignment tensor
    ga_all = list() #shape: [num_sims, (num_sampledTimesteps), num_atoms]
    #group relationship tensor
    gr_all = list() #shape: [num_sims, (num_sampledTimesteps), num_atoms, num_atoms]
    
    for i in range(num_sims):
        t = time.time()
        # return vectors of one simulation
        loc_sampled, vel_sampled, loc, vel, edges_sampled, edges, ga, gr, sampled_indices = sim.sample_trajectory(T=length, 
                                                                                                                  sample_freq=sample_freq)
        if i% 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time()-t))
        
        
        loc_sampled_all.append(loc_sampled)
        vel_sampled_all.append(vel_sampled)
        loc_all.append(loc)
        vel_all.append(vel)
        sampled_indices_all.append(sampled_indices)
        edges_sampled_all.append(edges_sampled)
        edges_all.append(edges)
        ga_all.append(ga)
        gr_all.append(gr)
        
    loc_sampled_all = np.stack(loc_sampled_all)
    vel_sampled_all = np.stack(vel_sampled_all)
    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_sampled_all = np.stack(edges_sampled_all)
    edges_all = np.stack(edges_all)
    ga_all = np.stack(ga_all)
    gr_all = np.stack(gr_all)
    sampled_indices_all = np.stack(sampled_indices_all)
    
    return loc_sampled_all, vel_sampled_all, loc_all, vel_all, edges_sampled_all, edges_all, ga_all, gr_all, sampled_indices_all



print("Generating {} simulations".format(args.num_sim))
loc_sampled_all_sim, vel_sampled_all_sim, loc_all_sim, vel_all_sim, edges_all_sampled_sim, edges_all_sim, ga_sim, gr_sim, sampled_indices_all_sim = generate_dataset(args.num_sim,
                                                                                                                           args.length,
                                                                                                                           args.sample_freq)
#print("Generating {} validation simulations".format(args.num_valid))
#loc_sampled_all_valid, vel_sampled_all_valid, loc_all_valid, vel_all_valid,  edges_all_sampled_valid, edges_all_valid, ga_valid, gr_valid, sampled_indices_all_valid = generate_dataset(args.num_valid,
#                                                                                                                           args.length,
#                                                                                                                           args.sample_freq)


#print("Generating {} test simulations".format(args.num_test))
#loc_sampled_all_test, vel_sampled_all_test, loc_all_test, vel_all_test, edges_all_sampled_test, edges_all_test, ga_test, gr_test, sampled_indices_all_test = generate_dataset(args.num_test,
#                                                                                                                           args.length,
#                                                                                                                           args.sample_freq)
#



np.save('loc_sampled_all_sim' + suffix + '.npy', loc_sampled_all_sim)
np.save('vel_sampled_all_sim' + suffix + '.npy', vel_sampled_all_sim)
#np.save('loc_all_train' + suffix + '.npy', loc_all_train)
#np.save('vel_all_train' + suffix + '.npy', vel_all_train)
#np.save("edges_all_train"+suffix+'.npy', edges_all_train)
#np.save("sampled_indices_all_train"+suffix+'.npy', sampled_indices_all_train)


#np.save('loc_sampled_all_valid' + suffix + '.npy', loc_sampled_all_valid)
#np.save('vel_sampled_all_valid' + suffix + '.npy', vel_sampled_all_valid)
#np.save('loc_all_valid' + suffix + '.npy', loc_all_valid)
#np.save('vel_all_valid' + suffix + '.npy', vel_all_valid)
#np.save("edges_all_valid"+suffix+'.npy', edges_all_valid)
#np.save("sampled_indices_all_valid"+suffix+'.npy', sampled_indices_all_valid)

#np.save('loc_sampled_all_test' + suffix + '.npy', loc_sampled_all_test)
#np.save('vel_sampled_all_test' + suffix + '.npy', vel_sampled_all_test)
#np.save('loc_all_test' + suffix + '.npy', loc_all_test)
#np.save('vel_all_test' + suffix + '.npy', vel_all_test)
#np.save("edges_all_test"+suffix+'.npy', edges_all_test)
#np.save("sampled_indices_all_test"+suffix+'.npy', sampled_indices_all_test)

np.save("edges_sampled_all_sim"+suffix+'.npy', edges_all_sampled_sim)
#np.save("edges_sampled_all_valid"+suffix+'.npy', edges_all_sampled_valid)
#np.save("edges_sampled_all_test"+suffix+'.npy', edges_all_sampled_test)

np.save("ga_sim"+suffix+'.npy', ga_sim)
#np.save("ga_valid"+suffix+'.npy', ga_valid)
#np.save("ga_test"+suffix+'.npy', ga_test)

np.save("gr_sim"+suffix+'.npy', gr_sim)
#np.save("gr_valid"+suffix+'.npy', gr_valid)
#np.save("gr_test"+suffix+'.npy', gr_test)

    
        

        
        
        
        
        
    
    

