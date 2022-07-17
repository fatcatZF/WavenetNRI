import pickle
import os

import utils as u
from utils import *
from models_solera import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, default="students03",
                    help='data folder name.')
parser.add_argument('--split', type=str, default="split10", help='splits of data.')

args = parser.parse_args()


data_path = os.path.join("data/pedestrian", args.data_folder)
data_path = os.path.join(data_path, args.split)

#data_path = os.path.join(data_folder, split)
print("data_path: ", data_path)




train_path = os.path.join(data_path, "examples_train_unnormalized.pkl")
print("train_path: ", train_path)
valid_path = os.path.join(data_path, "examples_valid_unnormalized.pkl")
print("valid_path: ", valid_path)
test_path = os.path.join(data_path, "examples_test_unnormalized.pkl")
print("test_path: ", test_path)



#load training, validation and test data
with open(train_path, 'rb') as f:
    examples_train = pickle.load(f)
with open(valid_path, 'rb') as f:
    examples_valid = pickle.load(f)
with open(test_path, 'rb') as f:
    examples_test = pickle.load(f)
    
    
#create ground
ground = build_ground(examples_train)



pairwise_features_train = []
pairwise_features_valid = []
pairwise_features_test = []

for example in examples_train:
    _,_,pairwise_features = compute_sims(example, ground)
    pairwise_features_train.append(pairwise_features)
    
for example in examples_valid:
    _,_,pairwise_features = compute_sims(example, ground)
    pairwise_features_valid.append(pairwise_features)
    
for example in examples_test:
    _,_,pairwise_features = compute_sims(example, ground)
    pairwise_features_test.append(pairwise_features)
    
    


with open(os.path.join(data_path, "pairwise_features_train.pkl"), 'wb') as f:
    pickle.dump(pairwise_features_train, f)

with open(os.path.join(data_path, "pairwise_features_valid.pkl"), 'wb') as f:
    pickle.dump(pairwise_features_valid, f)
    
with open(os.path.join(data_path, "pairwise_features_test.pkl"), 'wb') as f:
    pickle.dump(pairwise_features_test, f)







