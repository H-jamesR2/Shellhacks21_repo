import json
import numpy as np

from ntlk_utils import tokenize, stem, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#open a json file then load
with open('trainingData.json', 'r') as f:
    intents = json.load(f)

#print(intents)
all_words = []
tags = []
xy = []
#training data from json file...
# function_loops:
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append(w, tag)

ignore_words = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = [] 
Y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

#turn into numpy arrays..
X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    #dataset(idx)
    def __getitem__(self, index):
        return self.x_data[i], self.y_data[i]
    def __len__(self):
        return self.n_samples
    
# 
batch_size = 8

dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
    shuffle=True, num_workers=2)