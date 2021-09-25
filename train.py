import json
from ntlk_utils import tokenize, stem, bag_of_words

#open a json file then load
with open('intent.json', 'r') as f:
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
all_words = [stem(w) fro w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = [] 
Y_train = []
for (pattern_sentence, tag) in xy:

