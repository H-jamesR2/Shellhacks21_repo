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
